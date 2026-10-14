#include "RCGMP/Conversion/ConvertRCGMPToSCF.h"
#include "RCGMP/IR/RCGMPBridgeOps.h"
#include "RCGMP/IR/RCGMPOps.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::rcgmp {
namespace {

struct FromIntLowering : OpRewritePattern<FromIntOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(FromIntOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto fits = bridge::MpzFitsImmediateOp::create(rewriter, loc, op.getValue(),
                                                   op.getIsUnsigned());
    auto result =
        scf::IfOp::create(rewriter, loc, TypeRange{op.getType()}, fits,
                          /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(&result.getThenRegion().front());
    auto immediate = bridge::MpzFromImmediateOp::create(
        rewriter, loc, op.getType(), op.getValue(), op.getIsUnsigned());
    scf::YieldOp::create(rewriter, loc, immediate.getResult());
    rewriter.setInsertionPointToStart(&result.getElseRegion().front());
    auto heap = bridge::MpzPromoteOp::create(rewriter, loc, op.getType(),
                                             op.getValue(), op.getIsUnsigned());
    scf::YieldOp::create(rewriter, loc, heap.getResult());
    rewriter.replaceOp(op, result.getResults());
    return success();
  }
};

struct RetainLowering : OpRewritePattern<RetainOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RetainOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto immediate =
        bridge::MpzIsImmediateOp::create(rewriter, loc, op.getValue());
    auto branch = scf::IfOp::create(rewriter, loc, immediate, true);
    rewriter.setInsertionPointToStart(&branch.getElseRegion().front());
    auto overflow = bridge::MpzIncRefOp::create(rewriter, loc, op.getValue());
    auto one = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
    auto valid = arith::XOrIOp::create(rewriter, loc, overflow, one);
    cf::AssertOp::create(rewriter, loc, valid,
                         "rcgmp reference count overflow");
    rewriter.replaceOp(op, op.getValue());
    return success();
  }
};

struct ReleaseLowering : OpRewritePattern<ReleaseOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReleaseOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto immediate =
        bridge::MpzIsImmediateOp::create(rewriter, loc, op.getValue());
    auto branch = scf::IfOp::create(rewriter, loc, immediate, true);
    rewriter.setInsertionPointToStart(&branch.getElseRegion().front());
    auto last = bridge::MpzDecRefOp::create(rewriter, loc, op.getValue());
    auto destroy = scf::IfOp::create(rewriter, loc, last, false);
    rewriter.setInsertionPointToStart(&destroy.getThenRegion().front());
    bridge::MpzFreeOp::create(rewriter, loc, op.getValue());
    rewriter.eraseOp(op);
    return success();
  }
};

// Keep the heap precondition visible in SCF; uniqueness is never tested on
// tags.
Value canReuse(PatternRewriter &rewriter, Location loc, Value value) {
  auto immediate = bridge::MpzIsImmediateOp::create(rewriter, loc, value);
  auto branch = scf::IfOp::create(
      rewriter, loc, TypeRange{rewriter.getI1Type()}, immediate, true);
  rewriter.setInsertionPointToStart(&branch.getThenRegion().front());
  auto no = arith::ConstantIntOp::create(rewriter, loc, 0, 1);
  scf::YieldOp::create(rewriter, loc, no.getResult());
  rewriter.setInsertionPointToStart(&branch.getElseRegion().front());
  auto unique = bridge::MpzIsUniqueOp::create(rewriter, loc, value);
  scf::YieldOp::create(rewriter, loc, unique.getResult());
  rewriter.setInsertionPointAfter(branch);
  return branch.getResult(0);
}

struct AddILowering : OpRewritePattern<AddIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AddIOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // Separate symbols preserve the reference-count type in func.call while
    // both lower to the same pointer ABI. The C entry points share an
    // implementation.
    auto module = op->getParentOfType<ModuleOp>();
    StringRef name = cast<MpzType>(op.getType()).getAtomic()
                         ? "rcgmp_mpz_add_atomic"
                         : "rcgmp_mpz_add";
    auto signature = rewriter.getFunctionType(
        {op.getType(), op.getType(), rewriter.getI32Type()}, {op.getType()});
    auto function = module.lookupSymbol<func::FuncOp>(name);
    if (module.lookupSymbol(name)) {
      if (!function || function.getFunctionType() != signature)
        return rewriter.notifyMatchFailure(op,
                                           "incompatible runtime declaration");
    } else {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      function = func::FuncOp::create(rewriter, loc, name, signature);
      function.setPrivate();
    }
    unsigned width =
        DataLayout::closest(op).getTypeSizeInBits(op.getType()).getFixedValue();
    auto word = rewriter.getIntegerType(width);
    auto lhs = bridge::MpzToBitsOp::create(rewriter, loc, word, op.getLhs());
    auto rhs = bridge::MpzToBitsOp::create(rewriter, loc, word, op.getRhs());
    // This sum must wrap: overflow is a slow-path condition, not undefined
    // behavior.
    auto sum = arith::AddIOp::create(rewriter, loc, lhs, rhs);
    // Signed overflow iff both operand signs differ from the sum's sign.
    auto lhsChanged = arith::XOrIOp::create(rewriter, loc, lhs, sum);
    auto rhsChanged = arith::XOrIOp::create(rewriter, loc, rhs, sum);
    auto changed = arith::AndIOp::create(rewriter, loc, lhsChanged, rhsChanged);
    auto zero = arith::ConstantIntOp::create(rewriter, loc, 0, width);
    auto fits = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sge,
                                      changed, zero);
    // 4*x+1 + 4*y+1 has bit 1 set iff both handles are immediate.
    auto two = arith::ConstantIntOp::create(rewriter, loc, 2, width);
    auto tag = arith::AndIOp::create(rewriter, loc, sum, two);
    auto both = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                      tag, two);
    auto valid = arith::AndIOp::create(rewriter, loc, fits, both);
    auto branch =
        scf::IfOp::create(rewriter, loc, TypeRange{op.getType()}, valid, true);
    rewriter.setInsertionPointToStart(&branch.getThenRegion().front());
    auto three = arith::ConstantIntOp::create(rewriter, loc, 3, width);
    auto tagged = arith::XOrIOp::create(rewriter, loc, sum, three);
    auto fast =
        bridge::MpzFromBitsOp::create(rewriter, loc, op.getType(), tagged);
    scf::YieldOp::create(rewriter, loc, fast.getResult());
    rewriter.setInsertionPointToStart(&branch.getElseRegion().front());
    auto reuseLhs = canReuse(rewriter, loc, op.getLhs());
    auto reuseRhs = canReuse(rewriter, loc, op.getRhs());
    auto lhsFlag =
        arith::ExtUIOp::create(rewriter, loc, rewriter.getI32Type(), reuseLhs);
    auto rhsFlag =
        arith::ExtUIOp::create(rewriter, loc, rewriter.getI32Type(), reuseRhs);
    auto one = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
    // rhsFlag is zero or one; shifting by one cannot overflow either range.
    auto shifted = arith::ShLIOp::create(rewriter, loc, rhsFlag, one,
                                         arith::IntegerOverflowFlags::nuw |
                                             arith::IntegerOverflowFlags::nsw);
    auto flags = arith::OrIOp::create(rewriter, loc, lhsFlag, shifted);
    auto slow = func::CallOp::create(
        rewriter, loc, function, ValueRange{op.getLhs(), op.getRhs(), flags});
    scf::YieldOp::create(rewriter, loc, slow.getResults());
    rewriter.replaceOp(op, branch.getResults());
    return success();
  }
};

struct ConvertRCGMPToSCFPass
    : PassWrapper<ConvertRCGMPToSCFPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertRCGMPToSCFPass)
  StringRef getArgument() const override { return "convert-rcgmp-to-scf"; }
  StringRef getDescription() const override {
    return "Expand rcgmp operations into SCF and low-level bridge operations";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<RCGMPDialect, scf::SCFDialect, arith::ArithDialect,
                    cf::ControlFlowDialect, func::FuncDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns
        .add<FromIntLowering, RetainLowering, ReleaseLowering, AddILowering>(
            &getContext());
    ConversionTarget target(getContext());
    target.addIllegalOp<FromIntOp, RetainOp, ReleaseOp, AddIOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createConvertRCGMPToSCFPass() {
  return std::make_unique<ConvertRCGMPToSCFPass>();
}

} // namespace mlir::rcgmp
