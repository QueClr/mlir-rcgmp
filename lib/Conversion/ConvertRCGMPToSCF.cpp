#include "RCGMP/Conversion/ConvertRCGMPToSCF.h"
#include "FastPathLowering.h"
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

struct ConstantLowering : OpRewritePattern<ConstantOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto type = op.getType();
    unsigned width =
        DataLayout::closest(op).getTypeSizeInBits(type).getFixedValue();
    APInt value = op.getValue();
    if (!bridge::fitsImmediate(value, width)) {
      rewriter.replaceOpWithNewOp<bridge::MpzImmortalOp>(op, type,
                                                         op.getValueAttr());
      return success();
    }
    APInt tagged = value.sextOrTrunc(width).shl(2);
    tagged.setBit(0);
    auto bits = arith::ConstantOp::create(
        rewriter, op.getLoc(),
        rewriter.getIntegerAttr(rewriter.getIntegerType(width), tagged));
    rewriter.replaceOpWithNewOp<bridge::MpzFromBitsOp>(op, type, bits);
    return success();
  }
};

// Move the insertion point into a branch taken only for counted heap values.
// Immediate and immortal values have no reference count to update.
void enterCountedBranch(PatternRewriter &rewriter, Location loc, Value value) {
  auto immediate = bridge::MpzIsImmediateOp::create(rewriter, loc, value);
  auto heap = scf::IfOp::create(rewriter, loc, immediate, true);
  rewriter.setInsertionPointToStart(&heap.getElseRegion().front());
  auto immortal = bridge::MpzIsImmortalOp::create(rewriter, loc, value);
  auto counted = scf::IfOp::create(rewriter, loc, immortal, true);
  rewriter.setInsertionPointToStart(&counted.getElseRegion().front());
}

struct RetainLowering : OpRewritePattern<RetainOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RetainOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    enterCountedBranch(rewriter, loc, op.getValue());
    auto overflow = bridge::MpzIncRefOp::create(rewriter, loc, op.getValue());
    auto one = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
    auto valid = arith::XOrIOp::create(rewriter, loc, overflow, one);
    cf::AssertOp::create(rewriter, loc, valid,
                         "rcgmp reference count overflow");
    rewriter.eraseOp(op);
    return success();
  }
};

struct ReleaseLowering : OpRewritePattern<ReleaseOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReleaseOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    enterCountedBranch(rewriter, loc, op.getValue());
    auto last = bridge::MpzDecRefOp::create(rewriter, loc, op.getValue());
    auto destroy = scf::IfOp::create(rewriter, loc, last, false);
    rewriter.setInsertionPointToStart(&destroy.getThenRegion().front());
    bridge::MpzFreeOp::create(rewriter, loc, op.getValue());
    rewriter.eraseOp(op);
    return success();
  }
};

struct AddILowering : FastPathLowering<AddILowering> {
  using FastPathLowering::FastPathLowering;
  using Op = AddIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_add_atomic" : "rcgmp_mpz_add";
  }

  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *, ValueRange operands) {
    Value lhs = operands[0], rhs = operands[1];
    unsigned width = cast<IntegerType>(lhs.getType()).getWidth();
    auto checked = bridge::AddSIExtendedOp::create(
        rewriter, loc, lhs.getType(), rewriter.getI1Type(), lhs, rhs);
    Value sum = checked.getSum();
    auto one = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
    auto fits =
        arith::XOrIOp::create(rewriter, loc, checked.getOverflow(), one);
    // 4*x+1 + 4*y+1 has bit 1 set iff both handles are immediate.
    auto two = arith::ConstantIntOp::create(rewriter, loc, 2, width);
    auto tag = arith::AndIOp::create(rewriter, loc, sum, two);
    auto both = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                      tag, two);
    auto valid = arith::AndIOp::create(rewriter, loc, fits, both);
    auto three = arith::ConstantIntOp::create(rewriter, loc, 3, width);
    auto tagged = arith::XOrIOp::create(rewriter, loc, sum, three);
    return {tagged, valid};
  }
};

struct SubILowering : FastPathLowering<SubILowering> {
  using FastPathLowering::FastPathLowering;
  using Op = SubIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_sub_atomic" : "rcgmp_mpz_sub";
  }

  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *, ValueRange operands) {
    Value lhs = operands[0], rhs = operands[1];
    auto checked = bridge::SubSIExtendedOp::create(
        rewriter, loc, lhs.getType(), rewriter.getI1Type(), lhs, rhs);
    auto yes = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
    auto fits =
        arith::XOrIOp::create(rewriter, loc, checked.getOverflow(), yes);
    auto both = allImmediate(rewriter, loc, {lhs, rhs});
    auto valid = arith::AndIOp::create(rewriter, loc, fits, both);
    auto one = arith::ConstantIntOp::create(
        rewriter, loc, 1, cast<IntegerType>(lhs.getType()).getWidth());
    // Subtracting tagged immediates yields 4*(x-y); restore the low tag bit.
    auto tagged = arith::OrIOp::create(rewriter, loc, checked.getDiff(), one);
    return {tagged, valid};
  }
};

struct MulILowering : FastPathLowering<MulILowering> {
  using FastPathLowering::FastPathLowering;
  using Op = MulIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_mul_atomic" : "rcgmp_mpz_mul";
  }

  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *, ValueRange operands) {
    Value lhs = operands[0], rhs = operands[1];
    unsigned width = cast<IntegerType>(lhs.getType()).getWidth();
    auto two = arith::ConstantIntOp::create(rewriter, loc, 2, width);
    auto mask = arith::ConstantIntOp::create(rewriter, loc, -4, width);
    auto x = arith::ShRSIOp::create(rewriter, loc, lhs, two);
    auto scaledY = arith::AndIOp::create(rewriter, loc, rhs, mask);
    // x * (4*y) fits the signed word iff x*y fits the immediate payload.
    // The checked product remains defined for arbitrary heap-handle bits.
    auto product = bridge::MulSIOverflowOp::create(
        rewriter, loc, lhs.getType(), rewriter.getI1Type(), x, scaledY);
    auto yes = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
    auto fits =
        arith::XOrIOp::create(rewriter, loc, product.getOverflow(), yes);
    auto both = allImmediate(rewriter, loc, {lhs, rhs});
    auto valid = arith::AndIOp::create(rewriter, loc, fits, both);
    auto one = arith::ConstantIntOp::create(rewriter, loc, 1, width);
    auto tagged =
        arith::OrIOp::create(rewriter, loc, product.getProduct(), one);
    return {tagged, valid};
  }
};

struct AndILowering : FastPathLowering<AndILowering> {
  using FastPathLowering::FastPathLowering;
  using Op = AndIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_and_atomic" : "rcgmp_mpz_and";
  }

  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *, ValueRange operands) {
    Value lhs = operands[0], rhs = operands[1];
    auto bits = arith::AndIOp::create(rewriter, loc, lhs, rhs);
    return {bits, allImmediate(rewriter, loc, {lhs, rhs})};
  }
};

struct XOrILowering : FastPathLowering<XOrILowering> {
  using FastPathLowering::FastPathLowering;
  using Op = XOrIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_xor_atomic" : "rcgmp_mpz_xor";
  }

  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *, ValueRange operands) {
    Value lhs = operands[0], rhs = operands[1];
    auto bits = arith::XOrIOp::create(rewriter, loc, lhs, rhs);
    auto one = arith::ConstantIntOp::create(
        rewriter, loc, 1, cast<IntegerType>(lhs.getType()).getWidth());
    auto tagged = arith::OrIOp::create(rewriter, loc, bits, one);
    return {tagged, allImmediate(rewriter, loc, {lhs, rhs})};
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
    patterns.add<ConstantLowering, FromIntLowering, RetainLowering,
                 ReleaseLowering, AddILowering, SubILowering, MulILowering,
                 AndILowering, XOrILowering>(&getContext());
    ConversionTarget target(getContext());
    target.addIllegalOp<ConstantOp, FromIntOp, RetainOp, ReleaseOp, AddIOp,
                        SubIOp, MulIOp, AndIOp, XOrIOp>();
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
