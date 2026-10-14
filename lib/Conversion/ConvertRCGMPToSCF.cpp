#include "RCGMP/Conversion/ConvertRCGMPToSCF.h"
#include "RCGMP/IR/RCGMPBridgeOps.h"
#include "RCGMP/IR/RCGMPOps.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
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

struct ConvertRCGMPToSCFPass
    : PassWrapper<ConvertRCGMPToSCFPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertRCGMPToSCFPass)
  StringRef getArgument() const override { return "convert-rcgmp-to-scf"; }
  StringRef getDescription() const override {
    return "Expand rcgmp operations into SCF and low-level bridge operations";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<RCGMPDialect, scf::SCFDialect, arith::ArithDialect,
                    cf::ControlFlowDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FromIntLowering, RetainLowering, ReleaseLowering>(
        &getContext());
    ConversionTarget target(getContext());
    target.addIllegalOp<FromIntOp, RetainOp, ReleaseOp>();
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
