#ifndef RCGMP_CONVERSION_BINARY_LOWERING_H
#define RCGMP_CONVERSION_BINARY_LOWERING_H

#include "RCGMP/IR/RCGMPBridgeOps.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir::rcgmp {

struct BinaryFastPath {
  Value bits;
  Value valid;
};

// Derived supplies Op, getRuntimeName(atomic), and buildFastPath(rewriter, loc,
// lhsBits, rhsBits). The fast path must be safe to evaluate on any live handle;
// its result bits are interpreted as an immediate only when valid is true.
// The runtime consumes both operand references and accepts a two-bit reuse
// mask.
template <typename Derived> struct BinaryLowering : RewritePattern {
  explicit BinaryLowering(MLIRContext *context)
      : RewritePattern(Derived::Op::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *operation,
                                PatternRewriter &rewriter) const override {
    auto op = cast<typename Derived::Op>(operation);
    auto loc = op.getLoc();
    auto type = cast<MpzType>(op.getType());
    auto module = op->template getParentOfType<ModuleOp>();
    StringRef name = Derived::getRuntimeName(type.getAtomic());
    // Separate symbols keep atomic and nonatomic source signatures distinct.
    auto signature =
        rewriter.getFunctionType({type, type, rewriter.getI32Type()}, {type});
    auto function = module.template lookupSymbol<func::FuncOp>(name);
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
        DataLayout::closest(op).getTypeSizeInBits(type).getFixedValue();
    auto word = rewriter.getIntegerType(width);
    auto lhs = bridge::MpzToBitsOp::create(rewriter, loc, word, op.getLhs());
    auto rhs = bridge::MpzToBitsOp::create(rewriter, loc, word, op.getRhs());
    auto fast = Derived::buildFastPath(rewriter, loc, lhs, rhs);
    auto branch =
        scf::IfOp::create(rewriter, loc, TypeRange{type}, fast.valid, true);
    rewriter.setInsertionPointToStart(&branch.getThenRegion().front());
    auto immediate =
        bridge::MpzFromBitsOp::create(rewriter, loc, type, fast.bits);
    scf::YieldOp::create(rewriter, loc, immediate.getResult());
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

protected:
  static Value bothImmediate(PatternRewriter &rewriter, Location loc, Value lhs,
                             Value rhs) {
    auto common = arith::AndIOp::create(rewriter, loc, lhs, rhs);
    auto one = arith::ConstantIntOp::create(
        rewriter, loc, 1, cast<IntegerType>(lhs.getType()).getWidth());
    auto tag = arith::AndIOp::create(rewriter, loc, common, one);
    return arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq, tag,
                                 one);
  }

private:
  // A heap check must dominate each uniqueness load.
  static Value canReuse(PatternRewriter &rewriter, Location loc, Value value) {
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
};

} // namespace mlir::rcgmp

#endif
