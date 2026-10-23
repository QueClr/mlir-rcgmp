#include "RCGMP/IR/RCGMPBridgeOps.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>

#define GET_OP_CLASSES
#include "RCGMP/IR/RCGMPBridgeOps.cpp.inc"

namespace mlir::rcgmp::bridge {

bool fitsImmediate(const APInt &value, unsigned handleWidth) {
  // Signed W-2-bit payloads are immediate. A zero-width value is zero.
  return value.getBitWidth() == 0 ||
         value.getSignificantBits() <= handleWidth - 2;
}

// Checks that an operation reading the handle of value produces an integer of
// the handle width.
static LogicalResult verifyHandleWidth(Operation *op, Value value,
                                       IntegerType type) {
  auto width = DataLayout::closest(op).getTypeSizeInBits(value.getType());
  if (type.getWidth() != width.getFixedValue())
    return op->emitOpError("requires an integer matching the handle bit width");
  return success();
}

OpFoldResult MpzToBitsOp::fold(FoldAdaptor) {
  if (auto from = getValue().getDefiningOp<MpzFromBitsOp>())
    return from.getValue();
  return {};
}

OpFoldResult MpzFromBitsOp::fold(FoldAdaptor) {
  if (auto to = getValue().getDefiningOp<MpzToBitsOp>();
      to && to.getValue().getType() == getType())
    return to.getValue();
  return {};
}

// Whether bits are a constant, or the bitwise OR with a constant, whose low
// bit, the immediate tag, is set.
static bool hasImmediateTag(Value bits) {
  auto tagged = [](Value value) {
    APInt constant;
    return matchPattern(value, m_ConstantInt(&constant)) && constant[0];
  };
  if (auto ori = bits.getDefiningOp<arith::OrIOp>())
    return tagged(ori.getLhs()) || tagged(ori.getRhs());
  return tagged(bits);
}

OpFoldResult MpzIsImmediateOp::fold(FoldAdaptor) {
  Value value = getValue();
  std::optional<bool> immediate;
  if (value.getDefiningOp<MpzFromImmediateOp>())
    immediate = true;
  else if (value.getDefiningOp<MpzPromoteOp>() ||
           value.getDefiningOp<MpzImmortalOp>())
    immediate = false;
  else if (auto from = value.getDefiningOp<MpzFromBitsOp>()) {
    APInt bits;
    if (matchPattern(from.getValue(), m_ConstantInt(&bits)))
      immediate = bits[0];
    else if (hasImmediateTag(from.getValue()))
      immediate = true;
  }
  if (!immediate)
    return {};
  return BoolAttr::get(getContext(), *immediate);
}

namespace {

// Whether bits are the handle bits of value, or an AND of bits including
// them.
bool includesBits(Value bits, Value value) {
  if (auto to = bits.getDefiningOp<MpzToBitsOp>())
    return to.getValue() == value;
  if (auto both = bits.getDefiningOp<arith::AndIOp>())
    return includesBits(both.getLhs(), value) ||
           includesBits(both.getRhs(), value);
  return false;
}

// Whether condition is the tag test of value, or with conjunctions, a
// conjunction of tests that includes it. Fast paths test several handles at
// once as ((bits & ...) & 1) == 1.
bool testsTag(Value condition, Value value, bool conjunctions) {
  if (auto test = condition.getDefiningOp<MpzIsImmediateOp>())
    return test.getValue() == value;
  if (!conjunctions)
    return false;
  if (auto both = condition.getDefiningOp<arith::AndIOp>())
    return testsTag(both.getLhs(), value, true) ||
           testsTag(both.getRhs(), value, true);
  auto equal = condition.getDefiningOp<arith::CmpIOp>();
  if (!equal || equal.getPredicate() != arith::CmpIPredicate::eq ||
      !matchPattern(equal.getRhs(), m_One()))
    return false;
  auto tag = equal.getLhs().getDefiningOp<arith::AndIOp>();
  return tag && matchPattern(tag.getRhs(), m_One()) &&
         includesBits(tag.getLhs(), value);
}

// A tag test is true in the then region of an scf.if whose condition
// requires it, and false in the else region of one whose condition is it.
struct KnownTagTest : OpRewritePattern<MpzIsImmediateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MpzIsImmediateOp op,
                                PatternRewriter &rewriter) const override {
    Value value = op.getValue();
    for (Operation *parent = op->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (parent->hasTrait<OpTrait::IsIsolatedFromAbove>())
        break;
      auto branch = dyn_cast<scf::IfOp>(parent);
      if (!branch)
        continue;
      bool then = branch.getThenRegion().isAncestor(op->getParentRegion());
      if (testsTag(branch.getCondition(), value, then)) {
        rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(op, then, 1);
        return success();
      }
    }
    return failure();
  }
};

} // namespace

void MpzIsImmediateOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                   MLIRContext *context) {
  patterns.add<KnownTagTest>(context);
}

LogicalResult MpzToBitsOp::verify() {
  return verifyHandleWidth(*this, getValue(), getType());
}

LogicalResult MpzSizeOp::verify() {
  return verifyHandleWidth(*this, getValue(), getType());
}

LogicalResult MpzLimbOp::verify() {
  return verifyHandleWidth(*this, getValue(), getType());
}

LogicalResult MpzImmortalOp::verify() {
  auto type = dyn_cast<IntegerType>(getValueAttr().getType());
  if (!type || !type.isSignless())
    return emitOpError("requires a signless integer attribute");
  auto width = DataLayout::closest(*this).getTypeSizeInBits(getType());
  if (fitsImmediate(getValue(), width.getFixedValue()))
    return emitOpError("requires a value outside the immediate range");
  return success();
}

LogicalResult MpzFromBitsOp::verify() {
  auto width = DataLayout::closest(*this).getTypeSizeInBits(getType());
  if (cast<IntegerType>(getValue().getType()).getWidth() !=
      width.getFixedValue())
    return emitOpError("requires an integer matching the handle bit width");
  return success();
}

} // namespace mlir::rcgmp::bridge
