#include "RCGMP/IR/RCGMPBridgeOps.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>

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
