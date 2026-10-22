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

LogicalResult MpzToBitsOp::verify() {
  auto width =
      DataLayout::closest(*this).getTypeSizeInBits(getValue().getType());
  if (getType().getWidth() != width.getFixedValue())
    return emitOpError("requires an integer matching the handle bit width");
  return success();
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
