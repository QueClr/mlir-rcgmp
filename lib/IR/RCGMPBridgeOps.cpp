#include "RCGMP/IR/RCGMPBridgeOps.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>

#define GET_OP_CLASSES
#include "RCGMP/IR/RCGMPBridgeOps.cpp.inc"

namespace mlir::rcgmp::bridge {

LogicalResult MpzToBitsOp::verify() {
  auto width =
      DataLayout::closest(*this).getTypeSizeInBits(getValue().getType());
  if (getType().getWidth() != width.getFixedValue())
    return emitOpError("requires an integer matching the handle bit width");
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
