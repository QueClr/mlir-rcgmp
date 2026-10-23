#include "RCGMP/IR/RCGMPDialect.h"
#include "RCGMP/IR/RCGMPBridgeOps.h"
#include "RCGMP/IR/RCGMPOps.h"
#include "RCGMP/IR/RCGMPTypes.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/DialectImplementation.h>

#define GET_TYPEDEF_CLASSES
#include "RCGMP/IR/RCGMPTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "RCGMP/IR/RCGMPAttrs.cpp.inc"

#include "RCGMP/IR/RCGMPDialect.cpp.inc"

namespace mlir::rcgmp {

void RCGMPDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "RCGMP/IR/RCGMPOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "RCGMP/IR/RCGMPBridgeOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "RCGMP/IR/RCGMPTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "RCGMP/IR/RCGMPAttrs.cpp.inc"
      >();
}

Operation *RCGMPDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, Type type,
                                             Location loc) {
  if (isa<IntegerType>(type))
    return arith::ConstantOp::materialize(builder, value, type, loc);
  return nullptr;
}

} // namespace mlir::rcgmp
