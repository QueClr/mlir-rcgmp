#include "RCGMP/IR/RCGMPDialect.h"
#include "RCGMP/IR/RCGMPTypes.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/DialectImplementation.h>

#define GET_TYPEDEF_CLASSES
#include "RCGMP/IR/RCGMPTypes.cpp.inc"

#include "RCGMP/IR/RCGMPDialect.cpp.inc"

namespace mlir::rcgmp {

void RCGMPDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "RCGMP/IR/RCGMPTypes.cpp.inc"
      >();
}

} // namespace mlir::rcgmp
