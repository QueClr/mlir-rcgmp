#include "RCGMP/IR/RCGMPDialect.h"
#include "RCGMP/IR/RCGMPBridgeOps.h"
#include "RCGMP/IR/RCGMPOps.h"
#include "RCGMP/IR/RCGMPTypes.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Transforms/InliningUtils.h>

#define GET_TYPEDEF_CLASSES
#include "RCGMP/IR/RCGMPTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "RCGMP/IR/RCGMPAttrs.cpp.inc"

#include "RCGMP/IR/RCGMPDialect.cpp.inc"

namespace mlir::rcgmp {

namespace {
// Every rcgmp operation may be inlined. Ownership travels with SSA values,
// which inlining preserves, and no operation refers to its enclosing function
// or a symbol. The regions of rcgmp.match end in rcgmp.yield, which returns
// from the match alone, so they need no terminator handling.
struct RCGMPInlinerInterface : DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

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
  addInterfaces<RCGMPInlinerInterface>();
}

Operation *RCGMPDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, Type type,
                                             Location loc) {
  if (auto mpz = dyn_cast<MpzValueAttr>(value); mpz && isa<MpzType>(type))
    return ConstantOp::create(builder, loc, type, mpz.getValue());
  if ((isa<IntegerAttr>(value) && isa<IntegerType, IndexType>(type)) ||
      (isa<FloatAttr>(value) && isa<FloatType>(type)))
    return arith::ConstantOp::materialize(builder, value, type, loc);
  return nullptr;
}

} // namespace mlir::rcgmp
