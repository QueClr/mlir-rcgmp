#ifndef RCGMP_IR_OP_BASE_TD
#define RCGMP_IR_OP_BASE_TD

include "RCGMP/IR/RCGMPTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class RCGMPOp<string name, list<Trait> traits = []>
    : Op<RCGMPDialect, name, traits>;

def RCGMP_NotSpeculatableImplTrait
    : NativeOpTrait<"NotSpeculatableImplTrait"> {
  let cppNamespace = "::mlir::rcgmp";
}

// Marks an operation that must not execute on a path where it did not
// originally execute, because it may abort or requires a live reference.
def RCGMP_NotSpeculatable : TraitList<[
    ConditionallySpeculatable, RCGMP_NotSpeculatableImplTrait]>;

#endif
