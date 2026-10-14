#ifndef RCGMP_IR_OP_BASE_TD
#define RCGMP_IR_OP_BASE_TD

include "RCGMP/IR/RCGMPTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class RCGMPOp<string name, list<Trait> traits = []>
    : Op<RCGMPDialect, name, traits>;

#endif
