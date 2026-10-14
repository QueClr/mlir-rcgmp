#ifndef RCGMP_IR_TYPES_TD
#define RCGMP_IR_TYPES_TD

include "RCGMP/IR/RCGMPDialect.td"
include "mlir/IR/AttrTypeBase.td"

def RCGMP_MpzType : TypeDef<RCGMPDialect, "Mpz"> {
  let mnemonic = "mpz";
  let summary = "Reference-counted arbitrary-precision signed integer";
  let description = [{
    `!rcgmp.mpz` uses nonatomic reference counting; `!rcgmp.mpz<atomic>`
    uses atomic reference counting. Atomic reference counting governs the
    lifetime of shared allocations, not concurrent mutation of their value.
  }];
  let parameters = (ins DefaultValuedParameter<"bool", "false">:$atomic);
  let hasCustomAssemblyFormat = 1;
}

#endif
