#ifndef RCGMP_IR_TYPES_TD
#define RCGMP_IR_TYPES_TD

include "RCGMP/IR/RCGMPDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"

def RCGMP_MpzType : TypeDef<RCGMPDialect, "Mpz", [
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface,
                                ["getPreferredAlignment"]>]> {
  let mnemonic = "mpz";
  let summary = "Reference-counted arbitrary-precision signed integer";
  let description = [{
    `!rcgmp.mpz` uses nonatomic reference counting; `!rcgmp.mpz<atomic>`
    uses atomic reference counting. Atomic reference counting governs the
    lifetime of shared allocations, not concurrent mutation of their value.

    Both modes lower to an opaque `!llvm.ptr` in address space zero. The type's
    size and alignment follow that pointer's data layout. The handle may encode
    an immediate integer; its pointer representation does not imply that it is
    dereferenceable.

    With `W` the handle width from the data layout, the immediate range is
    `[-2^(W-3), 2^(W-3) - 1]`. Every value is canonical: an integer `n` in the
    immediate range is always the immediate whose `W`-bit two's-complement bits
    are `4*n + 1`, and a heap integer holds a value outside that range without
    leading zero limbs.
  }];
  let parameters = (ins DefaultValuedParameter<"bool", "false">:$atomic);
  let hasCustomAssemblyFormat = 1;
}

#endif
