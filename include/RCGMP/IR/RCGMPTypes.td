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
  }];
  let parameters = (ins DefaultValuedParameter<"bool", "false">:$atomic);
  let hasCustomAssemblyFormat = 1;
}

#endif
