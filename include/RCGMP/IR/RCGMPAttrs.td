#ifndef RCGMP_IR_ATTRS_TD
#define RCGMP_IR_ATTRS_TD

include "RCGMP/IR/RCGMPDialect.td"
include "RCGMP/IR/RCGMPEnums.td"
include "mlir/IR/AttrTypeBase.td"

def RCGMP_ArmAttr : AttrDef<RCGMPDialect, "Arm"> {
  let mnemonic = "arm";
  let summary = "The integers that one arm of rcgmp.match selects";
  let description = [{
    A `case` arm selects each of its values, an `at_least` arm selects the
    integers at least its single value, and a `default` arm, which has no
    values, selects every integer. Each value is an integer attribute of the
    narrowest signless type holding it in two's complement, as for
    `rcgmp.constant`.
  }];
  let parameters = (ins EnumParameter<RCGMP_ArmKind>:$kind,
                        "ArrayAttr":$values);
  let assemblyFormat = "`<` $kind `,` $values `>`";
  let genVerifyDecl = 1;
}

#endif
