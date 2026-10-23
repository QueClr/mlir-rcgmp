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

def RCGMP_MpzValueAttr : AttrDef<RCGMPDialect, "MpzValue"> {
  let mnemonic = "value";
  let summary = "The value of a constant integer, as folding represents it";
  let description = [{
    Folding gives the value of a constant `!rcgmp.mpz` this attribute rather
    than a builtin integer attribute, whose type would differ from the value's:
    patterns that match builtin integer constants, such as zero or one, must
    not mistake it for a fixed-width integer.
  }];
  let parameters = (ins APIntParameter<"">:$value);
  let assemblyFormat = "`<` $value `>`";
}

#endif
