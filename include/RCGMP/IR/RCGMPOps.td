#ifndef RCGMP_IR_OPS_TD
#define RCGMP_IR_OPS_TD

// Public operations with explicit ownership semantics.
include "RCGMP/IR/RCGMPOpBase.td"

def RCGMP_FromIntOp : RCGMPOp<"from_int"> {
  let arguments = (ins AnySignlessInteger:$value, UnitAttr:$isUnsigned);
  let results = (outs RCGMP_MpzType:$result);
  let assemblyFormat = "$value (`unsigned` $isUnsigned^)? attr-dict `:` type($value) `to` type($result)";

  let summary = "Create one owned integer from a fixed-width integer";
  let description = [{
    Interpret the input as signed two's complement unless `unsigned` is present.
    The result owns one reference. Inputs of any integer bit width are accepted.
  }];
}

def RCGMP_RetainOp : RCGMPOp<"retain", [SameOperandsAndResultType]> {
  let summary = "Borrow an integer and produce an additional owned reference";
  let arguments = (ins RCGMP_MpzType:$value);
  let results = (outs RCGMP_MpzType:$result);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

def RCGMP_ReleaseOp : RCGMPOp<"release"> {
  let summary = "Consume one owned reference";
  let arguments = (ins RCGMP_MpzType:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

#endif
