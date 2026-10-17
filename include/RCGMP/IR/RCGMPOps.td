#ifndef RCGMP_IR_OPS_TD
#define RCGMP_IR_OPS_TD

// Public operations with explicit ownership semantics.
include "RCGMP/IR/RCGMPOpBase.td"

// A consumed reference may be decremented, reused, or freed.
defvar RCGMP_ConsumedEffects = [MemRead, MemWrite, MemFree];

def RCGMP_FromIntOp : RCGMPOp<"from_int", [RCGMP_NotSpeculatable]> {
  let arguments = (ins AnySignlessInteger:$value, UnitAttr:$isUnsigned);
  let results = (outs Res<RCGMP_MpzType, "new integer", [MemAlloc]>:$result);
  let assemblyFormat = "$value (`unsigned` $isUnsigned^)? attr-dict `:` type($value) `to` type($result)";

  let summary = "Create one owned integer from a fixed-width integer";
  let description = [{
    Interpret the input as signed two's complement unless `unsigned` is present.
    The result owns one reference. Inputs of any integer bit width are accepted.
  }];
}

def RCGMP_RetainOp : RCGMPOp<"retain", [RCGMP_NotSpeculatable]> {
  let summary = "Add one owned reference to an integer";
  let description = [{
    The operand must own at least one reference and owns one more afterwards.
  }];
  let arguments = (ins Arg<RCGMP_MpzType, "retained integer",
                           [MemRead, MemWrite]>:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

def RCGMP_ReleaseOp : RCGMPOp<"release", [RCGMP_NotSpeculatable]> {
  let summary = "Consume one owned reference";
  let arguments = (ins Arg<RCGMP_MpzType, "released integer",
                           RCGMP_ConsumedEffects>:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

def RCGMP_AddIOp : RCGMPOp<"addi", [SameOperandsAndResultType, Commutative,
    RCGMP_NotSpeculatable]> {
  let summary = "Consume two integer references and produce their sum";
  let description = [{
    Each operand position consumes one owned reference. Adding a heap value to
    itself therefore requires two references, obtained with `rcgmp.retain`.
    The result owns one reference. A uniquely owned operand may be mutated and
    reused; shared operands retain their values. Integer addition is exact.
  }];
  let arguments = (ins
      Arg<RCGMP_MpzType, "consumed lhs", RCGMP_ConsumedEffects>:$lhs,
      Arg<RCGMP_MpzType, "consumed rhs", RCGMP_ConsumedEffects>:$rhs);
  let results = (outs Res<RCGMP_MpzType, "new integer", [MemAlloc]>:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

class RCGMP_BinaryOp<string name, string semantics, list<Trait> traits = []>
    : RCGMPOp<name, !listconcat([SameOperandsAndResultType,
                                 RCGMP_NotSpeculatable], traits)> {
  let arguments = (ins
      Arg<RCGMP_MpzType, "consumed lhs", RCGMP_ConsumedEffects>:$lhs,
      Arg<RCGMP_MpzType, "consumed rhs", RCGMP_ConsumedEffects>:$rhs);
  let results = (outs Res<RCGMP_MpzType, "new integer", [MemAlloc]>:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let description = [{
    Each operand position consumes one owned reference; aliased heap operands
    require two references. The result owns one reference. A uniquely owned
    operand may supply result storage; shared operands retain their values.
  }] # semantics;
}

def RCGMP_SubIOp : RCGMP_BinaryOp<"subi", [{
    Subtraction is exact, with no fixed-width overflow.
  }]> {
  let summary = "Consume two integer references and produce lhs minus rhs";
}

def RCGMP_MulIOp : RCGMP_BinaryOp<"muli", [{
    Multiplication is exact, with no fixed-width overflow.
  }], [Commutative]> {
  let summary = "Consume two integer references and produce their product";
}

def RCGMP_AndIOp : RCGMP_BinaryOp<"andi", [{
    Operands are interpreted as infinite two's-complement bit strings, including
    sign extension for negative integers.
  }], [Commutative]> {
  let summary = "Consume two integer references and produce their bitwise AND";
}

def RCGMP_XOrIOp : RCGMP_BinaryOp<"xori", [{
    Operands are interpreted as infinite two's-complement bit strings, including
    sign extension for negative integers.
  }], [Commutative]> {
  let summary = "Consume two integer references and produce their bitwise XOR";
}

#endif
