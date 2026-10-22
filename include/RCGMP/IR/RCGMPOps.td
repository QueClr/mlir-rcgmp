#ifndef RCGMP_IR_OPS_TD
#define RCGMP_IR_OPS_TD

// Public operations with explicit ownership semantics.
include "RCGMP/IR/RCGMPEnums.td"
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

def RCGMP_ConstantOp : RCGMPOp<"constant", [Pure]> {
  let summary = "Produce an integer constant that needs no reference counting";
  let description = [{
    The value is a signed integer of arbitrary width. Its attribute uses the
    narrowest signless integer type holding the value in two's complement, so
    equal constants have equal attributes; the custom syntax chooses that type.
    Values in the immediate range lower to tagged bits, and other values to a
    static immortal heap object.

    The result needs no reference counting: it may be consumed by any number of
    operations without `rcgmp.retain`, or by none without `rcgmp.release`.
    Retaining or releasing it has no effect.

    ```mlir
    %c = rcgmp.constant -340282366920938463463374607431768211456 : !rcgmp.mpz
    ```
  }];
  let arguments = (ins APIntAttr:$value);
  let results = (outs RCGMP_MpzType:$result);
  let assemblyFormat = "custom<SignedInteger>($value) attr-dict `:` type($result)";
  let builders = [OpBuilder<(ins "Type":$type, "const llvm::APInt &":$value)>];
  let hasVerifier = 1;
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

def RCGMP_CmpIOp : RCGMPOp<"cmpi", [SameTypeOperands, RCGMP_NotSpeculatable]> {
  let summary = "Compare two borrowed integers";
  let description = [{
    Compare the values of the operands with the predicate: equal (`eq`), not
    equal (`ne`), less than (`slt`), less than or equal (`sle`), greater than
    (`sgt`), or greater than or equal (`sge`). The predicates are the signed
    predicates of `arith.cmpi`. Comparison is exact.

    Both operands are borrowed: the comparison neither consumes nor creates a
    reference, and leaves reference counts unchanged. The caller must own a
    reference to each operand, unless it is the result of `rcgmp.constant`, and
    still owns that reference afterwards, so releasing it remains the caller's
    responsibility. The same value may appear as both operands without
    `rcgmp.retain`.

    ```mlir
    %less = rcgmp.cmpi slt, %x, %y : !rcgmp.mpz
    rcgmp.release %x : !rcgmp.mpz
    rcgmp.release %y : !rcgmp.mpz
    ```
  }];
  let arguments = (ins RCGMP_CmpIPredicateAttr:$predicate,
      Arg<RCGMP_MpzType, "borrowed lhs", [MemRead]>:$lhs,
      Arg<RCGMP_MpzType, "borrowed rhs", [MemRead]>:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
}

#endif
