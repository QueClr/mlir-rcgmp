#ifndef RCGMP_IR_OPS_TD
#define RCGMP_IR_OPS_TD

// Public operations with explicit ownership semantics.
include "RCGMP/IR/RCGMPAttrs.td"
include "RCGMP/IR/RCGMPEnums.td"
include "RCGMP/IR/RCGMPOpBase.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

// A consumed reference may be decremented, reused, or freed.
defvar RCGMP_ConsumedEffects = [MemRead, MemWrite, MemFree];

class RCGMP_ExtendOp<string name, string interpretation>
    : RCGMPOp<name, [RCGMP_NotSpeculatable]> {
  let arguments = (ins AnySignlessInteger:$value);
  let results = (outs Res<RCGMP_MpzType, "new integer", [MemAlloc]>:$result);
  let assemblyFormat = "$value attr-dict `:` type($value) `to` type($result)";
  let description = [{
    The input is interpreted as }] # interpretation # [{, of any
    integer bit width. The result owns one reference.
  }];
  let hasFolder = 1;
}

def RCGMP_ExtSIOp : RCGMP_ExtendOp<"extsi", "signed two's complement"> {
  let summary = "Create one owned integer by sign-extending a fixed-width integer";
}

def RCGMP_ExtUIOp : RCGMP_ExtendOp<"extui", "unsigned"> {
  let summary = "Create one owned integer by zero-extending a fixed-width integer";
}

def RCGMP_ConstantOp : RCGMPOp<"constant", [ConstantLike, Pure]> {
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
  let hasFolder = 1;
}

def RCGMP_RetainOp : RCGMPOp<"retain", [RCGMP_NotSpeculatable]> {
  let summary = "Add one owned reference to an integer";
  let description = [{
    The operand must own at least one reference and owns one more afterwards.
  }];
  let arguments = (ins Arg<RCGMP_MpzType, "retained integer",
                           [MemRead, MemWrite]>:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
  let hasCanonicalizeMethod = 1;
}

def RCGMP_ReleaseOp : RCGMPOp<"release", [RCGMP_NotSpeculatable]> {
  let summary = "Consume one owned reference";
  let arguments = (ins Arg<RCGMP_MpzType, "released integer",
                           RCGMP_ConsumedEffects>:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
  let hasCanonicalizeMethod = 1;
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
  let hasFolder = 1;
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
  let hasFolder = 1;
}

def RCGMP_SubIOp : RCGMP_BinaryOp<"subi", [{
    Subtraction is exact, with no fixed-width overflow.
  }]> {
  let summary = "Consume two integer references and produce lhs minus rhs";
}

def RCGMP_MonusIOp : RCGMP_BinaryOp<"monusi", [{
    The result is lhs minus rhs when lhs is greater than rhs, and zero
    otherwise, which is the subtraction of natural numbers.
  }]> {
  let summary = "Consume two integer references and produce lhs minus rhs, at least zero";
}

def RCGMP_MulIOp : RCGMP_BinaryOp<"muli", [{
    Multiplication is exact, with no fixed-width overflow.
  }], [Commutative]> {
  let summary = "Consume two integer references and produce their product";
}

def RCGMP_GcdIOp : RCGMP_BinaryOp<"gcdi", [{
    The result is the greatest common divisor of the absolute values of the
    operands, which is never negative. The greatest common divisor of zero and
    x is the absolute value of x, so that of two zeros is zero.
  }], [Commutative]> {
  let summary = "Consume two integer references and produce their greatest common divisor";
  let hasCanonicalizer = 1;
}

def RCGMP_AndIOp : RCGMP_BinaryOp<"andi", [{
    Operands are interpreted as infinite two's-complement bit strings, including
    sign extension for negative integers.
  }], [Commutative]> {
  let summary = "Consume two integer references and produce their bitwise AND";
  let hasCanonicalizer = 1;
}

def RCGMP_XOrIOp : RCGMP_BinaryOp<"xori", [{
    Operands are interpreted as infinite two's-complement bit strings, including
    sign extension for negative integers.
  }], [Commutative]> {
  let summary = "Consume two integer references and produce their bitwise XOR";
}

def RCGMP_OrIOp : RCGMP_BinaryOp<"ori", [{
    Operands are interpreted as infinite two's-complement bit strings, including
    sign extension for negative integers.
  }], [Commutative]> {
  let summary = "Consume two integer references and produce their bitwise OR";
}

class RCGMP_UnaryOp<string name, string semantics>
    : RCGMPOp<name, [SameOperandsAndResultType, RCGMP_NotSpeculatable]> {
  let arguments = (ins
      Arg<RCGMP_MpzType, "consumed operand", RCGMP_ConsumedEffects>:$operand);
  let results = (outs Res<RCGMP_MpzType, "new integer", [MemAlloc]>:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
  let description = [{
    The operand consumes one owned reference, and the result owns one
    reference. A uniquely owned operand may supply result storage; a shared
    operand retains its value.
  }] # semantics;
  let hasFolder = 1;
}

def RCGMP_NotIOp : RCGMP_UnaryOp<"noti", [{
    The operand is interpreted as an infinite two's-complement bit string, so
    the complement of x is -x - 1.
  }]> {
  let summary = "Consume an integer reference and produce its bitwise complement";
}

def RCGMP_NegIOp : RCGMP_UnaryOp<"negi", [{
    Negation is exact, with no fixed-width overflow.
  }]> {
  let summary = "Consume an integer reference and produce its negation";
}

def RCGMP_AbsIOp : RCGMP_UnaryOp<"absi", [{
    The absolute value is exact, with no fixed-width overflow.
  }]> {
  let summary = "Consume an integer reference and produce its absolute value";
}

class RCGMP_ShiftOp<string name, string semantics>
    : RCGMPOp<name, [AllTypesMatch<["value", "result"]>,
                     RCGMP_NotSpeculatable]> {
  let arguments = (ins
      Arg<RCGMP_MpzType, "consumed integer", RCGMP_ConsumedEffects>:$value,
      AnySignlessInteger:$amount);
  let results = (outs Res<RCGMP_MpzType, "new integer", [MemAlloc]>:$result);
  let assemblyFormat = [{
    $value `,` $amount attr-dict `:` type($result) `,` type($amount)
  }];
  let description = [{
    The integer operand consumes one owned reference, and the result owns one
    reference. A uniquely owned operand may supply result storage; a shared
    operand retains its value. The amount is an unsigned number of bits.
  }] # semantics;
  let hasFolder = 1;
}

def RCGMP_ShLIOp : RCGMP_ShiftOp<"shli", [{
    The result is the integer times two to the power of the amount, exactly. An
    amount whose result cannot be allocated terminates the program, like any
    allocation that fails.
  }]> {
  let summary = "Consume an integer reference and shift it left";
}

def RCGMP_ShRSIOp : RCGMP_ShiftOp<"shrsi", [{
    The result is the integer divided by two to the power of the amount,
    rounded toward negative infinity: an arithmetic shift of its infinite
    two's-complement bits.
  }]> {
  let summary = "Consume an integer reference and shift it right arithmetically";
}

def RCGMP_FloorMod2ExpIOp : RCGMP_ShiftOp<"floormod2expi", [{
    The result is the remainder of the integer by two to the power of the
    amount, rounded toward negative infinity as by `rcgmp.floormodsi`, so that
    it is nonnegative and less than that power: the low bits of the integer's
    infinite two's complement, as many as the amount, read as an unsigned
    integer. It is the remainder of the quotient that `rcgmp.shrsi` computes.
    The remainder by two to the power zero is zero. An amount whose result
    cannot be allocated terminates the program, like any allocation that fails.

    ```mlir
    %low = rcgmp.floormod2expi %x, %w : !rcgmp.mpz, i64
    ```
  }]> {
  let summary = "Consume an integer reference and reduce it modulo a power of two";
  let hasCanonicalizer = 1;
}

def RCGMP_IPowIOp : RCGMPOp<"ipowi", [AllTypesMatch<["base", "result"]>,
                                       RCGMP_NotSpeculatable]> {
  let summary = "Consume an integer reference and raise it to a power";
  let description = [{
    The base consumes one owned reference, and the result owns one reference.
    A uniquely owned base may supply result storage; a shared base retains its
    value. The exponent is an unsigned integer.

    The result is the base to the power of the exponent, exactly. Any base to
    the power zero is one, zero included. An exponent whose result cannot be
    allocated terminates the program, like any allocation that fails.

    ```mlir
    %p = rcgmp.ipowi %x, %n : !rcgmp.mpz, i64
    ```
  }];
  let arguments = (ins
      Arg<RCGMP_MpzType, "consumed base", RCGMP_ConsumedEffects>:$base,
      AnySignlessInteger:$exponent);
  let results = (outs Res<RCGMP_MpzType, "new integer", [MemAlloc]>:$result);
  let assemblyFormat = [{
    $base `,` $exponent attr-dict `:` type($result) `,` type($exponent)
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

defvar RCGMP_DivisionArguments = (ins
    Arg<RCGMP_MpzType, "consumed dividend", RCGMP_ConsumedEffects>:$lhs,
    Arg<RCGMP_MpzType, "consumed divisor", RCGMP_ConsumedEffects>:$rhs,
    DefaultValuedAttr<RCGMP_DivZeroAttr,
                      "::mlir::rcgmp::DivZero::trap">:$divZero);

defvar RCGMP_DivisionDescription = [{
    Each operand position consumes one owned reference; aliased heap operands
    require two references. Each result owns one reference. A uniquely owned
    operand may supply result storage; shared operands retain their values.
    Division is exact, with no fixed-width overflow.
  }];

defvar RCGMP_DivZeroDescription = [{
    The `divzero` mode sets what a zero divisor does. With `trap`, the default,
    it terminates the program. With `undefined`, it is undefined behavior. With
    `lean`, the quotient is zero and the remainder is lhs, as for the integer
    division and remainder of Lean.

    ```mlir
    %r = rcgmp.remsi %x, %y divzero<lean> : !rcgmp.mpz
    ```
  }];

class RCGMP_DivisionOp<string name, string semantics>
    : RCGMPOp<name, [SameOperandsAndResultType, RCGMP_NotSpeculatable]> {
  let arguments = RCGMP_DivisionArguments;
  let results = (outs Res<RCGMP_MpzType, "new integer", [MemAlloc]>:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs (`divzero` `` $divZero^)? attr-dict `:` type($result)
  }];
  let description =
      RCGMP_DivisionDescription # semantics # RCGMP_DivZeroDescription;
  let hasFolder = 1;
}

// Only quotients take `exact`: the remainder it asserts is zero, and the pairs
// would compute nothing more than a quotient.
defvar RCGMP_ExactDescription = [{
    With `exact`, lhs must be a multiple of rhs, and otherwise the result is
    undefined. The quotient then does not depend on how it is rounded. Zero is
    the only multiple of zero: with `lean`, the result of a zero divisor is
    undefined unless lhs is zero, whose quotient is zero, as for Lean's
    `Nat.divExact` and `Int.divExact`. With `trap`, a zero divisor still
    terminates the program.

    ```mlir
    %q = rcgmp.divsi %x, %y exact divzero<lean> : !rcgmp.mpz
    ```
  }];

class RCGMP_QuotientOp<string name, string semantics>
    : RCGMP_DivisionOp<name, semantics> {
  let arguments = !con(RCGMP_DivisionArguments, (ins UnitAttr:$exact));
  let assemblyFormat = [{
    $lhs `,` $rhs (`exact` $exact^)? (`divzero` `` $divZero^)? attr-dict `:`
    type($result)
  }];
  let description = RCGMP_DivisionDescription # semantics #
                    RCGMP_DivZeroDescription # RCGMP_ExactDescription;
  let hasCanonicalizeMethod = 1;
}

def RCGMP_DivSIOp : RCGMP_QuotientOp<"divsi", [{
    The quotient is rounded toward zero.
  }]> {
  let summary = "Consume two integer references and produce their truncated quotient";
}

def RCGMP_RemSIOp : RCGMP_DivisionOp<"remsi", [{
    The remainder of the quotient rounded toward zero has the sign of lhs.
  }]> {
  let summary = "Consume two integer references and produce their truncated remainder";
}

def RCGMP_FloorDivSIOp : RCGMP_QuotientOp<"floordivsi", [{
    The quotient is rounded toward negative infinity.
  }]> {
  let summary = "Consume two integer references and produce their floored quotient";
}

def RCGMP_FloorModSIOp : RCGMP_DivisionOp<"floormodsi", [{
    The remainder of the quotient rounded toward negative infinity has the sign
    of rhs.
  }]> {
  let summary = "Consume two integer references and produce their floored remainder";
  let hasCanonicalizer = 1;
}

def RCGMP_EDivSIOp : RCGMP_QuotientOp<"edivsi", [{
    The quotient is rounded toward negative infinity for a positive rhs and
    toward positive infinity for a negative rhs, so that the remainder is
    nonnegative.
  }]> {
  let summary = "Consume two integer references and produce their Euclidean quotient";
}

def RCGMP_EModSIOp : RCGMP_DivisionOp<"emodsi", [{
    The remainder of the Euclidean quotient is nonnegative and less than the
    magnitude of rhs.
  }]> {
  let summary = "Consume two integer references and produce their Euclidean remainder";
  let hasCanonicalizer = 1;
}

class RCGMP_DivisionPairOp<string name, string semantics>
    : RCGMPOp<name, [SameOperandsAndResultType, RCGMP_NotSpeculatable]> {
  let arguments = RCGMP_DivisionArguments;
  let results = (outs Res<RCGMP_MpzType, "new quotient", [MemAlloc]>:$quotient,
                      Res<RCGMP_MpzType, "new remainder", [MemAlloc]>:$remainder);
  let assemblyFormat = [{
    $lhs `,` $rhs (`divzero` `` $divZero^)? attr-dict `:` type($quotient)
  }];
  let description =
      RCGMP_DivisionDescription # semantics # RCGMP_DivZeroDescription;
  let hasFolder = 1;
}

def RCGMP_DivRemSIOp : RCGMP_DivisionPairOp<"divremsi", [{
    The quotient is rounded toward zero, and the remainder has the sign of lhs,
    as `rcgmp.divsi` and `rcgmp.remsi` compute them.
  }]> {
  let summary = "Consume two integer references and produce their truncated quotient and remainder";
}

def RCGMP_FloorDivModSIOp : RCGMP_DivisionPairOp<"floordivmodsi", [{
    The quotient is rounded toward negative infinity, and the remainder has the
    sign of rhs, as `rcgmp.floordivsi` and `rcgmp.floormodsi` compute them.
  }]> {
  let summary = "Consume two integer references and produce their floored quotient and remainder";
}

def RCGMP_EDivModSIOp : RCGMP_DivisionPairOp<"edivmodsi", [{
    The remainder is nonnegative and less than the magnitude of rhs, as
    `rcgmp.edivsi` and `rcgmp.emodsi` compute them.
  }]> {
  let summary = "Consume two integer references and produce their Euclidean quotient and remainder";
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
  let hasFolder = 1;
}

def RCGMP_TruncIOp : RCGMPOp<"trunci", [RCGMP_NotSpeculatable]> {
  let summary = "Truncate a borrowed integer to a fixed-width integer";
  let description = [{
    The result is the low bits of the integer's infinite two's-complement
    representation, so the integer is reduced modulo two to the power of the
    result width. The operand is borrowed, as by `rcgmp.cmpi`.

    As for `arith.trunci`, the `nsw` flag makes the result poison when the
    integer does not fit the result type as a signed integer, and `nuw` when
    it does not fit as an unsigned one.

    ```mlir
    %v = rcgmp.trunci %x overflow<nsw> : !rcgmp.mpz to i32
    ```
  }];
  let arguments = (ins Arg<RCGMP_MpzType, "borrowed integer", [MemRead]>:$value,
      DefaultValuedAttr<Arith_IntegerOverflowAttr,
                        "::mlir::arith::IntegerOverflowFlags::none">:$overflowFlags);
  let results = (outs AnySignlessInteger:$result);
  let assemblyFormat = [{
    $value (`overflow` `` $overflowFlags^)? attr-dict `:` type($value) `to`
    type($result)
  }];
  let hasFolder = 1;
}

def RCGMP_TruncIOverflowOp : RCGMPOp<"trunci_overflow",
    [RCGMP_NotSpeculatable]> {
  let summary = "Truncate a borrowed integer and report whether it fits";
  let description = [{
    The value is the result of `rcgmp.trunci`. The overflow bit is set when the
    integer does not fit the result type as a signed integer, or with
    `unsigned` as an unsigned integer. The operand is borrowed.

    ```mlir
    %v, %overflow = rcgmp.trunci_overflow %x unsigned : !rcgmp.mpz to i32
    ```
  }];
  let arguments = (ins Arg<RCGMP_MpzType, "borrowed integer", [MemRead]>:$value,
      UnitAttr:$isUnsigned);
  let results = (outs AnySignlessInteger:$result, I1:$overflow);
  let assemblyFormat = [{
    $value (`unsigned` $isUnsigned^)? attr-dict `:` type($value) `to`
    type($result)
  }];
  let hasFolder = 1;
}

def RCGMP_Log2IOp : RCGMPOp<"log2i", [RCGMP_NotSpeculatable]> {
  let summary = "Compute the base-2 logarithm of a borrowed integer";
  let description = [{
    The result is the base-2 logarithm of the absolute value of the integer,
    rounded down: one less than the number of bits of the magnitude. The
    logarithm of zero is zero, as for Lean's `Nat.log2`. The operand is
    borrowed, as by `rcgmp.cmpi`.

    The result is an unsigned integer. A result type too narrow for the
    logarithm holds its low bits: the logarithm is reduced modulo two to the
    power of the result width.

    ```mlir
    %log = rcgmp.log2i %x : !rcgmp.mpz to i64
    ```
  }];
  let arguments = (ins Arg<RCGMP_MpzType, "borrowed integer", [MemRead]>:$value);
  let results = (outs AnySignlessInteger:$result);
  let assemblyFormat = "$value attr-dict `:` type($value) `to` type($result)";
  let hasFolder = 1;
}

def RCGMP_TestBitIOp : RCGMPOp<"testbiti", [RCGMP_NotSpeculatable]> {
  let summary = "Test one bit of a borrowed integer";
  let description = [{
    The result is the bit of the integer's infinite two's-complement
    representation at the index, as for Lean's `Nat.testBit` and `Int.testBit`
    and GMP's `mpz_tstbit`: bits beyond the magnitude are zero for a
    nonnegative integer and one for a negative one. The index is an unsigned
    number of bits, of any width, counted from the least significant bit. The
    operand is borrowed, as by `rcgmp.cmpi`.

    ```mlir
    %bit = rcgmp.testbiti %x, %n : !rcgmp.mpz, i64
    ```
  }];
  let arguments = (ins Arg<RCGMP_MpzType, "borrowed integer", [MemRead]>:$value,
      AnySignlessInteger:$index);
  let results = (outs I1:$result);
  let assemblyFormat = [{
    $value `,` $index attr-dict `:` type($value) `,` type($index)
  }];
  let hasFolder = 1;
}

def RCGMP_LimbCountOp : RCGMPOp<"limb_count", [RCGMP_NotSpeculatable]> {
  let summary = "Count the limbs of the magnitude of a borrowed integer";
  let description = [{
    The result is the number of limbs of the absolute value of the integer,
    as for GMP's `mpz_size`: the least `n` such that the absolute value is
    below 2^(W * n), where the limb width W is the handle width from the data
    layout. Zero has no limbs. The operand is borrowed, as by `rcgmp.cmpi`.

    With `rcgmp.limb`, this reads the magnitude of an integer limb by limb.
    Both depend only on the value, not on whether it is immediate.

    ```mlir
    %n = rcgmp.limb_count %x : !rcgmp.mpz
    ```
  }];
  let arguments = (ins Arg<RCGMP_MpzType, "borrowed integer", [MemRead]>:$value);
  let results = (outs Index:$result);
  let assemblyFormat = "$value attr-dict `:` type($value)";
  let hasFolder = 1;
}

def RCGMP_LimbOp : RCGMPOp<"limb", [RCGMP_NotSpeculatable]> {
  let summary = "Read one limb of the magnitude of a borrowed integer";
  let description = [{
    The result is the limb of the absolute value of the integer at the index,
    counted from the least significant, as for GMP's `mpz_getlimbn`: the
    absolute value shifted right by W * index bits, modulo 2^W. Its width must
    equal the limb width W, which is the handle width from the data layout.
    Every index at or beyond `rcgmp.limb_count` reads zero. The operand is
    borrowed, as by `rcgmp.cmpi`.

    ```mlir
    %limb = rcgmp.limb %x[%i] : !rcgmp.mpz to i64
    ```
  }];
  let arguments = (ins Arg<RCGMP_MpzType, "borrowed integer", [MemRead]>:$value,
      Index:$index);
  let results = (outs AnySignlessInteger:$result);
  let assemblyFormat = [{
    $value `[` $index `]` attr-dict `:` type($value) `to` type($result)
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
}

def RCGMP_MatchOp : RCGMPOp<"match", [RCGMP_NotSpeculatable,
    RecursiveMemoryEffects, SingleBlockImplicitTerminator<"YieldOp">,
    DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getSuccessorInputs"]>]> {
  let summary = "Select a region by the value of a consumed integer";
  let description = [{
    Run the region of the first arm that selects the value of the scrutinee,
    and return the values it yields. An arm is one of:

    - `case k, ...`, selecting each listed integer;
    - `at_least c`, selecting the integers at least `c`;
    - `at_most c`, selecting the integers at most `c`;
    - `default`, selecting every integer. It may only be the last arm.

    Arms are tested in order, so an arm may select integers that an earlier arm
    also selects. Values of any width may be listed. If no arm selects the
    scrutinee, the behavior is undefined.

    The match consumes one owned reference to the scrutinee. An `at_least c`
    or `at_most c` region may take an argument, which owns one reference to
    the distance of the scrutinee from `c`: the scrutinee minus `c`, or `c`
    minus the scrutinee, so that it is never negative. A `default` region may
    take one owning the scrutinee. When the selected region takes no argument,
    the match releases the scrutinee itself.

    Lean's `Int.ofNat n` and `Int.negSucc n`, for example, are `at_least 0`
    and `at_most -1`, whose arguments are `n`.

    ```mlir
    %r = rcgmp.match %n : !rcgmp.mpz -> i64 {
      case 0, 1 {
        rcgmp.yield %one : i64
      }
      at_least 2 (%k) {
        %f = func.call @f(%k) : (!rcgmp.mpz) -> i64
        rcgmp.yield %f : i64
      }
    }
    ```
  }];
  let arguments = (ins
      Arg<RCGMP_MpzType, "consumed scrutinee", RCGMP_ConsumedEffects>:$scrutinee,
      TypedArrayAttrBase<RCGMP_ArmAttr, "match arms">:$arms);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region VariadicRegion<SizedRegion<1>>:$bodies);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{
    ArmAttr getArm(unsigned index) {
      return ::llvm::cast<ArmAttr>(getArms()[index]);
    }

    // The integers each arm selects, after earlier arms have selected theirs.
    SmallVector<MatchInterval> getIntervals();
  }];
}

def RCGMP_YieldOp : RCGMPOp<"yield", [Pure, ReturnLike, Terminator,
    HasParent<"MatchOp">]> {
  let summary = "Return values from a match arm";
  let arguments = (ins Variadic<AnyType>:$results);
  let assemblyFormat = "attr-dict ($results^ `:` type($results))?";
  let builders = [OpBuilder<(ins), [{ /* Yields no values. */ }]>];
}

#endif
