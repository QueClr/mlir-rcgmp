#ifndef RCGMP_IR_BRIDGE_OPS_TD
#define RCGMP_IR_BRIDGE_OPS_TD

// Low-level operations emitted by the SCF lowering and deferred until LLVM.
include "RCGMP/IR/RCGMPOpBase.td"

class RCGMPBridgeOp<string name, list<Trait> traits = []>
    : RCGMPOp<"bridge." # name, traits> {
  let cppNamespace = "::mlir::rcgmp::bridge";
}

// Generic integer arithmetic, independent of the mpz representation.
def RCGMP_AddSIExtendedOp : RCGMPBridgeOp<"addsi_extended", [Pure, Commutative,
    AllTypesMatch<["lhs", "rhs", "sum"]>]> {
  let summary = "Add signed integers and report signed overflow";
  let description = [{
    The sum is the low N bits of the mathematical sum of the N-bit operands.
    Overflow is true when that mathematical sum does not fit a signed N-bit
    integer. Both results are defined even on overflow. Unlike an addition
    carrying `nsw`, overflowing does not produce poison.

    This is the scalar signed counterpart of `arith.addui_extended` and lowers
    directly to `llvm.sadd.with.overflow`. It does not inspect mpz tags or change
    ownership, and can be used by any arithmetic lowering needing a checked sum.
  }];
  let arguments = (ins AnySignlessInteger:$lhs, AnySignlessInteger:$rhs);
  let results = (outs AnySignlessInteger:$sum, I1:$overflow);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($sum) `,` type($overflow)";
}

def RCGMP_SubSIExtendedOp : RCGMPBridgeOp<"subsi_extended", [Pure,
    AllTypesMatch<["lhs", "rhs", "diff"]>]> {
  let summary = "Subtract signed integers and report signed overflow";
  let description = [{
    Return the low N bits of lhs minus rhs and a flag indicating whether the
    mathematical difference does not fit a signed N-bit integer. Both results
    remain defined on overflow. This scalar signed counterpart of
    `arith.subui_extended` lowers to `llvm.ssub.with.overflow`.
  }];
  let arguments = (ins AnySignlessInteger:$lhs, AnySignlessInteger:$rhs);
  let results = (outs AnySignlessInteger:$diff, I1:$overflow);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($diff) `,` type($overflow)";
}

def RCGMP_MulSIOverflowOp : RCGMPBridgeOp<"mulsi_overflow", [Pure, Commutative,
    AllTypesMatch<["lhs", "rhs", "product"]>]> {
  let summary = "Multiply signed integers and report signed overflow";
  let description = [{
    Return the low N bits of the mathematical product and a flag indicating
    whether the product does not fit a signed N-bit integer. Both results remain
    defined on overflow. Unlike `arith.mulsi_extended`, the second result is an
    overflow flag rather than the high half. This scalar operation lowers
    directly to `llvm.smul.with.overflow`.
  }];
  let arguments = (ins AnySignlessInteger:$lhs, AnySignlessInteger:$rhs);
  let results = (outs AnySignlessInteger:$product, I1:$overflow);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($product) `,` type($overflow)";
}

def RCGMP_ExpectOp : RCGMPBridgeOp<"expect", [Pure]> {
  let summary = "Hint the likely value of a condition";
  let description = [{
    The result is the operand. It tells code generation that the operand
    usually equals `expected`, so that branches on the result lay out that case
    as the fall-through path. It lowers to `llvm.intr.expect`.
  }];
  let arguments = (ins I1:$value, BoolAttr:$expected);
  let results = (outs I1:$result);
  let assemblyFormat = "$value `,` $expected attr-dict";
}

// mpz representation and reference-count primitives.
def RCGMP_MpzIsImmediateOp : RCGMPBridgeOp<"mpz.is_immediate", [Pure]> {
  let summary = "Test the immediate tag without accessing memory";
  let arguments = (ins RCGMP_MpzType:$value);
  let results = (outs I1:$result);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

def RCGMP_MpzFitsImmediateOp : RCGMPBridgeOp<"mpz.fits_imm", [Pure]> {
  let summary = "Test whether a fixed-width integer fits the immediate range";
  let arguments = (ins AnySignlessInteger:$value, UnitAttr:$isUnsigned);
  let results = (outs I1:$result);
  let assemblyFormat = "$value (`unsigned` $isUnsigned^)? attr-dict `:` type($value)";
}

def RCGMP_MpzFromImmediateOp : RCGMPBridgeOp<"mpz.from_imm", [Pure]> {
  let arguments = (ins AnySignlessInteger:$value, UnitAttr:$isUnsigned);
  let results = (outs RCGMP_MpzType:$result);
  let assemblyFormat = "$value (`unsigned` $isUnsigned^)? attr-dict `:` type($value) `to` type($result)";

  let summary = "Encode an integer proven to fit the immediate range";
}

def RCGMP_MpzImmortalOp : RCGMPBridgeOp<"mpz.immortal", [Pure]> {
  let summary = "Address a static immortal integer";
  let description = [{
    The value is a signless integer attribute read as signed two's complement.
    It must lie outside the immediate range of the handle width from the data
    layout. The result is a heap handle to a static, read-only object with a
    zero reference count. It needs no reference counting.
  }];
  let arguments = (ins APIntAttr:$value);
  let results = (outs RCGMP_MpzType:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def RCGMP_MpzPromoteOp : RCGMPBridgeOp<"mpz.promote", [RCGMP_NotSpeculatable]> {
  let arguments = (ins AnySignlessInteger:$value, UnitAttr:$isUnsigned);
  let results =
      (outs Res<RCGMP_MpzType, "new heap integer", [MemAlloc]>:$result);
  let assemblyFormat = "$value (`unsigned` $isUnsigned^)? attr-dict `:` type($value) `to` type($result)";

  let summary = "Allocate an integer that does not fit the immediate range";
}

def RCGMP_MpzIsImmortalOp : RCGMPBridgeOp<"mpz.is_immortal",
    [RCGMP_NotSpeculatable]> {
  let summary = "Test whether a live heap value is immortal";
  let description = [{
    The operand must be a live heap handle. Immortal values are static objects
    with a zero reference count, which a counted value never reaches while a
    reference to it remains. Reference counting must skip them. Atomic counts
    use a relaxed load.
  }];
  let arguments = (ins Arg<RCGMP_MpzType, "heap integer", [MemRead]>:$value);
  let results = (outs I1:$result);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

def RCGMP_MpzIncRefOp : RCGMPBridgeOp<"mpz.inc_ref", [RCGMP_NotSpeculatable]> {
  let summary = "Increment a heap reference count and report overflow";
  let description = [{
    The operand must be a live heap value that is not immortal. The result is
    true if the reference count overflowed; the caller must terminate execution
    in that case.
  }];
  let arguments = (ins Arg<RCGMP_MpzType, "heap integer",
                           [MemRead, MemWrite]>:$value);
  let results = (outs I1:$overflow);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

def RCGMP_MpzDecRefOp : RCGMPBridgeOp<"mpz.dec_ref", [RCGMP_NotSpeculatable]> {
  let summary = "Consume a heap reference and report whether it was the last";
  let description = [{
    The operand must be a live heap value that is not immortal, with one owned
    reference available.
    A true result must be followed by `bridge.mpz.free`. Atomic decrements use
    release ordering; the matching acquire fence is performed by free.
  }];
  let arguments = (ins Arg<RCGMP_MpzType, "heap integer",
                           [MemRead, MemWrite]>:$value);
  let results = (outs I1:$last);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

def RCGMP_MpzFreeOp : RCGMPBridgeOp<"mpz.free", [RCGMP_NotSpeculatable]> {
  let summary = "Free a heap integer after its final reference was released";
  let arguments = (ins Arg<RCGMP_MpzType, "heap integer", [MemFree]>:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

def RCGMP_MpzToBitsOp : RCGMPBridgeOp<"mpz.to_bits", [Pure]> {
  let summary = "Reinterpret an mpz handle as its native-width bit pattern";
  let description = [{
    This unchecked representation cast preserves the tag; it does not decode an
    integer payload or change ownership. The result width must equal the handle
    width from the data layout. It is independent of the width of MLIR index.
  }];
  let arguments = (ins RCGMP_MpzType:$value);
  let results = (outs AnySignlessInteger:$result);
  let assemblyFormat = "$value attr-dict `:` type($value) `to` type($result)";
  let hasVerifier = 1;
}

def RCGMP_MpzFromBitsOp : RCGMPBridgeOp<"mpz.from_bits", [Pure]> {
  let summary = "Reinterpret native-width bits as an mpz handle without checks";
  let description = [{
    No tagging, range checks, allocation, or reference-count changes occur.
    The input width must equal the handle width from the data layout. The caller
    must establish a canonical handle, as defined by the `!rcgmp.mpz` type,
    before using the result; a heap handle must also point to a live
    allocation. Reconstructing a heap handle does not create an owned reference.
  }];
  let arguments = (ins AnySignlessInteger:$value);
  let results = (outs RCGMP_MpzType:$result);
  let assemblyFormat = "$value attr-dict `:` type($value) `to` type($result)";
  let hasVerifier = 1;
}

def RCGMP_MpzIsUniqueOp : RCGMPBridgeOp<"mpz.is_unique", [RCGMP_NotSpeculatable]> {
  let summary = "Test whether a live heap value has exactly one reference";
  let description = [{
    The operand must be a heap handle with an owned reference held by the caller.
    The result is false for immortal values. Atomic counts use an acquire load. A
    true result licenses mutation only while the reference remains exclusive;
    retaining or publishing an alias invalidates that permission. This operation
    is not speculatable or pure.
  }];
  let arguments = (ins Arg<RCGMP_MpzType, "heap integer", [MemRead]>:$value);
  let results = (outs I1:$result);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

def RCGMP_MpzSizeOp : RCGMPBridgeOp<"mpz.size", [RCGMP_NotSpeculatable]> {
  let summary = "Load the signed limb count of a live heap value";
  let description = [{
    The operand must be a heap handle with a reference held by the caller. The
    result is the number of magnitude limbs, negated for a negative value, and
    its width must equal the handle width. A canonical heap value has no
    leading zero limbs, so of two values the one with the smaller signed count
    is smaller.
  }];
  let arguments = (ins Arg<RCGMP_MpzType, "heap integer", [MemRead]>:$value);
  let results = (outs AnySignlessInteger:$result);
  let assemblyFormat = "$value attr-dict `:` type($value) `to` type($result)";
  let hasVerifier = 1;
}

def RCGMP_MpzLimbOp : RCGMPBridgeOp<"mpz.limb", [RCGMP_NotSpeculatable]> {
  let summary = "Load one magnitude limb of a live heap value";
  let description = [{
    The operand must be a heap handle with a reference held by the caller, and
    the index must be below its limb count. Limbs are numbered from the least
    significant. The result width must equal the handle width.
  }];
  let arguments = (ins Arg<RCGMP_MpzType, "heap integer", [MemRead]>:$value,
                       Index:$index);
  let results = (outs AnySignlessInteger:$result);
  let assemblyFormat = [{
    $value `[` $index `]` attr-dict `:` type($value) `to` type($result)
  }];
  let hasVerifier = 1;
}

#endif
