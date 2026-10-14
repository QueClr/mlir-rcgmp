#ifndef RCGMP_IR_BRIDGE_OPS_TD
#define RCGMP_IR_BRIDGE_OPS_TD

// Low-level operations emitted by the SCF lowering and deferred until LLVM.
include "RCGMP/IR/RCGMPOpBase.td"

class RCGMPBridgeOp<string name, list<Trait> traits = []>
    : RCGMPOp<"bridge." # name, traits> {
  let cppNamespace = "::mlir::rcgmp::bridge";
}

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

def RCGMP_MpzPromoteOp : RCGMPBridgeOp<"mpz.promote"> {
  let arguments = (ins AnySignlessInteger:$value, UnitAttr:$isUnsigned);
  let results = (outs RCGMP_MpzType:$result);
  let assemblyFormat = "$value (`unsigned` $isUnsigned^)? attr-dict `:` type($value) `to` type($result)";

  let summary = "Allocate an integer that does not fit the immediate range";
}

def RCGMP_MpzIncRefOp : RCGMPBridgeOp<"mpz.inc_ref"> {
  let summary = "Increment a heap reference count and report overflow";
  let description = [{
    The operand must be a live heap value. The result is true if the reference
    count overflowed; the caller must terminate execution in that case.
  }];
  let arguments = (ins RCGMP_MpzType:$value);
  let results = (outs I1:$overflow);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

def RCGMP_MpzDecRefOp : RCGMPBridgeOp<"mpz.dec_ref"> {
  let summary = "Consume a heap reference and report whether it was the last";
  let description = [{
    The operand must be a live heap value with one owned reference available.
    A true result must be followed by `bridge.mpz.free`. Atomic decrements use
    release ordering; the matching acquire fence is performed by free.
  }];
  let arguments = (ins RCGMP_MpzType:$value);
  let results = (outs I1:$last);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

def RCGMP_MpzFreeOp : RCGMPBridgeOp<"mpz.free"> {
  let summary = "Free a heap integer after its final reference was released";
  let arguments = (ins RCGMP_MpzType:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

#endif
