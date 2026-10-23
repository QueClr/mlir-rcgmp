// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -cse -canonicalize | %FileCheck %s

// The truncation and the release test the same tag, so their branches merge,
// and the release needs no test of its own.
// CHECK-LABEL: func.func @trunc_release
// CHECK: %[[TEST:.*]] = rcgmp.bridge.mpz.is_immediate %arg0
// CHECK: scf.if %[[TEST]]
// CHECK: } else {
// CHECK-NOT: is_immediate
// CHECK: rcgmp.bridge.mpz.is_unique %arg0
// CHECK: return
func.func @trunc_release(%x: !rcgmp.mpz) -> i32 {
  %t = rcgmp.trunci %x : !rcgmp.mpz to i32
  rcgmp.release %x : !rcgmp.mpz
  return %t : i32
}

// A constant operand is known immediate: the fast path tests only the other
// operand, and so does the reuse mask.
// CHECK-LABEL: func.func @constant_operand
// CHECK-NOT: is_immediate
// CHECK: } else {
// CHECK-NEXT: %[[TEST:.*]] = rcgmp.bridge.mpz.is_immediate %arg0
// CHECK-NEXT: scf.if %[[TEST]] -> (i1)
// CHECK-NOT: is_immediate
// CHECK: call @rcgmp_mpz_mul
func.func @constant_operand(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %c = rcgmp.constant 5 : !rcgmp.mpz
  %r = rcgmp.muli %x, %c : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
