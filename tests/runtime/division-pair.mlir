// RUN: %rcgmp-opt %s -canonicalize | %FileCheck %s
// Entry points of division-pair.cpp, canonicalized before lowering: a quotient
// and a remainder of the same operands become one division.

// CHECK-LABEL: func.func @truncated
// CHECK: rcgmp.divremsi %arg0, %arg1 :
// CHECK-NOT: rcgmp.divsi
// CHECK-NOT: rcgmp.remsi
func.func @truncated(%a: !rcgmp.mpz, %b: !rcgmp.mpz)
    -> (!rcgmp.mpz, !rcgmp.mpz) {
  rcgmp.retain %a : !rcgmp.mpz
  rcgmp.retain %b : !rcgmp.mpz
  %q = rcgmp.divsi %a, %b : !rcgmp.mpz
  %r = rcgmp.remsi %a, %b : !rcgmp.mpz
  return %q, %r : !rcgmp.mpz, !rcgmp.mpz
}

// The remainder comes first, and a zero divisor is Lean's.
// CHECK-LABEL: func.func @floored_lean_atomic
// CHECK: rcgmp.floordivmodsi %arg0, %arg1 divzero<lean> : <atomic>
// CHECK-NOT: rcgmp.floordivsi
// CHECK-NOT: rcgmp.floormodsi
func.func @floored_lean_atomic(%a: !rcgmp.mpz<atomic>, %b: !rcgmp.mpz<atomic>)
    -> (!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>) {
  rcgmp.retain %a : !rcgmp.mpz<atomic>
  rcgmp.retain %b : !rcgmp.mpz<atomic>
  %r = rcgmp.floormodsi %a, %b divzero<lean> : !rcgmp.mpz<atomic>
  %q = rcgmp.floordivsi %a, %b divzero<lean> : !rcgmp.mpz<atomic>
  return %q, %r : !rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>
}

// The dividend stays alive after the pair, which a later sum consumes.
// CHECK-LABEL: func.func @euclidean_kept
// CHECK: rcgmp.edivmodsi %arg0, %arg1 :
// CHECK-NOT: rcgmp.edivsi
// CHECK-NOT: rcgmp.emodsi
func.func @euclidean_kept(%a: !rcgmp.mpz, %b: !rcgmp.mpz)
    -> (!rcgmp.mpz, !rcgmp.mpz) {
  rcgmp.retain %a : !rcgmp.mpz
  rcgmp.retain %a : !rcgmp.mpz
  rcgmp.retain %b : !rcgmp.mpz
  %q = rcgmp.edivsi %a, %b : !rcgmp.mpz
  %r = rcgmp.emodsi %a, %b : !rcgmp.mpz
  %s = rcgmp.addi %r, %a : !rcgmp.mpz
  return %q, %s : !rcgmp.mpz, !rcgmp.mpz
}

// Natural numbers, whose roundings agree, divided by a constant.
// CHECK-LABEL: func.func @natural_by_ten
// CHECK: rcgmp.divremsi %arg0, %{{.*}} nonnegative divzero<lean> :
// CHECK-NOT: rcgmp.divsi
// CHECK-NOT: rcgmp.emodsi
func.func @natural_by_ten(%n: !rcgmp.mpz) -> (!rcgmp.mpz, !rcgmp.mpz) {
  %ten = rcgmp.constant 10 : !rcgmp.mpz
  rcgmp.retain %n : !rcgmp.mpz
  %q = rcgmp.divsi %n, %ten nonnegative divzero<lean> : !rcgmp.mpz
  %r = rcgmp.emodsi %n, %ten nonnegative divzero<lean> : !rcgmp.mpz
  return %q, %r : !rcgmp.mpz, !rcgmp.mpz
}
