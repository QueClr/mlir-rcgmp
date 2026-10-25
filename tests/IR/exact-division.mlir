// RUN: %rcgmp-opt %s | %rcgmp-opt | %FileCheck %s --check-prefix=ROUND
// RUN: %rcgmp-opt %s -canonicalize | %FileCheck %s

// exact precedes the zero-divisor mode, whose default is elided.
// ROUND-LABEL: func.func @round_trip
// ROUND: rcgmp.divsi %{{[^ ]*}}, %{{[^ ]*}} exact : !rcgmp.mpz
// ROUND: rcgmp.floordivsi %{{.*}}, %{{.*}} exact divzero<undefined> : !rcgmp.mpz
// ROUND: rcgmp.edivsi %{{.*}}, %{{.*}} exact divzero<lean> : <atomic>
// ROUND: rcgmp.divsi %{{.*}}, %{{.*}} exact divzero<lean> : <atomic>
func.func @round_trip(%a: !rcgmp.mpz, %b: !rcgmp.mpz, %c: !rcgmp.mpz,
                      %d: !rcgmp.mpz<atomic>, %e: !rcgmp.mpz<atomic>,
                      %f: !rcgmp.mpz<atomic>) -> (!rcgmp.mpz, !rcgmp.mpz<atomic>) {
  %q = rcgmp.divsi %a, %b exact divzero<trap> : !rcgmp.mpz
  %fq = rcgmp.floordivsi %q, %c exact divzero<undefined> : !rcgmp.mpz
  %eq = rcgmp.edivsi %d, %e exact divzero<lean> : !rcgmp.mpz<atomic>
  %g = "rcgmp.divsi"(%eq, %f) {divZero = #rcgmp.divzero<lean>, exact}
      : (!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic>
  return %fq, %g : !rcgmp.mpz, !rcgmp.mpz<atomic>
}

// Constants of which lhs is a multiple of rhs fold, beyond any fixed width,
// and the lean quotient of zero by zero is zero.
// CHECK-LABEL: func.func @fold
// CHECK-DAG: %[[POWER:.*]] = rcgmp.constant -340282366920938463463374607431768211456 : !rcgmp.mpz
// CHECK-DAG: %[[M3:.*]] = rcgmp.constant -3 : !rcgmp.mpz
// CHECK-DAG: %[[BOUND:.*]] = rcgmp.constant 2305843009213693952 : !rcgmp.mpz
// CHECK-DAG: %[[ZERO:.*]] = rcgmp.constant 0 : !rcgmp.mpz
// CHECK: return %[[POWER]], %[[M3]], %[[M3]], %[[BOUND]], %[[ZERO]]
func.func @fold() -> (!rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz) {
  %wide = rcgmp.constant 1020847100762815390390123822295304634368 : !rcgmp.mpz
  %m3 = rcgmp.constant -3 : !rcgmp.mpz
  %m12 = rcgmp.constant -12 : !rcgmp.mpz
  %four = rcgmp.constant 4 : !rcgmp.mpz
  %min = rcgmp.constant -2305843009213693952 : !rcgmp.mpz
  %m1 = rcgmp.constant -1 : !rcgmp.mpz
  %zero = rcgmp.constant 0 : !rcgmp.mpz
  %a = rcgmp.divsi %wide, %m3 exact : !rcgmp.mpz
  %b = rcgmp.floordivsi %m12, %four exact : !rcgmp.mpz
  %c = rcgmp.edivsi %m12, %four exact divzero<undefined> : !rcgmp.mpz
  %d = rcgmp.divsi %min, %m1 exact : !rcgmp.mpz
  %e = rcgmp.divsi %zero, %zero exact divzero<lean> : !rcgmp.mpz
  return %a, %b, %c, %d, %e : !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz
}

// An exact quotient of constants that is not exact is undefined, and is left
// for the program to compute, as is any quotient by a trapping or undefined
// zero divisor.
// CHECK-LABEL: func.func @unfolded
// CHECK: rcgmp.divsi %{{.*}}, %{{.*}} exact : !rcgmp.mpz
// CHECK: rcgmp.divsi %{{.*}}, %{{.*}} exact divzero<undefined> : !rcgmp.mpz
// CHECK: rcgmp.divsi %{{.*}}, %{{.*}} exact : !rcgmp.mpz
func.func @unfolded() -> (!rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz) {
  %seven = rcgmp.constant 7 : !rcgmp.mpz
  %two = rcgmp.constant 2 : !rcgmp.mpz
  %zero = rcgmp.constant 0 : !rcgmp.mpz
  %a = rcgmp.edivsi %seven, %two exact : !rcgmp.mpz
  %b = rcgmp.divsi %zero, %zero exact divzero<undefined> : !rcgmp.mpz
  %c = rcgmp.divsi %zero, %zero exact : !rcgmp.mpz
  return %a, %b, %c : !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz
}

// Floored and Euclidean exact quotients become truncated ones, keeping the
// zero-divisor mode and nonnegative. A lean exact quotient by zero is zero,
// and lhs is released.
// CHECK-LABEL: func.func @canonicalize
// CHECK-SAME: (%[[X:.*]]: !rcgmp.mpz, %[[Y:.*]]: !rcgmp.mpz, %[[Z:.*]]: !rcgmp.mpz)
// CHECK-DAG: %[[ZERO:.*]] = rcgmp.constant 0 : !rcgmp.mpz
// CHECK-DAG: %[[F:.*]] = rcgmp.divsi %[[X]], %[[Y]] exact divzero<lean> : !rcgmp.mpz
// CHECK-DAG: %[[E:.*]] = rcgmp.divsi %[[F]], %[[Y]] exact nonnegative divzero<undefined> : !rcgmp.mpz
// CHECK-DAG: rcgmp.release %[[Z]] : !rcgmp.mpz
// CHECK-NOT: floordivsi
// CHECK-NOT: edivsi
// CHECK: return %[[E]], %[[ZERO]]
func.func @canonicalize(%x: !rcgmp.mpz, %y: !rcgmp.mpz, %z: !rcgmp.mpz) -> (!rcgmp.mpz, !rcgmp.mpz) {
  %zero = rcgmp.constant 0 : !rcgmp.mpz
  rcgmp.retain %y : !rcgmp.mpz
  %f = rcgmp.floordivsi %x, %y exact divzero<lean> : !rcgmp.mpz
  %e = rcgmp.edivsi %f, %y exact nonnegative divzero<undefined> : !rcgmp.mpz
  %l = rcgmp.edivsi %z, %zero exact divzero<lean> : !rcgmp.mpz
  return %e, %l : !rcgmp.mpz, !rcgmp.mpz
}
