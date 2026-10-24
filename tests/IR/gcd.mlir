// RUN: %rcgmp-opt %s | %rcgmp-opt | %FileCheck %s --check-prefix=ROUND
// RUN: %rcgmp-opt %s -canonicalize | %FileCheck %s

// ROUND-LABEL: func.func @round_trip
// ROUND-SAME: (%[[X:.*]]: !rcgmp.mpz, %[[Y:.*]]: !rcgmp.mpz, %[[Z:.*]]: !rcgmp.mpz<atomic>)
// ROUND: rcgmp.gcdi %[[X]], %[[Y]] : !rcgmp.mpz
// ROUND: rcgmp.gcdi %[[Z]], %[[Z]] : <atomic>
func.func @round_trip(%x: !rcgmp.mpz, %y: !rcgmp.mpz, %z: !rcgmp.mpz<atomic>) -> (!rcgmp.mpz, !rcgmp.mpz<atomic>) {
  %r = rcgmp.gcdi %x, %y : !rcgmp.mpz
  rcgmp.retain %z : !rcgmp.mpz<atomic>
  %s = rcgmp.gcdi %z, %z : !rcgmp.mpz<atomic>
  return %r, %s : !rcgmp.mpz, !rcgmp.mpz<atomic>
}

// Constants fold to the nonnegative greatest common divisor of their absolute
// values, beyond any fixed width.
// CHECK-LABEL: func.func @fold
// CHECK-DAG: %[[POWER:.*]] = rcgmp.constant 340282366920938463463374607431768211456 : !rcgmp.mpz
// CHECK-DAG: %[[SIX:.*]] = rcgmp.constant 6 : !rcgmp.mpz
// CHECK-DAG: %[[ZERO:.*]] = rcgmp.constant 0 : !rcgmp.mpz
// CHECK: return %[[POWER]], %[[SIX]], %[[ZERO]], %[[POWER]]
func.func @fold() -> (!rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz) {
  %min = rcgmp.constant -340282366920938463463374607431768211456 : !rcgmp.mpz
  %m12 = rcgmp.constant -12 : !rcgmp.mpz
  %m18 = rcgmp.constant -18 : !rcgmp.mpz
  %zero = rcgmp.constant 0 : !rcgmp.mpz
  %a = rcgmp.gcdi %min, %min : !rcgmp.mpz
  %b = rcgmp.gcdi %m12, %m18 : !rcgmp.mpz
  %c = rcgmp.gcdi %zero, %zero : !rcgmp.mpz
  %d = rcgmp.gcdi %zero, %min : !rcgmp.mpz
  return %a, %b, %c, %d : !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz
}

// With zero, the greatest common divisor is the absolute value of the other
// operand. With one or minus one, it is one, and the other operand is
// released. A constant on the left moves to the right first.
// CHECK-LABEL: func.func @constant_operand
// CHECK-SAME: (%[[X:.*]]: !rcgmp.mpz, %[[Y:.*]]: !rcgmp.mpz, %[[Z:.*]]: !rcgmp.mpz, %[[W:.*]]: !rcgmp.mpz)
// CHECK-DAG: %[[ONE:.*]] = rcgmp.constant 1 : !rcgmp.mpz
// CHECK-DAG: %[[TWELVE:.*]] = rcgmp.constant -12 : !rcgmp.mpz
// CHECK-DAG: %[[ABS:.*]] = rcgmp.absi %[[X]] : !rcgmp.mpz
// CHECK-DAG: rcgmp.release %[[Y]] : !rcgmp.mpz
// CHECK-DAG: rcgmp.release %[[Z]] : !rcgmp.mpz
// CHECK-DAG: %[[OTHER:.*]] = rcgmp.gcdi %[[W]], %[[TWELVE]] : !rcgmp.mpz
// CHECK: return %[[ABS]], %[[ONE]], %[[ONE]], %[[OTHER]]
func.func @constant_operand(%x: !rcgmp.mpz, %y: !rcgmp.mpz, %z: !rcgmp.mpz, %w: !rcgmp.mpz) -> (!rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz) {
  %zero = rcgmp.constant 0 : !rcgmp.mpz
  %one = rcgmp.constant 1 : !rcgmp.mpz
  %m1 = rcgmp.constant -1 : !rcgmp.mpz
  %m12 = rcgmp.constant -12 : !rcgmp.mpz
  %a = rcgmp.gcdi %zero, %x : !rcgmp.mpz
  %b = rcgmp.gcdi %y, %one : !rcgmp.mpz
  %c = rcgmp.gcdi %m1, %z : !rcgmp.mpz
  %d = rcgmp.gcdi %m12, %w : !rcgmp.mpz
  return %a, %b, %c, %d : !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz
}
