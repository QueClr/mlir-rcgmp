// RUN: %rcgmp-opt %s -canonicalize | %FileCheck %s

// Operations on constants fold exactly, beyond any fixed width.
// CHECK-LABEL: func.func @arithmetic
// CHECK-DAG: %[[SUM:.*]] = rcgmp.constant 340282366920938463463374607431768211457 : !rcgmp.mpz
// CHECK-DAG: %[[ZERO:.*]] = rcgmp.constant 0 : !rcgmp.mpz
// CHECK-DAG: %[[SQUARE:.*]] = rcgmp.constant 115792089237316195423570985008687907853269984665640564039457584007913129639936 : !rcgmp.mpz
// CHECK-DAG: %[[AND:.*]] = rcgmp.constant 1 : !rcgmp.mpz
// CHECK-DAG: %[[XOR:.*]] = rcgmp.constant -4 : !rcgmp.mpz
// CHECK-DAG: %[[OR:.*]] = rcgmp.constant -3 : !rcgmp.mpz
// CHECK-DAG: %[[NOT:.*]] = rcgmp.constant -340282366920938463463374607431768211458 : !rcgmp.mpz
// CHECK-DAG: %[[NEG:.*]] = rcgmp.constant -340282366920938463463374607431768211457 : !rcgmp.mpz
// CHECK: return %[[SUM]], %[[ZERO]], %[[SQUARE]], %[[AND]], %[[XOR]], %[[OR]], %[[NOT]], %[[NEG]], %[[SUM]]
func.func @arithmetic() -> (!rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz) {
  %big = rcgmp.constant 340282366920938463463374607431768211456 : !rcgmp.mpz
  %one = rcgmp.constant 1 : !rcgmp.mpz
  %m3 = rcgmp.constant -3 : !rcgmp.mpz
  %sum = rcgmp.addi %big, %one : !rcgmp.mpz
  %diff = rcgmp.subi %m3, %m3 : !rcgmp.mpz
  %square = rcgmp.muli %big, %big : !rcgmp.mpz
  %and = rcgmp.andi %m3, %one : !rcgmp.mpz
  %xor = rcgmp.xori %m3, %one : !rcgmp.mpz
  %or = rcgmp.ori %m3, %one : !rcgmp.mpz
  %not = rcgmp.noti %sum : !rcgmp.mpz
  %neg = rcgmp.negi %sum : !rcgmp.mpz
  %abs = rcgmp.absi %neg : !rcgmp.mpz
  return %sum, %diff, %square, %and, %xor, %or, %not, %neg, %abs : !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz
}

// Monus stops at zero, beyond any fixed width.
// CHECK-LABEL: func.func @monus
// CHECK-DAG: %[[DIFF:.*]] = rcgmp.constant 340282366920938463463374607431768211459 : !rcgmp.mpz
// CHECK-DAG: %[[ZERO:.*]] = rcgmp.constant 0 : !rcgmp.mpz
// CHECK: return %[[DIFF]], %[[ZERO]], %[[ZERO]]
func.func @monus() -> (!rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz) {
  %big = rcgmp.constant 340282366920938463463374607431768211456 : !rcgmp.mpz
  %m3 = rcgmp.constant -3 : !rcgmp.mpz
  %one = rcgmp.constant 1 : !rcgmp.mpz
  %diff = rcgmp.monusi %big, %m3 : !rcgmp.mpz
  %clamped = rcgmp.monusi %m3, %big : !rcgmp.mpz
  %pred = rcgmp.monusi %m3, %one : !rcgmp.mpz
  return %diff, %clamped, %pred : !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz
}

// Shift amounts are unsigned; a right shift past the width leaves the sign.
// CHECK-LABEL: func.func @shifts
// CHECK-DAG: %[[LEFT:.*]] = rcgmp.constant -4722366482869645213696 : !rcgmp.mpz
// CHECK-DAG: %[[SIGN:.*]] = rcgmp.constant -1 : !rcgmp.mpz
// CHECK: return %[[LEFT]], %[[SIGN]]
func.func @shifts() -> (!rcgmp.mpz, !rcgmp.mpz) {
  %x = rcgmp.constant -4 : !rcgmp.mpz
  %seventy = arith.constant 70 : i8
  %all = arith.constant -1 : i8
  %left = rcgmp.shli %x, %seventy : !rcgmp.mpz, i8
  %right = rcgmp.shrsi %x, %all : !rcgmp.mpz, i8
  return %left, %right : !rcgmp.mpz, !rcgmp.mpz
}

// Truncated division rounds toward zero, floored division toward negative
// infinity, and the minimum by -1 leaves the immediate range exactly.
// CHECK-LABEL: func.func @division
// CHECK-DAG: %[[M1:.*]] = rcgmp.constant -1 : !rcgmp.mpz
// CHECK-DAG: %[[TWO:.*]] = rcgmp.constant 2 : !rcgmp.mpz
// CHECK-DAG: %[[M2:.*]] = rcgmp.constant -2 : !rcgmp.mpz
// CHECK-DAG: %[[BOUND:.*]] = rcgmp.constant 2305843009213693952 : !rcgmp.mpz
// CHECK: return %[[M1]], %[[TWO]], %[[M2]], %[[M2]], %[[M1]], %[[TWO]], %[[BOUND]]
func.func @division() -> (!rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz) {
  %six = rcgmp.constant 6 : !rcgmp.mpz
  %m4 = rcgmp.constant -4 : !rcgmp.mpz
  %min = rcgmp.constant -2305843009213693952 : !rcgmp.mpz
  %m1 = rcgmp.constant -1 : !rcgmp.mpz
  %q = rcgmp.divsi %six, %m4 : !rcgmp.mpz
  %r = rcgmp.remsi %six, %m4 : !rcgmp.mpz
  %fq = rcgmp.floordivsi %six, %m4 : !rcgmp.mpz
  %fr = rcgmp.floormodsi %six, %m4 : !rcgmp.mpz
  %q2, %r2 = rcgmp.divremsi %six, %m4 : !rcgmp.mpz
  %bound = rcgmp.divsi %min, %m1 : !rcgmp.mpz
  return %q, %r, %fq, %fr, %q2, %r2, %bound : !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz
}

// Euclidean division leaves a nonnegative remainder whatever the signs.
// CHECK-LABEL: func.func @euclidean
// CHECK-DAG: %[[M4:.*]] = rcgmp.constant -4 : !rcgmp.mpz
// CHECK-DAG: %[[ONE:.*]] = rcgmp.constant 1 : !rcgmp.mpz
// CHECK-DAG: %[[FOUR:.*]] = rcgmp.constant 4 : !rcgmp.mpz
// CHECK-DAG: %[[M3:.*]] = rcgmp.constant -3 : !rcgmp.mpz
// CHECK: return %[[M4]], %[[ONE]], %[[FOUR]], %[[ONE]], %[[M3]], %[[ONE]]
func.func @euclidean() -> (!rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz) {
  %m7 = rcgmp.constant -7 : !rcgmp.mpz
  %seven = rcgmp.constant 7 : !rcgmp.mpz
  %two = rcgmp.constant 2 : !rcgmp.mpz
  %m2 = rcgmp.constant -2 : !rcgmp.mpz
  %q = rcgmp.edivsi %m7, %two : !rcgmp.mpz
  %r = rcgmp.emodsi %m7, %two : !rcgmp.mpz
  %q2, %r2 = rcgmp.edivmodsi %m7, %m2 : !rcgmp.mpz
  %q3, %r3 = rcgmp.edivmodsi %seven, %m2 : !rcgmp.mpz
  return %q, %r, %q2, %r2, %q3, %r3 : !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz
}

// A lean division by zero has the quotient zero and the remainder lhs. The
// remainder takes over the reference to lhs; a quotient alone releases it.
// CHECK-LABEL: func.func @lean
// CHECK-DAG: %[[ZERO:.*]] = rcgmp.constant 0 : !rcgmp.mpz
// CHECK-DAG: %[[M7:.*]] = rcgmp.constant -7 : !rcgmp.mpz
// CHECK-NOT: divzero
// CHECK: rcgmp.release %arg0 : !rcgmp.mpz
// CHECK-NOT: divzero
// CHECK-NOT: rcgmp.release
// CHECK: return %[[ZERO]], %[[M7]], %[[ZERO]], %[[M7]], %[[ZERO]], %arg1, %[[ZERO]], %arg2
func.func @lean(%x: !rcgmp.mpz, %y: !rcgmp.mpz, %z: !rcgmp.mpz) -> (!rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz) {
  %m7 = rcgmp.constant -7 : !rcgmp.mpz
  %zero = rcgmp.constant 0 : !rcgmp.mpz
  %q = rcgmp.floordivsi %m7, %zero divzero<lean> : !rcgmp.mpz
  %r = rcgmp.emodsi %m7, %zero divzero<lean> : !rcgmp.mpz
  %q2, %r2 = rcgmp.divremsi %m7, %zero divzero<lean> : !rcgmp.mpz
  %xq = rcgmp.divsi %x, %zero divzero<lean> : !rcgmp.mpz
  %yr = rcgmp.remsi %y, %zero divzero<lean> : !rcgmp.mpz
  %zq, %zr = rcgmp.edivmodsi %z, %zero divzero<lean> : !rcgmp.mpz
  return %q, %r, %q2, %r2, %xq, %yr, %zq, %zr : !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz
}

// Conversions in and out, and comparisons, fold to integer constants.
// CHECK-LABEL: func.func @conversions
// CHECK-DAG: %[[U:.*]] = rcgmp.constant 255 : !rcgmp.mpz
// CHECK-DAG: %[[S:.*]] = rcgmp.constant -1 : !rcgmp.mpz
// CHECK-DAG: %[[LOW:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[TRUE:.*]] = arith.constant true
// CHECK-DAG: %[[FALSE:.*]] = arith.constant false
// CHECK: return %[[U]], %[[S]], %[[LOW]], %[[LOW]], %[[TRUE]], %[[FALSE]], %[[TRUE]]
func.func @conversions() -> (!rcgmp.mpz, !rcgmp.mpz, i32, i32, i1, i1, i1) {
  %byte = arith.constant -1 : i8
  %u = rcgmp.extui %byte : i8 to !rcgmp.mpz
  %s = rcgmp.extsi %byte : i8 to !rcgmp.mpz
  %wide = rcgmp.constant 4294967297 : !rcgmp.mpz
  %low = rcgmp.trunci %wide : !rcgmp.mpz to i32
  %v, %o = rcgmp.trunci_overflow %wide unsigned : !rcgmp.mpz to i32
  %lt = rcgmp.cmpi slt, %s, %u : !rcgmp.mpz
  %eq = rcgmp.cmpi eq, %s, %u : !rcgmp.mpz
  return %u, %s, %low, %v, %o, %eq, %lt : !rcgmp.mpz, !rcgmp.mpz, i32, i32, i1, i1, i1
}

// Logarithms are of the magnitude, zero for zero, and keep the low bits that
// fit the result.
// CHECK-LABEL: func.func @logarithms
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[LOG:.*]] = arith.constant 128 : i64
// CHECK-DAG: %[[LOW:.*]] = arith.constant 0 : i7
// CHECK: return %[[ZERO]], %[[LOG]], %[[LOG]], %[[LOW]]
func.func @logarithms() -> (i64, i64, i64, i7) {
  %zero = rcgmp.constant 0 : !rcgmp.mpz
  %big = rcgmp.constant 340282366920938463463374607431768211456 : !rcgmp.mpz
  %negative = rcgmp.constant -680564733841876926926749214863536422911 : !rcgmp.mpz
  %a = rcgmp.log2i %zero : !rcgmp.mpz to i64
  %b = rcgmp.log2i %big : !rcgmp.mpz to i64
  %c = rcgmp.log2i %negative : !rcgmp.mpz to i64
  %d = rcgmp.log2i %big : !rcgmp.mpz to i7
  return %a, %b, %c, %d : i64, i64, i64, i7
}

// Constants need no reference counting.
// CHECK-LABEL: func.func @counting
// CHECK-NOT: rcgmp.retain
// CHECK-NOT: rcgmp.release
// CHECK: return
func.func @counting() {
  %c = rcgmp.constant 340282366920938463463374607431768211456 : !rcgmp.mpz
  rcgmp.retain %c : !rcgmp.mpz
  rcgmp.release %c : !rcgmp.mpz
  rcgmp.release %c : !rcgmp.mpz
  return
}

// Not folded: a zero divisor, which terminates or is undefined; a truncation
// that a flag makes poison; a result past 4096 bits; and non-constant operands.
// CHECK-LABEL: func.func @unfolded
// CHECK: rcgmp.divsi
// CHECK: rcgmp.remsi %arg1, %{{[^ ]*}} : !rcgmp.mpz
// CHECK: rcgmp.floordivsi %arg2, {{.*}} divzero<undefined>
// CHECK: rcgmp.trunci {{.*}} overflow<nsw>
// CHECK: rcgmp.shli
// CHECK: rcgmp.addi %arg0
func.func @unfolded(%x: !rcgmp.mpz, %y: !rcgmp.mpz, %z: !rcgmp.mpz) -> (!rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, i8, !rcgmp.mpz, !rcgmp.mpz) {
  %six = rcgmp.constant 6 : !rcgmp.mpz
  %zero = rcgmp.constant 0 : !rcgmp.mpz
  %big = rcgmp.constant 300 : !rcgmp.mpz
  %far = arith.constant 5000 : i64
  %q = rcgmp.divsi %six, %zero : !rcgmp.mpz
  %r = rcgmp.remsi %y, %zero : !rcgmp.mpz
  %fq = rcgmp.floordivsi %z, %zero divzero<undefined> : !rcgmp.mpz
  %t = rcgmp.trunci %big overflow<nsw> : !rcgmp.mpz to i8
  %l = rcgmp.shli %six, %far : !rcgmp.mpz, i64
  %s = rcgmp.addi %x, %six : !rcgmp.mpz
  return %q, %r, %fq, %t, %l, %s : !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, i8, !rcgmp.mpz, !rcgmp.mpz
}
