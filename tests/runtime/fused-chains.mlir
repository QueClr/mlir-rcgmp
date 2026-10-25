// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf | %FileCheck %s --check-prefix=SCF
// Entry points tested by fused-chains.cpp. A chain of operations consuming
// each other's integer results takes one fast path, valid when every
// operation's is; the slow path performs the operations one by one.

// The product and the sum are computed together, and one branch selects them.
// The product by a constant takes a range test, which also keeps the sum
// immediate, so the sum takes no test. The slow path calls the runtime alone.
// SCF-LABEL: func.func @collatz_step
// SCF: %[[X:.*]] = rcgmp.bridge.mpz.to_bits %arg0
// SCF: arith.shrsi %[[X]]
// SCF: arith.cmpi ule
// SCF: %[[SCALED:.*]] = arith.muli %[[X]]
// SCF: %[[PRODUCT:.*]] = arith.subi %[[SCALED]]
// SCF-NOT: rcgmp.bridge.addsi_extended
// SCF: %[[SUM:.*]] = arith.addi %[[PRODUCT]]
// SCF: %[[LIKELY:.*]] = rcgmp.bridge.expect %{{.*}}, true
// SCF: scf.if %[[LIKELY]] -> (!rcgmp.mpz) {
// SCF: rcgmp.bridge.mpz.from_bits %[[SUM]]
// SCF: } else {
// SCF-NOT: arith.muli
// SCF: func.call @rcgmp_mpz_mul
// SCF-NOT: rcgmp.bridge.addsi_extended
// SCF: func.call @rcgmp_mpz_add
func.func @collatz_step(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %three = rcgmp.constant 3 : !rcgmp.mpz
  %one = rcgmp.constant 1 : !rcgmp.mpz
  %t = rcgmp.muli %x, %three : !rcgmp.mpz
  %r = rcgmp.addi %t, %one : !rcgmp.mpz
  return %r : !rcgmp.mpz
}

// A comparison borrows the remainder that it tests, which is immediate on the
// fast path, so the remainder is released on the slow path alone.
// SCF-LABEL: func.func @even
// SCF: scf.if %{{.*}} -> (i1) {
// SCF-NOT: rcgmp_mpz_release
// SCF: } else {
// SCF: func.call @rcgmp_mpz_fdiv_r_2exp
// SCF: func.call @rcgmp_mpz_cmp
// SCF: rcgmp.bridge.mpz.is_immediate
// SCF: rcgmp.bridge.mpz.is_unique
// SCF: return
func.func @even(%x: !rcgmp.mpz) -> i32 {
  %zero = rcgmp.constant 0 : !rcgmp.mpz
  %c1 = arith.constant 1 : i64
  %r = rcgmp.floormod2expi %x, %c1 : !rcgmp.mpz, i64
  %t = rcgmp.cmpi eq, %r, %zero : !rcgmp.mpz
  rcgmp.release %r : !rcgmp.mpz
  %e = arith.extui %t : i1 to i32
  return %e : i32
}

// A chain may consume an operand at several positions, and its operations
// may take other operands, such as a shift amount.
// SCF-LABEL: func.func @polynomial_atomic
// SCF: rcgmp.bridge.mulsi_overflow
// SCF: rcgmp.bridge.subsi_extended
// SCF: rcgmp.bridge.mpz.from_bits
func.func @polynomial_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  rcgmp.retain %x : !rcgmp.mpz<atomic>
  %s = rcgmp.muli %x, %x : !rcgmp.mpz<atomic>
  %d = rcgmp.subi %s, %y : !rcgmp.mpz<atomic>
  %c1 = arith.constant 1 : i64
  %q = rcgmp.shrsi %d, %c1 : !rcgmp.mpz<atomic>, i64
  return %q : !rcgmp.mpz<atomic>
}

// A negative factor bounds the product too, but not enough to keep the sum
// immediate, which is then tested.
// SCF-LABEL: func.func @affine_negative
// SCF: arith.cmpi ule
// SCF: arith.muli
// SCF: rcgmp.bridge.addsi_extended
func.func @affine_negative(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %factor = rcgmp.constant -5 : !rcgmp.mpz
  %seven = rcgmp.constant 7 : !rcgmp.mpz
  %t = rcgmp.muli %factor, %x : !rcgmp.mpz
  %r = rcgmp.addi %t, %seven : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
