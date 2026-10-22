// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf | %FileCheck %s --check-prefix=SCF
// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts | %FileCheck %s --check-prefix=LLVM

// Equal values share one read-only object for both reference-counting modes.
// LLVM-DAG: llvm.mlir.global private unnamed_addr constant @__rcgmp_mpz_2000000000000000(dense<[0, 1, 1, 2305843009213693952]> : tensor<4xi64>) {addr_space = 0 : i32, alignment = 8 : i64} : !llvm.array<4 x i64>
// LLVM-DAG: llvm.mlir.global private unnamed_addr constant @__rcgmp_mpz_neg_2000000000000001(dense<[0, -1, 1, 2305843009213693953]> : tensor<4xi64>)
// LLVM-DAG: llvm.mlir.global private unnamed_addr constant @__rcgmp_mpz_neg_100000000000000000000000000000000(dense<[0, -3, 3, 0, 0, 1]> : tensor<6xi64>)
// LLVM-DAG: llvm.mlir.global private unnamed_addr constant @__rcgmp_mpz_ffffffffffffffff(dense<[0, 1, 1, -1]> : tensor<4xi64>) {addr_space = 0 : i32, alignment = 8 : i64}
// LLVM-DAG: llvm.mlir.global private unnamed_addr constant @__rcgmp_mpz_neg_8000000000000000(dense<[0, -1, 1, -9223372036854775808]> : tensor<4xi64>)
// LLVM-DAG: llvm.mlir.global private unnamed_addr constant @__rcgmp_mpz_30000000000000002(dense<[0, 2, 2, 2, 3]> : tensor<5xi64>)

// SCF-LABEL: func.func @imm_max
// SCF: %[[BITS:.*]] = arith.constant 9223372036854775805 : i64
// SCF: rcgmp.bridge.mpz.from_bits %[[BITS]] : i64 to !rcgmp.mpz
func.func @imm_max() -> !rcgmp.mpz {
  %c = rcgmp.constant 2305843009213693951 : !rcgmp.mpz
  return %c : !rcgmp.mpz
}
// SCF-LABEL: func.func @imm_min
// SCF: arith.constant -9223372036854775807 : i64
func.func @imm_min() -> !rcgmp.mpz {
  %c = rcgmp.constant -2305843009213693952 : !rcgmp.mpz
  return %c : !rcgmp.mpz
}
// SCF-LABEL: func.func @above_max
// SCF: rcgmp.bridge.mpz.immortal 2305843009213693952 : i63 : !rcgmp.mpz
// LLVM-LABEL: llvm.func @above_max
// LLVM: llvm.mlir.addressof @__rcgmp_mpz_2000000000000000 : !llvm.ptr
func.func @above_max() -> !rcgmp.mpz {
  %c = rcgmp.constant 2305843009213693952 : !rcgmp.mpz
  return %c : !rcgmp.mpz
}
// LLVM-LABEL: llvm.func @above_max_atomic
// LLVM: llvm.mlir.addressof @__rcgmp_mpz_2000000000000000 : !llvm.ptr
func.func @above_max_atomic() -> !rcgmp.mpz<atomic> {
  %c = rcgmp.constant 2305843009213693952 : !rcgmp.mpz<atomic>
  return %c : !rcgmp.mpz<atomic>
}
func.func @below_min() -> !rcgmp.mpz {
  %c = rcgmp.constant -2305843009213693953 : !rcgmp.mpz
  return %c : !rcgmp.mpz
}
func.func @wide() -> !rcgmp.mpz<atomic> {
  %c = rcgmp.constant -340282366920938463463374607431768211456 : !rcgmp.mpz<atomic>
  return %c : !rcgmp.mpz<atomic>
}
func.func @one_limb_max() -> !rcgmp.mpz<atomic> {
  %c = rcgmp.constant 18446744073709551615 : !rcgmp.mpz<atomic>
  return %c : !rcgmp.mpz<atomic>
}
func.func @int64_min() -> !rcgmp.mpz {
  %c = rcgmp.constant -9223372036854775808 : !rcgmp.mpz
  return %c : !rcgmp.mpz
}
func.func @limb_order() -> !rcgmp.mpz {
  %c = rcgmp.constant 0x30000000000000002 : !rcgmp.mpz
  return %c : !rcgmp.mpz
}
// Attributes narrower than the handle are sign-extended before tagging.
// SCF-LABEL: func.func @negatives
// SCF-NOT: rcgmp.bridge.mpz.immortal
// SCF: %[[A:.*]] = arith.constant -3 : i64
// SCF: rcgmp.bridge.mpz.from_bits %[[A]] : i64 to !rcgmp.mpz
// SCF: %[[B:.*]] = arith.constant -19 : i64
// SCF: rcgmp.bridge.mpz.from_bits %[[B]] : i64 to !rcgmp.mpz
// SCF-NOT: rcgmp.bridge.mpz.immortal
// SCF: return
func.func @negatives() -> (!rcgmp.mpz, !rcgmp.mpz) {
  %a = rcgmp.constant -1 : !rcgmp.mpz
  %b = rcgmp.constant -5 : !rcgmp.mpz
  return %a, %b : !rcgmp.mpz, !rcgmp.mpz
}

// Constants may be retained, released, and consumed without balancing.
func.func @unbalanced() -> !rcgmp.mpz {
  %c = rcgmp.constant 2305843009213693952 : !rcgmp.mpz
  rcgmp.retain %c : !rcgmp.mpz
  rcgmp.release %c : !rcgmp.mpz
  rcgmp.release %c : !rcgmp.mpz
  return %c : !rcgmp.mpz
}
func.func @square() -> !rcgmp.mpz {
  %c = rcgmp.constant -340282366920938463463374607431768211456 : !rcgmp.mpz
  %r = rcgmp.muli %c, %c : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @add_constant(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %c = rcgmp.constant 2305843009213693952 : !rcgmp.mpz
  %r = rcgmp.addi %c, %x : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @add_constant_atomic(%x: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %c = rcgmp.constant -340282366920938463463374607431768211456 : !rcgmp.mpz<atomic>
  %r = rcgmp.addi %x, %c : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}

// Constants as either operand of each binary operation.
func.func @addi_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.addi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @addi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.addi %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @subi_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.subi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @subi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.subi %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @muli_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.muli %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @muli_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.muli %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @andi_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.andi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @andi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.andi %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @xori_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.xori %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @xori_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.xori %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}

// Releasing a constant any number of times leaves it intact.
func.func @release_many(%n: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %a = rcgmp.constant 2305843009213693952 : !rcgmp.mpz
  %b = rcgmp.constant -340282366920938463463374607431768211456 : !rcgmp.mpz<atomic>
  scf.for %i = %zero to %n step %one {
    rcgmp.release %a : !rcgmp.mpz
    rcgmp.release %b : !rcgmp.mpz<atomic>
  }
  return
}

// A constant consumed on every iteration is never reused in place.
func.func @accumulate_local(%n: index) -> !rcgmp.mpz {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %c = rcgmp.constant 18446744073709551615 : !rcgmp.mpz
  %init = rcgmp.constant 0 : !rcgmp.mpz
  %sum = scf.for %i = %zero to %n step %one iter_args(%acc = %init) -> (!rcgmp.mpz) {
    %next = rcgmp.addi %acc, %c : !rcgmp.mpz
    scf.yield %next : !rcgmp.mpz
  }
  %r = rcgmp.muli %sum, %c : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @accumulate_atomic(%n: index) -> !rcgmp.mpz<atomic> {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %c = rcgmp.constant -340282366920938463463374607431768211456 : !rcgmp.mpz<atomic>
  %init = rcgmp.constant 0 : !rcgmp.mpz<atomic>
  %sum = scf.for %i = %zero to %n step %one iter_args(%acc = %init) -> (!rcgmp.mpz<atomic>) {
    %next = rcgmp.addi %acc, %c : !rcgmp.mpz<atomic>
    scf.yield %next : !rcgmp.mpz<atomic>
  }
  %r = rcgmp.muli %sum, %c : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
