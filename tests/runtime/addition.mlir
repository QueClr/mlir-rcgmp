// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf | %FileCheck %s --check-prefix=SCF
// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts | %FileCheck %s --check-prefix=LLVM
// SCF-DAG: func.func private @rcgmp_mpz_add_atomic(!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>, i32) -> !rcgmp.mpz<atomic>
// SCF-DAG: func.func private @rcgmp_mpz_add(!rcgmp.mpz, !rcgmp.mpz, i32) -> !rcgmp.mpz
// SCF-LABEL: func.func @add_local
// SCF: %[[L:.*]] = rcgmp.bridge.mpz.to_bits
// SCF: %[[R:.*]] = rcgmp.bridge.mpz.to_bits
// The checked sum remains defined on signed overflow.
// SCF: %[[SUM:.*]], %[[OVERFLOW:.*]] = rcgmp.bridge.addsi_extended %[[L]], %[[R]] : i64, i1
// SCF: arith.xori %[[OVERFLOW]]
// SCF: arith.andi
// SCF: arith.cmpi eq
// SCF: scf.if
// SCF: rcgmp.bridge.mpz.from_bits
// SCF: scf.yield
// SCF: } else {
// SCF: rcgmp.bridge.mpz.is_immediate
// SCF: scf.if
// SCF: } else {
// SCF: rcgmp.bridge.mpz.is_unique
// SCF: rcgmp.bridge.mpz.is_immediate
// SCF: rcgmp.bridge.mpz.is_unique
// SCF: arith.extui
// SCF: arith.shli {{.*}} overflow<nsw, nuw>
// SCF: arith.ori
// SCF: call @rcgmp_mpz_add
// LLVM-LABEL: llvm.func @add_local
// LLVM: %[[LL:.*]] = llvm.ptrtoint
// LLVM: %[[RR:.*]] = llvm.ptrtoint
// LLVM: %[[PAIR:.*]] = "llvm.intr.sadd.with.overflow"(%[[LL]], %[[RR]]) : (i64, i64) -> !llvm.struct<(i64, i1)>
// LLVM: llvm.extractvalue %[[PAIR]][0]
// LLVM: llvm.extractvalue %[[PAIR]][1]
// LLVM: llvm.xor
// LLVM: llvm.and
// LLVM: llvm.intr.expect
// LLVM: llvm.cond_br
// LLVM: llvm.inttoptr
// LLVM: llvm.load
// LLVM: llvm.shl {{.*}} overflow<nsw, nuw>
// LLVM: llvm.call @rcgmp_mpz_add
func.func @add_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.addi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
// LLVM-LABEL: llvm.func @add_atomic
// LLVM: llvm.load {{.*}} atomic acquire
// LLVM: llvm.load {{.*}} atomic acquire
// LLVM: llvm.call @rcgmp_mpz_add_atomic
func.func @add_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.addi %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @double_local(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  rcgmp.retain %x : !rcgmp.mpz
  %r = rcgmp.addi %x, %x : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @double_atomic(%x: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  rcgmp.retain %x : !rcgmp.mpz<atomic>
  %r = rcgmp.addi %x, %x : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}

// A constant addend, on either side, is added to the other operand scaled, and
// leaves its tag, which the sum keeps, as the only tag to test.
// SCF-LABEL: func.func @add_five
// SCF: %[[X:.*]] = rcgmp.bridge.mpz.to_bits %arg0
// SCF: %[[TWENTY:.*]] = arith.constant 20 : i64
// SCF: %[[SUM:.*]], %[[OVERFLOW:.*]] = rcgmp.bridge.addsi_extended %[[X]], %[[TWENTY]]
// SCF: arith.andi %[[SUM]]
// SCF: rcgmp.bridge.expect
// SCF: rcgmp.bridge.mpz.from_bits %[[SUM]]
func.func @add_five(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %five = rcgmp.constant 5 : !rcgmp.mpz
  %r = rcgmp.addi %five, %x : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @subtract_five(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %five = rcgmp.constant -5 : !rcgmp.mpz
  %r = rcgmp.addi %x, %five : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
