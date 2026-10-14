// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf | %FileCheck %s --check-prefix=SCF
// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts | %FileCheck %s --check-prefix=LLVM
// SCF-DAG: func.func private @rcgmp_mpz_add_atomic(!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>, i32) -> !rcgmp.mpz<atomic>
// SCF-DAG: func.func private @rcgmp_mpz_add(!rcgmp.mpz, !rcgmp.mpz, i32) -> !rcgmp.mpz
// SCF-LABEL: func.func @add_local
// SCF: %[[L:.*]] = rcgmp.bridge.mpz.to_bits
// SCF: %[[R:.*]] = rcgmp.bridge.mpz.to_bits
// The checked sum must wrap; in particular it must not carry nsw/nuw.
// SCF: %[[SUM:.*]] = arith.addi %[[L]], %[[R]] : i64
// SCF: arith.xori %[[L]], %[[SUM]]
// SCF: arith.xori %[[R]], %[[SUM]]
// SCF: arith.andi
// SCF: arith.cmpi sge
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
// LLVM: llvm.add %[[LL]], %[[RR]] : i64
// LLVM: llvm.xor
// LLVM: llvm.and
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
  %alias = rcgmp.retain %x : !rcgmp.mpz
  %r = rcgmp.addi %x, %alias : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @double_atomic(%x: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %alias = rcgmp.retain %x : !rcgmp.mpz<atomic>
  %r = rcgmp.addi %x, %alias : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
