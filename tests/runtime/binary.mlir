// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf | %FileCheck %s --check-prefix=SCF
// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts | %FileCheck %s --check-prefix=LLVM
// SCF-LABEL: func.func @sub_local
// SCF: rcgmp.bridge.subsi_extended
// SCF: scf.if
// SCF: rcgmp.bridge.mpz.from_bits
// SCF: rcgmp.bridge.mpz.is_unique
// SCF: func.call @rcgmp_mpz_sub
// LLVM-LABEL: llvm.func @sub_local
// LLVM: "llvm.intr.ssub.with.overflow"
// LLVM: llvm.call @rcgmp_mpz_sub
func.func @sub_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.subi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
// SCF-LABEL: func.func @sub_atomic
// SCF: rcgmp.bridge.subsi_extended
// SCF: scf.if
// SCF: rcgmp.bridge.mpz.from_bits
// SCF: rcgmp.bridge.mpz.is_unique
// SCF: func.call @rcgmp_mpz_sub_atomic
// LLVM-LABEL: llvm.func @sub_atomic
// LLVM: "llvm.intr.ssub.with.overflow"
// LLVM: llvm.load {{.*}} atomic acquire
// LLVM: llvm.call @rcgmp_mpz_sub_atomic
func.func @sub_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.subi %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
// SCF-LABEL: func.func @mul_local
// SCF: arith.shrsi
// SCF: arith.andi
// SCF: rcgmp.bridge.mulsi_overflow
// SCF: scf.if
// SCF: rcgmp.bridge.mpz.from_bits
// SCF: rcgmp.bridge.mpz.is_unique
// SCF: func.call @rcgmp_mpz_mul
// LLVM-LABEL: llvm.func @mul_local
// LLVM: "llvm.intr.smul.with.overflow"
// LLVM: llvm.call @rcgmp_mpz_mul
func.func @mul_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.muli %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
// SCF-LABEL: func.func @mul_atomic
// SCF: arith.shrsi
// SCF: arith.andi
// SCF: rcgmp.bridge.mulsi_overflow
// SCF: scf.if
// SCF: rcgmp.bridge.mpz.from_bits
// SCF: rcgmp.bridge.mpz.is_unique
// SCF: func.call @rcgmp_mpz_mul_atomic
// LLVM-LABEL: llvm.func @mul_atomic
// LLVM: "llvm.intr.smul.with.overflow"
// LLVM: llvm.load {{.*}} atomic acquire
// LLVM: llvm.call @rcgmp_mpz_mul_atomic
func.func @mul_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.muli %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
// SCF-LABEL: func.func @and_local
// SCF: arith.andi
// SCF: scf.if
// SCF: rcgmp.bridge.mpz.from_bits
// SCF: rcgmp.bridge.mpz.is_unique
// SCF: func.call @rcgmp_mpz_and
// LLVM-LABEL: llvm.func @and_local
// LLVM: llvm.call @rcgmp_mpz_and
func.func @and_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.andi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
// SCF-LABEL: func.func @and_atomic
// SCF: arith.andi
// SCF: scf.if
// SCF: rcgmp.bridge.mpz.from_bits
// SCF: rcgmp.bridge.mpz.is_unique
// SCF: func.call @rcgmp_mpz_and_atomic
// LLVM-LABEL: llvm.func @and_atomic
// LLVM: llvm.load {{.*}} atomic acquire
// LLVM: llvm.call @rcgmp_mpz_and_atomic
func.func @and_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.andi %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
// SCF-LABEL: func.func @xor_local
// SCF: arith.xori
// SCF: arith.ori
// SCF: scf.if
// SCF: rcgmp.bridge.mpz.from_bits
// SCF: rcgmp.bridge.mpz.is_unique
// SCF: func.call @rcgmp_mpz_xor
// LLVM-LABEL: llvm.func @xor_local
// LLVM: llvm.call @rcgmp_mpz_xor
func.func @xor_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.xori %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
// SCF-LABEL: func.func @xor_atomic
// SCF: arith.xori
// SCF: arith.ori
// SCF: scf.if
// SCF: rcgmp.bridge.mpz.from_bits
// SCF: rcgmp.bridge.mpz.is_unique
// SCF: func.call @rcgmp_mpz_xor_atomic
// LLVM-LABEL: llvm.func @xor_atomic
// LLVM: llvm.load {{.*}} atomic acquire
// LLVM: llvm.call @rcgmp_mpz_xor_atomic
func.func @xor_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.xori %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
// SCF-LABEL: func.func @or_local
// SCF: arith.ori
// SCF: scf.if
// SCF: rcgmp.bridge.mpz.from_bits
// SCF: rcgmp.bridge.mpz.is_unique
// SCF: func.call @rcgmp_mpz_or
// LLVM-LABEL: llvm.func @or_local
// LLVM: llvm.call @rcgmp_mpz_or
func.func @or_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.ori %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
// SCF-LABEL: func.func @or_atomic
// SCF: arith.ori
// SCF: scf.if
// SCF: rcgmp.bridge.mpz.from_bits
// SCF: rcgmp.bridge.mpz.is_unique
// SCF: func.call @rcgmp_mpz_or_atomic
// LLVM-LABEL: llvm.func @or_atomic
// LLVM: llvm.load {{.*}} atomic acquire
// LLVM: llvm.call @rcgmp_mpz_or_atomic
func.func @or_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.ori %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
