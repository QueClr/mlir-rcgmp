// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf | %FileCheck %s --check-prefix=SCF
// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts | %FileCheck %s --check-prefix=LLVM

// SCF-LABEL: func.func @from_signed
// SCF: rcgmp.bridge.mpz.fits_imm
// SCF: scf.if
// SCF: rcgmp.bridge.mpz.from_imm
// SCF: } else {
// SCF: rcgmp.bridge.mpz.promote
// LLVM-LABEL: llvm.func @from_signed
// LLVM: llvm.alloca
// LLVM: llvm.icmp "sle"
// LLVM: llvm.icmp "sge"
// LLVM: llvm.shl
// LLVM: llvm.inttoptr
// LLVM: llvm.call @rcgmp_mpz_from_words
func.func @from_signed(%x: i64) -> !rcgmp.mpz {
  %r = rcgmp.from_int %x : i64 to !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @from_unsigned(%x: i64) -> !rcgmp.mpz {
  %r = rcgmp.from_int %x unsigned : i64 to !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @from_atomic(%x: i64) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.from_int %x : i64 to !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
// SCF-LABEL: func.func @retain_local
// SCF: rcgmp.bridge.mpz.is_immediate
// SCF: scf.if
// SCF: rcgmp.bridge.mpz.is_immortal
// SCF: scf.if
// SCF: rcgmp.bridge.mpz.inc_ref
// SCF: cf.assert
// LLVM-LABEL: llvm.func @retain_local
// LLVM: llvm.load
// LLVM: llvm.icmp "eq"
// LLVM: llvm.cond_br
// LLVM: llvm.load
// LLVM: llvm.add
// LLVM: llvm.store
func.func @retain_local(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  rcgmp.retain %x : !rcgmp.mpz
  return %x : !rcgmp.mpz
}
// LLVM-LABEL: llvm.func @retain_atomic
// LLVM: llvm.load {{.*}} atomic monotonic
// LLVM: llvm.atomicrmw add {{.*}} monotonic
func.func @retain_atomic(%x: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  rcgmp.retain %x : !rcgmp.mpz<atomic>
  return %x : !rcgmp.mpz<atomic>
}
// SCF-LABEL: func.func @release_local
// SCF: rcgmp.bridge.mpz.is_immediate
// SCF: rcgmp.bridge.mpz.is_immortal
// SCF: rcgmp.bridge.mpz.dec_ref
// SCF: scf.if
// SCF: rcgmp.bridge.mpz.free
// LLVM-LABEL: llvm.func @release_local
// LLVM: llvm.load
// LLVM: llvm.sub
// LLVM: llvm.store
// LLVM: llvm.call @rcgmp_mpz_free
func.func @release_local(%x: !rcgmp.mpz) {
  rcgmp.release %x : !rcgmp.mpz
  return
}
// LLVM-LABEL: llvm.func @release_atomic
// LLVM: llvm.load {{.*}} atomic monotonic
// LLVM: llvm.atomicrmw sub {{.*}} release
// LLVM: llvm.call @rcgmp_mpz_free
func.func @release_atomic(%x: !rcgmp.mpz<atomic>) {
  rcgmp.release %x : !rcgmp.mpz<atomic>
  return
}
func.func @from_min129() -> !rcgmp.mpz {
  %x = arith.constant -340282366920938463463374607431768211456 : i129
  %r = rcgmp.from_int %x : i129 to !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @from_max129() -> !rcgmp.mpz {
  %x = arith.constant 680564733841876926926749214863536422911 : i129
  %r = rcgmp.from_int %x unsigned : i129 to !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @from_signed_i1() -> !rcgmp.mpz {
  %x = arith.constant true
  %r = rcgmp.from_int %x : i1 to !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @from_unsigned_i1() -> !rcgmp.mpz {
  %x = arith.constant true
  %r = rcgmp.from_int %x unsigned : i1 to !rcgmp.mpz
  return %r : !rcgmp.mpz
}
