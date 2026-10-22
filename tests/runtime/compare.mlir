// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf | %FileCheck %s --check-prefix=SCF
// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts | %FileCheck %s --check-prefix=LLVM
// Entry points compared against GMP by compare.cpp. Each returns its i1 result
// zero-extended to i32, which C receives as an int32_t.

// Borrowed operands need no uniqueness test, and the runtime order is compared
// with zero.
// SCF-DAG: func.func private @rcgmp_mpz_cmp(!rcgmp.mpz, !rcgmp.mpz) -> i32
// SCF-DAG: func.func private @rcgmp_mpz_cmp_atomic(!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>) -> i32
// SCF-LABEL: func.func @cmpi_slt_local
// SCF-SAME: (%[[X:.*]]: !rcgmp.mpz, %[[Y:.*]]: !rcgmp.mpz)
// SCF: %[[XBITS:.*]] = rcgmp.bridge.mpz.to_bits %[[X]]
// SCF: %[[YBITS:.*]] = rcgmp.bridge.mpz.to_bits %[[Y]]
// SCF: %[[FAST:.*]] = arith.cmpi slt, %[[XBITS]], %[[YBITS]]
// SCF: scf.if %{{.*}} -> (i1) {
// SCF-NEXT: scf.yield %[[FAST]] : i1
// SCF-NEXT: } else {
// SCF-NEXT: %[[ORDER:.*]] = func.call @rcgmp_mpz_cmp(%[[X]], %[[Y]])
// SCF-NEXT: %[[ZERO:.*]] = arith.constant 0 : i32
// SCF-NEXT: %[[SLOW:.*]] = arith.cmpi slt, %[[ORDER]], %[[ZERO]] : i32
// SCF-NEXT: scf.yield %[[SLOW]] : i1
// LLVM-LABEL: llvm.func @cmpi_sge_atomic
// LLVM: llvm.icmp "sge"
// LLVM-NOT: atomic
// LLVM: llvm.call @rcgmp_mpz_cmp_atomic
func.func @cmpi_eq_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> i32 {
  %r = rcgmp.cmpi eq, %x, %y : !rcgmp.mpz
  %e = arith.extui %r : i1 to i32
  return %e : i32
}
func.func @cmpi_eq_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> i32 {
  %r = rcgmp.cmpi eq, %x, %y : !rcgmp.mpz<atomic>
  %e = arith.extui %r : i1 to i32
  return %e : i32
}
func.func @cmpi_ne_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> i32 {
  %r = rcgmp.cmpi ne, %x, %y : !rcgmp.mpz
  %e = arith.extui %r : i1 to i32
  return %e : i32
}
func.func @cmpi_ne_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> i32 {
  %r = rcgmp.cmpi ne, %x, %y : !rcgmp.mpz<atomic>
  %e = arith.extui %r : i1 to i32
  return %e : i32
}
func.func @cmpi_slt_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> i32 {
  %r = rcgmp.cmpi slt, %x, %y : !rcgmp.mpz
  %e = arith.extui %r : i1 to i32
  return %e : i32
}
func.func @cmpi_slt_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> i32 {
  %r = rcgmp.cmpi slt, %x, %y : !rcgmp.mpz<atomic>
  %e = arith.extui %r : i1 to i32
  return %e : i32
}
func.func @cmpi_sle_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> i32 {
  %r = rcgmp.cmpi sle, %x, %y : !rcgmp.mpz
  %e = arith.extui %r : i1 to i32
  return %e : i32
}
func.func @cmpi_sle_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> i32 {
  %r = rcgmp.cmpi sle, %x, %y : !rcgmp.mpz<atomic>
  %e = arith.extui %r : i1 to i32
  return %e : i32
}
func.func @cmpi_sgt_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> i32 {
  %r = rcgmp.cmpi sgt, %x, %y : !rcgmp.mpz
  %e = arith.extui %r : i1 to i32
  return %e : i32
}
func.func @cmpi_sgt_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> i32 {
  %r = rcgmp.cmpi sgt, %x, %y : !rcgmp.mpz<atomic>
  %e = arith.extui %r : i1 to i32
  return %e : i32
}
func.func @cmpi_sge_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> i32 {
  %r = rcgmp.cmpi sge, %x, %y : !rcgmp.mpz
  %e = arith.extui %r : i1 to i32
  return %e : i32
}
func.func @cmpi_sge_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> i32 {
  %r = rcgmp.cmpi sge, %x, %y : !rcgmp.mpz<atomic>
  %e = arith.extui %r : i1 to i32
  return %e : i32
}
