// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf | %FileCheck %s --check-prefix=SCF
// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts | %FileCheck %s --check-prefix=LLVM
// Entry points tested by division.cpp.

// SCF-DAG: func.func private @rcgmp_mpz_tdiv_q(!rcgmp.mpz, !rcgmp.mpz, i32) -> !rcgmp.mpz
// SCF-DAG: func.func private @rcgmp_mpz_tdiv_r_atomic(!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>, i32) -> !rcgmp.mpz<atomic>
// SCF-DAG: func.func private @rcgmp_mpz_fdiv_q(!rcgmp.mpz, !rcgmp.mpz, i32) -> !rcgmp.mpz
// SCF-DAG: func.func private @rcgmp_mpz_fdiv_r(!rcgmp.mpz, !rcgmp.mpz, i32) -> !rcgmp.mpz
// SCF-DAG: func.func private @rcgmp_mpz_tdiv_qr(!rcgmp.mpz, !rcgmp.mpz, i32) -> (!rcgmp.mpz, !rcgmp.mpz)
// SCF-DAG: func.func private @rcgmp_mpz_fdiv_qr_atomic(!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>, i32) -> (!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>)

// A zero divisor is replaced by one and invalidates the fast path, as does
// the quotient of the immediate minimum by -1.
// SCF-LABEL: func.func @divsi_local
// SCF: %[[LHS:.*]] = rcgmp.bridge.mpz.to_bits %arg0
// SCF: %[[RHS:.*]] = rcgmp.bridge.mpz.to_bits %arg1
// SCF: %[[TWO:.*]] = arith.constant 2 : i64
// SCF: %[[ZERO:.*]] = arith.constant 0 : i64
// SCF: %[[X:.*]] = arith.shrsi %[[LHS]], %[[TWO]]
// SCF: %[[Y:.*]] = arith.shrsi %[[RHS]], %[[TWO]]
// SCF: %[[NONZERO:.*]] = arith.cmpi ne, %[[Y]], %[[ZERO]]
// SCF: %[[DIVISOR:.*]] = arith.select %[[NONZERO]], %[[Y]]
// SCF: %[[Q:.*]] = arith.divsi %[[X]], %[[DIVISOR]]
// SCF: %[[MAX:.*]] = arith.constant 2305843009213693951 : i64
// SCF: arith.cmpi sle, %[[Q]], %[[MAX]]
// SCF: scf.if
// SCF: func.call @rcgmp_mpz_tdiv_q
// LLVM-LABEL: llvm.func @divsi_local
// LLVM: llvm.sdiv
// LLVM: llvm.call @rcgmp_mpz_tdiv_q

// The floored remainder moves by one divisor when its sign differs.
// SCF-LABEL: func.func @floormodsi_local
// SCF: %[[R:.*]] = arith.remsi
// SCF: arith.cmpi ne, %[[R]]
// SCF: %[[SIGNS:.*]] = arith.xori %[[R]]
// SCF: arith.cmpi slt, %[[SIGNS]]
// SCF: %[[ADJUST:.*]] = arith.andi
// SCF: arith.select %[[ADJUST]]
// SCF-NOT: arith.cmpi sle
// SCF: scf.if

// assume_nonzero divides by the untagged divisor directly.
// SCF-LABEL: func.func @divsi_assume
// SCF: arith.shrsi
// SCF: %[[Y:.*]] = arith.shrsi
// SCF-NOT: arith.select
// SCF: arith.divsi %{{.*}}, %[[Y]]

// Both results come from one runtime call, returned as a pair of pointers.
// SCF-LABEL: func.func @divremsi_local
// SCF: scf.if %{{.*}} -> (!rcgmp.mpz, !rcgmp.mpz)
// SCF: func.call @rcgmp_mpz_tdiv_qr({{.*}}) : (!rcgmp.mpz, !rcgmp.mpz, i32) -> (!rcgmp.mpz, !rcgmp.mpz)
// LLVM-LABEL: llvm.func @divremsi_local
// LLVM: llvm.call @rcgmp_mpz_tdiv_qr({{.*}}) : (!llvm.ptr, !llvm.ptr, i32) -> !llvm.struct<(ptr, ptr)>
func.func @divsi_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.divsi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @divsi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.divsi %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @remsi_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.remsi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @remsi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.remsi %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @floordivsi_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.floordivsi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @floordivsi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.floordivsi %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @floormodsi_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.floormodsi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @floormodsi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.floormodsi %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @divsi_assume(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.divsi %x, %y assume_nonzero : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @divremsi_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> (!rcgmp.mpz, !rcgmp.mpz) {
  %q, %r = rcgmp.divremsi %x, %y : !rcgmp.mpz
  return %q, %r : !rcgmp.mpz, !rcgmp.mpz
}
func.func @divremsi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> (!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>) {
  %q, %r = rcgmp.divremsi %x, %y : !rcgmp.mpz<atomic>
  return %q, %r : !rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>
}
func.func @floordivmodsi_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> (!rcgmp.mpz, !rcgmp.mpz) {
  %q, %r = rcgmp.floordivmodsi %x, %y : !rcgmp.mpz
  return %q, %r : !rcgmp.mpz, !rcgmp.mpz
}
func.func @floordivmodsi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> (!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>) {
  %q, %r = rcgmp.floordivmodsi %x, %y : !rcgmp.mpz<atomic>
  return %q, %r : !rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>
}
