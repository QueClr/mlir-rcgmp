// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf | %FileCheck %s --check-prefix=SCF
// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts | %FileCheck %s --check-prefix=LLVM
// Entry points tested by division.cpp.

// SCF-DAG: func.func private @rcgmp_mpz_tdiv_q(!rcgmp.mpz, !rcgmp.mpz, i32) -> !rcgmp.mpz
// SCF-DAG: func.func private @rcgmp_mpz_tdiv_r_atomic(!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>, i32) -> !rcgmp.mpz<atomic>
// SCF-DAG: func.func private @rcgmp_mpz_fdiv_q(!rcgmp.mpz, !rcgmp.mpz, i32) -> !rcgmp.mpz
// SCF-DAG: func.func private @rcgmp_mpz_fdiv_r(!rcgmp.mpz, !rcgmp.mpz, i32) -> !rcgmp.mpz
// SCF-DAG: func.func private @rcgmp_mpz_tdiv_qr(!rcgmp.mpz, !rcgmp.mpz, i32) -> (!rcgmp.mpz, !rcgmp.mpz)
// SCF-DAG: func.func private @rcgmp_mpz_fdiv_qr_atomic(!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>, i32) -> (!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>)
// SCF-DAG: func.func private @rcgmp_mpz_ediv_q(!rcgmp.mpz, !rcgmp.mpz, i32) -> !rcgmp.mpz
// SCF-DAG: func.func private @rcgmp_mpz_ediv_r_atomic(!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>, i32) -> !rcgmp.mpz<atomic>
// SCF-DAG: func.func private @rcgmp_mpz_ediv_qr(!rcgmp.mpz, !rcgmp.mpz, i32) -> (!rcgmp.mpz, !rcgmp.mpz)
// SCF-DAG: func.func private @rcgmp_mpz_tdiv_r_lean(!rcgmp.mpz, !rcgmp.mpz, i32) -> !rcgmp.mpz
// SCF-DAG: func.func private @rcgmp_mpz_fdiv_qr_lean_atomic(!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>, i32) -> (!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>)

// A zero divisor is replaced by one and invalidates the fast path, as does
// dividing the immediate minimum by -1, which the operands show before the
// division.
// SCF-LABEL: func.func @divsi_local
// SCF: %[[LHS:.*]] = rcgmp.bridge.mpz.to_bits %arg0
// SCF: %[[RHS:.*]] = rcgmp.bridge.mpz.to_bits %arg1
// SCF: %[[TWO:.*]] = arith.constant 2 : i64
// SCF: %[[ZERO:.*]] = arith.constant 0 : i64
// SCF: %[[X:.*]] = arith.shrsi %[[LHS]], %[[TWO]]
// SCF: %[[Y:.*]] = arith.shrsi %[[RHS]], %[[TWO]]
// SCF: %[[NONZERO:.*]] = arith.cmpi ne, %[[Y]], %[[ZERO]]
// SCF: %[[DIVISOR:.*]] = arith.select %[[NONZERO]], %[[Y]]
// SCF: arith.divsi %[[X]], %[[DIVISOR]]
// SCF: %[[MIN:.*]] = arith.constant -2305843009213693952 : i64
// SCF: arith.cmpi ne, %[[X]], %[[MIN]]
// SCF: %[[MINUS_ONE:.*]] = arith.constant -1 : i64
// SCF: arith.cmpi ne, %[[DIVISOR]], %[[MINUS_ONE]]
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
// SCF-NOT: arith.constant -2305843009213693952
// SCF: scf.if

// An undefined zero divisor divides by the untagged divisor directly.
// SCF-LABEL: func.func @divsi_undefined
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

// Euclidean division divides magnitudes unsigned, a negative dividend
// complemented.
// SCF-LABEL: func.func @edivsi_local
// SCF: %[[X:.*]] = arith.shrsi %{{.*}}, %[[TWO:.*]] : i64
// SCF: %[[SHIFT:.*]] = arith.constant 63 : i64
// SCF: %[[SIGN:.*]] = arith.shrsi %[[X]], %[[SHIFT]]
// SCF: %[[A:.*]] = arith.xori %[[X]], %[[SIGN]]
// SCF: arith.divui %[[A]]
// SCF: scf.if
// SCF: func.call @rcgmp_mpz_ediv_q
// LLVM-LABEL: llvm.func @edivsi_local
// LLVM: llvm.udiv
// LLVM: llvm.call @rcgmp_mpz_ediv_q

// A positive power-of-two divisor divides as floored division does, and a
// negative one as Euclidean division.
// SCF-LABEL: func.func @edivmodsi_by_8
// SCF-NOT: arith.remui
// SCF: arith.remsi
// SCF: func.call @rcgmp_mpz_ediv_qr
// SCF-LABEL: func.func @edivmodsi_by_minus_8
// SCF-NOT: arith.remsi
// SCF: arith.remui
// SCF: func.call @rcgmp_mpz_ediv_qr

// A lean zero divisor leaves the fast path valid. The word division divides
// by the word minimum instead, which leaves the truncated quotient zero and
// the remainder x.
// SCF-LABEL: func.func @divsi_lean_local
// SCF: %[[ZERO:.*]] = arith.constant 0 : i64
// SCF: arith.shrsi
// SCF: %[[Y:.*]] = arith.shrsi
// SCF: %[[IMMEDIATE:.*]] = arith.cmpi eq
// SCF: %[[NONZERO:.*]] = arith.cmpi ne, %[[Y]], %[[ZERO]]
// SCF: %[[MIN:.*]] = arith.constant -9223372036854775808 : i64
// SCF: %[[DIVISOR:.*]] = arith.select %[[NONZERO]], %[[Y]], %[[MIN]]
// SCF: arith.divsi %{{.*}}, %[[DIVISOR]]
// SCF: %[[VALID:.*]] = arith.andi %[[IMMEDIATE]], %{{.*}} : i1
// SCF: scf.if %[[VALID]]
// SCF: func.call @rcgmp_mpz_tdiv_q_lean(

// Floored and Euclidean division adjust the remainder by the zero divisor
// itself, which leaves it x, but move the quotient of a negative x, which is
// then replaced by zero.
// SCF-LABEL: func.func @floordivmodsi_lean_local
// SCF: arith.shrsi
// SCF: %[[Y:.*]] = arith.shrsi
// SCF: %[[NONZERO:.*]] = arith.cmpi ne, %[[Y]]
// SCF: %[[DIVISOR:.*]] = arith.select %[[NONZERO]], %[[Y]]
// SCF: %[[R:.*]] = arith.remsi %{{.*}}, %[[DIVISOR]]
// SCF: arith.xori %[[R]], %[[Y]]
// SCF: %[[Q:.*]] = arith.subi
// SCF: arith.select %{{.*}}, %[[Y]]
// SCF: arith.select %[[NONZERO]], %[[Q]]
// SCF: func.call @rcgmp_mpz_fdiv_qr_lean(
// SCF-LABEL: func.func @edivsi_lean_atomic
// SCF: %[[NONZERO:.*]] = arith.cmpi ne
// SCF: %[[B:.*]] = arith.subi
// SCF: %[[DIVISOR:.*]] = arith.select %[[NONZERO]], %[[B]]
// SCF: arith.divui %{{.*}}, %[[DIVISOR]]
// SCF: %[[Q:.*]] = arith.subi
// SCF: arith.select %[[NONZERO]], %[[Q]]
// SCF: func.call @rcgmp_mpz_ediv_q_lean_atomic(
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
func.func @divsi_undefined(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.divsi %x, %y divzero<undefined> : !rcgmp.mpz
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
func.func @edivsi_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.edivsi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @edivsi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.edivsi %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @emodsi_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.emodsi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @emodsi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.emodsi %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @edivmodsi_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> (!rcgmp.mpz, !rcgmp.mpz) {
  %q, %r = rcgmp.edivmodsi %x, %y : !rcgmp.mpz
  return %q, %r : !rcgmp.mpz, !rcgmp.mpz
}
func.func @edivmodsi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> (!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>) {
  %q, %r = rcgmp.edivmodsi %x, %y : !rcgmp.mpz<atomic>
  return %q, %r : !rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>
}
func.func @edivmodsi_by_8(%x: !rcgmp.mpz) -> (!rcgmp.mpz, !rcgmp.mpz) {
  %c = rcgmp.constant 8 : !rcgmp.mpz
  %q, %r = rcgmp.edivmodsi %x, %c : !rcgmp.mpz
  return %q, %r : !rcgmp.mpz, !rcgmp.mpz
}
func.func @edivmodsi_by_minus_8(%x: !rcgmp.mpz) -> (!rcgmp.mpz, !rcgmp.mpz) {
  %c = rcgmp.constant -8 : !rcgmp.mpz
  %q, %r = rcgmp.edivmodsi %x, %c : !rcgmp.mpz
  return %q, %r : !rcgmp.mpz, !rcgmp.mpz
}
func.func @divsi_lean_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.divsi %x, %y divzero<lean> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @divsi_lean_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.divsi %x, %y divzero<lean> : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @remsi_lean_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.remsi %x, %y divzero<lean> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @remsi_lean_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.remsi %x, %y divzero<lean> : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @floordivsi_lean_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.floordivsi %x, %y divzero<lean> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @floordivsi_lean_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.floordivsi %x, %y divzero<lean> : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @floormodsi_lean_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.floormodsi %x, %y divzero<lean> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @floormodsi_lean_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.floormodsi %x, %y divzero<lean> : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @divremsi_lean_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> (!rcgmp.mpz, !rcgmp.mpz) {
  %q, %r = rcgmp.divremsi %x, %y divzero<lean> : !rcgmp.mpz
  return %q, %r : !rcgmp.mpz, !rcgmp.mpz
}
func.func @divremsi_lean_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> (!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>) {
  %q, %r = rcgmp.divremsi %x, %y divzero<lean> : !rcgmp.mpz<atomic>
  return %q, %r : !rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>
}
func.func @floordivmodsi_lean_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> (!rcgmp.mpz, !rcgmp.mpz) {
  %q, %r = rcgmp.floordivmodsi %x, %y divzero<lean> : !rcgmp.mpz
  return %q, %r : !rcgmp.mpz, !rcgmp.mpz
}
func.func @floordivmodsi_lean_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> (!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>) {
  %q, %r = rcgmp.floordivmodsi %x, %y divzero<lean> : !rcgmp.mpz<atomic>
  return %q, %r : !rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>
}
func.func @edivsi_lean_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.edivsi %x, %y divzero<lean> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @edivsi_lean_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.edivsi %x, %y divzero<lean> : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @emodsi_lean_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.emodsi %x, %y divzero<lean> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @emodsi_lean_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.emodsi %x, %y divzero<lean> : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @edivmodsi_lean_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> (!rcgmp.mpz, !rcgmp.mpz) {
  %q, %r = rcgmp.edivmodsi %x, %y divzero<lean> : !rcgmp.mpz
  return %q, %r : !rcgmp.mpz, !rcgmp.mpz
}
func.func @edivmodsi_lean_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> (!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>) {
  %q, %r = rcgmp.edivmodsi %x, %y divzero<lean> : !rcgmp.mpz<atomic>
  return %q, %r : !rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>
}
