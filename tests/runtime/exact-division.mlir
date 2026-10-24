// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf | %FileCheck %s --check-prefix=SCF
// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts | %FileCheck %s --check-prefix=LLVM
// Entry points tested by exact-division.cpp.

// SCF-DAG: func.func private @rcgmp_mpz_tdiv_q_exact(!rcgmp.mpz, !rcgmp.mpz, i32) -> !rcgmp.mpz
// SCF-DAG: func.func private @rcgmp_mpz_tdiv_q_exact_atomic(!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>, i32) -> !rcgmp.mpz<atomic>

// An exact quotient divides once, without a remainder or a rounding
// adjustment. A trapping zero divisor is replaced by one and invalidates the
// fast path, as does dividing the immediate minimum by -1.
// SCF-LABEL: func.func @divsi_exact_local
// SCF: %[[X:.*]] = arith.shrsi
// SCF: %[[Y:.*]] = arith.shrsi
// SCF: %[[NONZERO:.*]] = arith.cmpi ne, %[[Y]]
// SCF: %[[DIVISOR:.*]] = arith.select %[[NONZERO]], %[[Y]]
// SCF-NOT: arith.remsi
// SCF: arith.divsi %[[X]], %[[DIVISOR]] exact
// SCF-NOT: arith.remsi
// SCF: arith.constant -2305843009213693952 : i64
// SCF: scf.if
// SCF: func.call @rcgmp_mpz_tdiv_q_exact(
// LLVM-LABEL: llvm.func @divsi_exact_local
// LLVM: llvm.sdiv exact
// LLVM: llvm.call @rcgmp_mpz_tdiv_q_exact(

// Floored and Euclidean exact quotients lower as truncated ones.
// SCF-LABEL: func.func @floordivsi_exact_local
// SCF-NOT: arith.remsi
// SCF: arith.divsi {{.*}} exact
// SCF-NOT: arith.remsi
// SCF: func.call @rcgmp_mpz_tdiv_q_exact(
// SCF-LABEL: func.func @edivsi_exact_local
// SCF-NOT: arith.divui
// SCF: arith.divsi {{.*}} exact
// SCF-NOT: arith.select
// SCF: func.call @rcgmp_mpz_tdiv_q_exact(

// An undefined zero divisor divides by the untagged divisor directly.
// SCF-LABEL: func.func @divsi_exact_undefined_local
// SCF: arith.shrsi
// SCF: %[[Y:.*]] = arith.shrsi
// SCF-NOT: arith.select
// SCF: arith.divsi %{{.*}}, %[[Y]] exact

// A lean zero divisor leaves the fast path valid and is replaced by the word
// minimum, of which zero, the only lhs it divides, is a multiple. The runtime
// call is the trapping one.
// SCF-LABEL: func.func @divsi_exact_lean_local
// SCF: %[[NONZERO:.*]] = arith.cmpi ne
// SCF: %[[MIN:.*]] = arith.constant -9223372036854775808 : i64
// SCF: %[[DIVISOR:.*]] = arith.select %[[NONZERO]], %{{.*}}, %[[MIN]]
// SCF: arith.divsi %{{.*}}, %[[DIVISOR]] exact
// SCF-NOT: arith.select
// SCF: func.call @rcgmp_mpz_tdiv_q_exact(
// SCF-LABEL: func.func @edivsi_exact_lean_atomic
// SCF: func.call @rcgmp_mpz_tdiv_q_exact_atomic(

// A constant divisor reaches LLVM as an exact division by its untagged bits,
// which LLVM folds to a constant and reduces to shifts and a multiplication
// by its inverse.
// LLVM-LABEL: llvm.func @divsi_exact_by_12
// LLVM: %[[TAGGED:.*]] = llvm.mlir.constant(49 : i64) : i64
// LLVM: llvm.sdiv exact
func.func @divsi_exact_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.divsi %x, %y exact : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @divsi_exact_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.divsi %x, %y exact : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @floordivsi_exact_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.floordivsi %x, %y exact : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @edivsi_exact_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.edivsi %x, %y exact : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @divsi_exact_undefined_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.divsi %x, %y exact divzero<undefined> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @divsi_exact_undefined_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.divsi %x, %y exact divzero<undefined> : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @divsi_exact_lean_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.divsi %x, %y exact divzero<lean> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @divsi_exact_lean_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.divsi %x, %y exact divzero<lean> : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @edivsi_exact_lean_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.edivsi %x, %y exact divzero<lean> : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @divsi_exact_by_12(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %c = rcgmp.constant 12 : !rcgmp.mpz
  %r = rcgmp.divsi %x, %c exact : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @divsi_exact_by_minus_8(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %c = rcgmp.constant -8 : !rcgmp.mpz
  %r = rcgmp.divsi %x, %c exact : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @divsi_exact_by_minus_7(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %c = rcgmp.constant -7 : !rcgmp.mpz
  %r = rcgmp.divsi %x, %c exact : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
