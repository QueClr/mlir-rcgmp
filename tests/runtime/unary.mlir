// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf | %FileCheck %s --check-prefix=SCF
// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts | %FileCheck %s --check-prefix=LLVM
// Entry points tested by unary.cpp.

// SCF-DAG: func.func private @rcgmp_mpz_not(!rcgmp.mpz, i32) -> !rcgmp.mpz
// SCF-DAG: func.func private @rcgmp_mpz_neg_atomic(!rcgmp.mpz<atomic>, i32) -> !rcgmp.mpz<atomic>
// SCF-DAG: func.func private @rcgmp_mpz_abs(!rcgmp.mpz, i32) -> !rcgmp.mpz

// Complementing the payload keeps the tag, so only the tag is tested.
// SCF-LABEL: func.func @not_local
// SCF: %[[BITS:.*]] = rcgmp.bridge.mpz.to_bits
// SCF: %[[MASK:.*]] = arith.constant -4 : i64
// SCF: %[[NOT:.*]] = arith.xori %[[BITS]], %[[MASK]]
// SCF-NOT: arith.cmpi ne
// SCF: scf.if
// SCF-NEXT: rcgmp.bridge.mpz.from_bits %[[NOT]]
// SCF: rcgmp.bridge.mpz.is_unique
// SCF: func.call @rcgmp_mpz_not
// LLVM-LABEL: llvm.func @not_local
// LLVM: llvm.call @rcgmp_mpz_not
func.func @not_local(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.noti %x : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @not_atomic(%x: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.noti %x : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}

// Negation is 2 minus the tagged bits, except at the tagged minimum.
// SCF-LABEL: func.func @neg_local
// SCF: %[[BITS:.*]] = rcgmp.bridge.mpz.to_bits
// SCF: %[[TWO:.*]] = arith.constant 2 : i64
// SCF: %[[NEG:.*]] = arith.subi %[[TWO]], %[[BITS]]
// SCF: %[[MIN:.*]] = arith.constant -9223372036854775807 : i64
// SCF: arith.cmpi ne, %[[BITS]], %[[MIN]]
// SCF: scf.if
// SCF-NEXT: rcgmp.bridge.mpz.from_bits %[[NEG]]
// SCF: func.call @rcgmp_mpz_neg
// LLVM-LABEL: llvm.func @neg_atomic
// LLVM: llvm.load {{.*}} atomic acquire
// LLVM: llvm.call @rcgmp_mpz_neg_atomic
func.func @neg_local(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.negi %x : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @neg_atomic(%x: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.negi %x : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}

// The absolute value selects the negation of negative tagged bits.
// SCF-LABEL: func.func @abs_local
// SCF: %[[BITS:.*]] = rcgmp.bridge.mpz.to_bits
// SCF: %[[NEG:.*]] = arith.subi %{{.*}}, %[[BITS]]
// SCF: arith.cmpi ne, %[[BITS]], %{{.*}}
// SCF: %[[NEGATIVE:.*]] = arith.cmpi slt, %[[BITS]], %{{.*}}
// SCF: %[[ABS:.*]] = arith.select %[[NEGATIVE]], %[[NEG]], %[[BITS]]
// SCF: scf.if
// SCF-NEXT: rcgmp.bridge.mpz.from_bits %[[ABS]]
// SCF: func.call @rcgmp_mpz_abs
// LLVM-LABEL: llvm.func @abs_local
// LLVM: llvm.select
// LLVM: llvm.call @rcgmp_mpz_abs
func.func @abs_local(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.absi %x : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @abs_atomic(%x: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.absi %x : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
