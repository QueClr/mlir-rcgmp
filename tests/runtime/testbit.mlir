// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf | %FileCheck %s --check-prefix=SCF
// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts | %FileCheck %s --check-prefix=LLVM
// Entry points compared against GMP by testbit.cpp and the differential tests.
// Each returns its i1 result zero-extended to i32, which C receives as an
// int32_t.

// An immediate shifts its tagged bits by the index clamped to 61, which brings
// the bit, or the sign, to bit 2.
// SCF-DAG: func.func private @rcgmp_mpz_tstbit(!rcgmp.mpz, i64) -> i32
// SCF-DAG: func.func private @rcgmp_mpz_tstbit_atomic(!rcgmp.mpz<atomic>, i64) -> i32
// SCF-LABEL: func.func @testbiti_local
// SCF-SAME: (%[[X:.*]]: !rcgmp.mpz, %[[N:.*]]: i64)
// SCF: %[[BITS:.*]] = rcgmp.bridge.mpz.to_bits %[[X]]
// SCF: %[[C61:.*]] = arith.constant 61 : i64
// SCF: %[[SHIFT:.*]] = arith.minui %[[N]], %[[C61]] : i64
// SCF: %[[SHIFTED:.*]] = arith.shrsi %[[BITS]], %[[SHIFT]] : i64
// SCF: %[[C4:.*]] = arith.constant 4 : i64
// SCF: %[[BIT:.*]] = arith.andi %[[SHIFTED]], %[[C4]] : i64
// SCF: %[[FAST:.*]] = arith.cmpi ne, %[[BIT]], %{{.*}} : i64
// SCF: scf.if %{{.*}} -> (i1) {
// SCF-NEXT: scf.yield %[[FAST]] : i1
// SCF-NEXT: } else {
// A heap integer reads the limb that holds the bit, unless the index lies
// beyond its magnitude, where the bit is the sign.
// SCF: %[[POSITION:.*]] = arith.divui %[[N]], %{{.*}} : i64
// SCF: %[[OFFSET:.*]] = arith.remui %[[N]], %{{.*}} : i64
// SCF: %[[INDEX:.*]] = arith.index_castui %[[POSITION]] : i64 to index
// SCF: %[[SIZE:.*]] = rcgmp.bridge.mpz.size %[[X]]
// SCF: %[[NEGATIVE:.*]] = arith.cmpi slt, %[[SIZE]], %{{.*}} : i64
// SCF: scf.if %[[NEGATIVE]] -> (i1) {
// SCF: %[[COUNT:.*]] = arith.subi %{{.*}}, %[[SIZE]] : i64
// SCF: %[[INSIDE:.*]] = arith.cmpi ult, %[[POSITION]], %[[COUNT]] : i64
// SCF: scf.if %[[INSIDE]] -> (i1) {
// SCF: %[[LIMB:.*]] = rcgmp.bridge.mpz.limb %[[X]][%[[INDEX]]]
// A negative integer complements the limb, adding one at position zero, and
// scans its lower limbs in the runtime only when limb 0 is zero.
// SCF: %[[FIRST:.*]] = arith.cmpi eq, %[[POSITION]], %{{.*}} : i64
// SCF: %[[LOW:.*]] = rcgmp.bridge.mpz.limb %[[X]][%{{.*}}]
// SCF: %[[LOWZERO:.*]] = arith.cmpi eq, %[[LOW]], %{{.*}} : i64
// SCF: %[[ABOVE:.*]] = arith.xori %[[FIRST]], %{{.*}} : i1
// SCF: %[[SCAN:.*]] = arith.andi %[[ABOVE]], %[[LOWZERO]] : i1
// SCF: scf.if %[[SCAN]] -> (i1) {
// SCF: func.call @rcgmp_mpz_tstbit(%[[X]], %[[N]])
// SCF: } else {
// SCF: %[[COMPLEMENT:.*]] = arith.xori %[[LIMB]], %{{.*}} : i64
// SCF: %[[CARRY:.*]] = arith.extui %[[FIRST]] : i1 to i64
// SCF: %[[TWOS:.*]] = arith.addi %[[COMPLEMENT]], %[[CARRY]] : i64
// SCF: %[[SHIFTED:.*]] = arith.shrui %[[TWOS]], %[[OFFSET]] : i64
// SCF: arith.trunci %[[SHIFTED]] : i64 to i1
// SCF: } else {
// SCF-NEXT: %[[TRUE:.*]] = arith.constant true
// SCF-NEXT: scf.yield %[[TRUE]] : i1
// SCF: } else {
// SCF: %[[INSIDE:.*]] = arith.cmpi ult, %[[POSITION]], %[[SIZE]] : i64
// SCF: scf.if %[[INSIDE]] -> (i1) {
// SCF: %[[LIMB:.*]] = rcgmp.bridge.mpz.limb %[[X]][%[[INDEX]]]
// SCF: %[[SHIFTED:.*]] = arith.shrui %[[LIMB]], %[[OFFSET]] : i64
// SCF: arith.trunci %[[SHIFTED]] : i64 to i1
// SCF: } else {
// SCF-NEXT: %[[FALSE:.*]] = arith.constant false
// SCF-NEXT: scf.yield %[[FALSE]] : i1
// Indices are zero-extended or saturated to the word.
// SCF-LABEL: func.func @testbiti_8
// SCF: arith.extui %{{.*}} : i8 to i64
// SCF-LABEL: func.func @testbiti_128
// SCF: arith.minui %{{.*}} : i128
// SCF: arith.trunci %{{.*}} : i128 to i64
// LLVM-LABEL: llvm.func @testbiti_atomic
// LLVM-COUNT-1: llvm.call @rcgmp_mpz_tstbit_atomic
// LLVM-NOT: llvm.call
// LLVM: llvm.return
func.func @testbiti_local(%x: !rcgmp.mpz, %n: i64) -> i32 {
  %r = rcgmp.testbiti %x, %n : !rcgmp.mpz, i64
  %e = arith.extui %r : i1 to i32
  return %e : i32
}
func.func @testbiti_atomic(%x: !rcgmp.mpz<atomic>, %n: i64) -> i32 {
  %r = rcgmp.testbiti %x, %n : !rcgmp.mpz<atomic>, i64
  %e = arith.extui %r : i1 to i32
  return %e : i32
}
func.func @testbiti_8(%x: !rcgmp.mpz, %n: i8) -> i32 {
  %r = rcgmp.testbiti %x, %n : !rcgmp.mpz, i8
  %e = arith.extui %r : i1 to i32
  return %e : i32
}
func.func @testbiti_128(%x: !rcgmp.mpz, %n: i128) -> i32 {
  %r = rcgmp.testbiti %x, %n : !rcgmp.mpz, i128
  %e = arith.extui %r : i1 to i32
  return %e : i32
}
