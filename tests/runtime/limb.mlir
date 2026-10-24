// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf | %FileCheck %s --check-prefix=SCF
// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts | %FileCheck %s --check-prefix=LLVM
// Entry points compared against GMP by limb.cpp and the differential tests.
// Counts and indices cross the C boundary as 64-bit words.

// An immediate has one limb unless it is zero, whose tagged bits are 1, and a
// heap integer has the absolute value of its signed count.
// SCF-LABEL: func.func @limb_count_local
// SCF-SAME: (%[[X:.*]]: !rcgmp.mpz)
// SCF: %[[IMMEDIATE:.*]] = rcgmp.bridge.mpz.is_immediate %[[X]]
// SCF: scf.if %[[IMMEDIATE]] -> (i64) {
// SCF: %[[BITS:.*]] = rcgmp.bridge.mpz.to_bits %[[X]]
// SCF: %[[C1:.*]] = arith.constant 1 : i64
// SCF: %[[NONZERO:.*]] = arith.cmpi ne, %[[BITS]], %[[C1]] : i64
// SCF: arith.extui %[[NONZERO]] : i1 to i64
// SCF: } else {
// SCF: %[[SIZE:.*]] = rcgmp.bridge.mpz.size %[[X]]
// SCF: math.absi %[[SIZE]] : i64
// SCF: arith.index_castui %{{.*}} : i64 to index
// An immediate's limb 0 is the absolute value of its payload. A heap integer
// loads a limb only below its count; every other limb is zero.
// SCF-LABEL: func.func @limb_local
// SCF-SAME: (%[[X:.*]]: !rcgmp.mpz, %{{.*}}: i64)
// SCF: %[[INDEX:.*]] = arith.index_castui %{{.*}} : i64 to index
// SCF: %[[IMMEDIATE:.*]] = rcgmp.bridge.mpz.is_immediate %[[X]]
// SCF: scf.if %[[IMMEDIATE]] -> (i64) {
// SCF: %[[BITS:.*]] = rcgmp.bridge.mpz.to_bits %[[X]]
// SCF: %[[PAYLOAD:.*]] = arith.shrsi %[[BITS]], %{{.*}} : i64
// SCF: %[[FIRST:.*]] = arith.cmpi eq, %[[INDEX]], %{{.*}} : index
// SCF: %[[ABS:.*]] = math.absi %[[PAYLOAD]] : i64
// SCF: arith.select %[[FIRST]], %[[ABS]], %{{.*}} : i64
// SCF: } else {
// SCF: %[[SIZE:.*]] = rcgmp.bridge.mpz.size %[[X]]
// SCF: %[[COUNT:.*]] = arith.index_castui %{{.*}} : i64 to index
// SCF: %[[INSIDE:.*]] = arith.cmpi ult, %[[INDEX]], %[[COUNT]] : index
// SCF: scf.if %[[INSIDE]] -> (i64) {
// SCF: rcgmp.bridge.mpz.limb %[[X]][%[[INDEX]]]
// SCF: } else {
// No path calls the runtime.
// LLVM-LABEL: llvm.func @limb_count_local
// LLVM-NOT: llvm.call
// LLVM-LABEL: llvm.func @limb_count_atomic
// LLVM-NOT: llvm.call
// LLVM-LABEL: llvm.func @limb_local
// LLVM-NOT: llvm.call
// LLVM-LABEL: llvm.func @limb_atomic
// LLVM-NOT: llvm.call
// LLVM: llvm.return
func.func @limb_count_local(%x: !rcgmp.mpz) -> i64 {
  %n = rcgmp.limb_count %x : !rcgmp.mpz
  %r = arith.index_castui %n : index to i64
  return %r : i64
}
func.func @limb_count_atomic(%x: !rcgmp.mpz<atomic>) -> i64 {
  %n = rcgmp.limb_count %x : !rcgmp.mpz<atomic>
  %r = arith.index_castui %n : index to i64
  return %r : i64
}
func.func @limb_local(%x: !rcgmp.mpz, %i: i64) -> i64 {
  %index = arith.index_castui %i : i64 to index
  %r = rcgmp.limb %x[%index] : !rcgmp.mpz to i64
  return %r : i64
}
func.func @limb_atomic(%x: !rcgmp.mpz<atomic>, %i: i64) -> i64 {
  %index = arith.index_castui %i : i64 to index
  %r = rcgmp.limb %x[%index] : !rcgmp.mpz<atomic> to i64
  return %r : i64
}
