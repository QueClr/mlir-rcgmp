// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf | %FileCheck %s --check-prefix=SCF
// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts | %FileCheck %s --check-prefix=LLVM
// Entry points compared against GMP by log2.cpp and the differential tests.

// An immediate yields its absolute value, and a heap integer its most
// significant limb and the number of limbs below it; one count of leading
// zeros follows, and no runtime function is called.
// SCF-NOT: func.func private
// SCF-LABEL: func.func @log2i_local
// SCF-SAME: (%[[X:[^:]*]]: !rcgmp.mpz {rcgmp.borrowed})
// SCF: %[[IMMEDIATE:.*]] = rcgmp.bridge.mpz.is_immediate %[[X]]
// SCF: %[[R:.*]]:2 = scf.if %[[IMMEDIATE]] -> (i64, i64) {
// SCF: %[[BITS:.*]] = rcgmp.bridge.mpz.to_bits %[[X]]
// SCF: %[[VALUE:.*]] = arith.shrsi %[[BITS]], %{{.*}} : i64
// SCF: %[[ABS:.*]] = math.absi %[[VALUE]] : i64
// SCF: scf.yield %{{.*}}, %[[ABS]] : i64, i64
// SCF: } else {
// SCF: %[[SIZE:.*]] = rcgmp.bridge.mpz.size %[[X]]
// SCF: %[[COUNT:.*]] = math.absi %[[SIZE]] : i64
// SCF: %[[TOP:.*]] = arith.subi %[[COUNT]], %{{.*}} : i64
// SCF: %[[INDEX:.*]] = arith.index_castui %[[TOP]] : i64 to index
// SCF: %[[LIMB:.*]] = rcgmp.bridge.mpz.limb %[[X]][%[[INDEX]]]
// SCF: scf.yield %[[TOP]], %[[LIMB]] : i64, i64
// SCF: %[[M:.*]] = arith.ori %[[R]]#1, %{{.*}} : i64
// SCF: %[[ZEROS:.*]] = math.ctlz %[[M]] : i64
// SCF: %[[LOW:.*]] = arith.subi %{{.*}}, %[[ZEROS]] : i64
// SCF: %[[HIGH:.*]] = arith.muli %[[R]]#0, %{{.*}} : i64
// SCF: %[[LOG:.*]] = arith.addi %[[HIGH]], %[[LOW]] : i64
// SCF: return %[[LOG]]
// A narrow result is truncated, and a wide one is computed at its width.
// SCF-LABEL: func.func @log2i_8
// SCF: arith.trunci %{{.*}} : i64 to i8
// SCF-LABEL: func.func @log2i_128
// SCF: math.ctlz %{{.*}} : i64
// SCF: arith.extui %{{.*}} : i64 to i128
// SCF: arith.extui %{{.*}} : i64 to i128
// SCF: arith.muli %{{.*}} : i128
// LLVM-LABEL: llvm.func @log2i_atomic
// LLVM-NOT: llvm.call
// LLVM: "llvm.intr.ctlz"
// LLVM-NOT: llvm.call
// LLVM: llvm.return
func.func @log2i_local(%x: !rcgmp.mpz {rcgmp.borrowed}) -> i64 {
  %r = rcgmp.log2i %x : !rcgmp.mpz to i64
  return %r : i64
}
func.func @log2i_atomic(%x: !rcgmp.mpz<atomic> {rcgmp.borrowed}) -> i64 {
  %r = rcgmp.log2i %x : !rcgmp.mpz<atomic> to i64
  return %r : i64
}
func.func @log2i_8(%x: !rcgmp.mpz {rcgmp.borrowed}) -> i8 {
  %r = rcgmp.log2i %x : !rcgmp.mpz to i8
  return %r : i8
}
func.func @log2i_128(%x: !rcgmp.mpz {rcgmp.borrowed}) -> i128 {
  %r = rcgmp.log2i %x : !rcgmp.mpz to i128
  return %r : i128
}
