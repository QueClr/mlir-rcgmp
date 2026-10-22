// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf | %FileCheck %s --check-prefix=SCF
// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts | %FileCheck %s --check-prefix=LLVM
// Entry points tested by shift.cpp.

// SCF-DAG: func.func private @rcgmp_mpz_shl(!rcgmp.mpz, i64, i32) -> !rcgmp.mpz
// SCF-DAG: func.func private @rcgmp_mpz_shl_atomic(!rcgmp.mpz<atomic>, i64, i32) -> !rcgmp.mpz<atomic>
// SCF-DAG: func.func private @rcgmp_mpz_shr(!rcgmp.mpz, i64, i32) -> !rcgmp.mpz
// SCF-DAG: func.func private @rcgmp_mpz_shr_atomic(!rcgmp.mpz<atomic>, i64, i32) -> !rcgmp.mpz<atomic>

// Shifting the tagged bits without their tag left and back detects overflow.
// SCF-LABEL: func.func @shl_local
// SCF: %[[BITS:.*]] = rcgmp.bridge.mpz.to_bits
// SCF: %[[ONE:.*]] = arith.constant 1 : i64
// SCF: %[[SCALED:.*]] = arith.xori %[[BITS]], %[[ONE]]
// SCF: %[[LIMIT:.*]] = arith.constant 63 : i64
// SCF: %[[SHIFT:.*]] = arith.minui %arg1, %[[LIMIT]]
// SCF: %[[SHIFTED:.*]] = arith.shli %[[SCALED]], %[[SHIFT]]
// SCF: %[[BACK:.*]] = arith.shrsi %[[SHIFTED]], %[[SHIFT]]
// SCF: %[[KEPT:.*]] = arith.cmpi eq, %[[BACK]], %[[SCALED]]
// SCF: arith.andi %[[KEPT]]
// SCF: %[[TAGGED:.*]] = arith.ori %[[SHIFTED]], %[[ONE]]
// SCF: scf.if
// SCF-NEXT: rcgmp.bridge.mpz.from_bits %[[TAGGED]]
// SCF: func.call @rcgmp_mpz_shl(%arg0, %arg1, %{{.*}}) : (!rcgmp.mpz, i64, i32)
// LLVM-LABEL: llvm.func @shl_local
// LLVM: llvm.intr.umin
// LLVM: llvm.call @rcgmp_mpz_shl
func.func @shl_local(%x: !rcgmp.mpz, %n: i64) -> !rcgmp.mpz {
  %r = rcgmp.shli %x, %n : !rcgmp.mpz, i64
  return %r : !rcgmp.mpz
}
func.func @shl_atomic(%x: !rcgmp.mpz<atomic>, %n: i64) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.shli %x, %n : !rcgmp.mpz<atomic>, i64
  return %r : !rcgmp.mpz<atomic>
}

// Shifting the tagged bits right and clearing their two low bits leaves the
// shifted payload, which is tagged again.
// SCF-LABEL: func.func @shr_local
// SCF: %[[BITS:.*]] = rcgmp.bridge.mpz.to_bits
// SCF: %[[LIMIT:.*]] = arith.constant 63 : i64
// SCF: %[[SHIFT:.*]] = arith.minui %arg1, %[[LIMIT]]
// SCF: %[[SHIFTED:.*]] = arith.shrsi %[[BITS]], %[[SHIFT]]
// SCF: %[[MASK:.*]] = arith.constant -4 : i64
// SCF: %[[SCALED:.*]] = arith.andi %[[SHIFTED]], %[[MASK]]
// SCF: %[[TAGGED:.*]] = arith.ori %[[SCALED]]
// SCF: scf.if
// SCF-NEXT: rcgmp.bridge.mpz.from_bits %[[TAGGED]]
// SCF: func.call @rcgmp_mpz_shr
// LLVM-LABEL: llvm.func @shr_atomic
// LLVM: llvm.load {{.*}} atomic acquire
// LLVM: llvm.call @rcgmp_mpz_shr_atomic
func.func @shr_local(%x: !rcgmp.mpz, %n: i64) -> !rcgmp.mpz {
  %r = rcgmp.shrsi %x, %n : !rcgmp.mpz, i64
  return %r : !rcgmp.mpz
}
func.func @shr_atomic(%x: !rcgmp.mpz<atomic>, %n: i64) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.shrsi %x, %n : !rcgmp.mpz<atomic>, i64
  return %r : !rcgmp.mpz<atomic>
}

// Amounts are unsigned: a narrow one is zero-extended.
// SCF-LABEL: func.func @shl_narrow
// SCF: %[[AMOUNT:.*]] = arith.extui %arg1 : i8 to i64
// SCF: arith.minui %[[AMOUNT]]
// SCF: func.call @rcgmp_mpz_shl(%arg0, %[[AMOUNT]], %{{.*}})
func.func @shl_narrow(%x: !rcgmp.mpz, %n: i8) -> !rcgmp.mpz {
  %r = rcgmp.shli %x, %n : !rcgmp.mpz, i8
  return %r : !rcgmp.mpz
}

// A wide amount saturates to the largest word.
// SCF-LABEL: func.func @shr_wide
// SCF: %[[MAX:.*]] = arith.constant 18446744073709551615 : i128
// SCF: %[[CLAMPED:.*]] = arith.minui %arg1, %[[MAX]]
// SCF: %[[AMOUNT:.*]] = arith.trunci %[[CLAMPED]] : i128 to i64
// SCF: func.call @rcgmp_mpz_shr(%arg0, %[[AMOUNT]], %{{.*}})
func.func @shr_wide(%x: !rcgmp.mpz, %n: i128) -> !rcgmp.mpz {
  %r = rcgmp.shrsi %x, %n : !rcgmp.mpz, i128
  return %r : !rcgmp.mpz
}
