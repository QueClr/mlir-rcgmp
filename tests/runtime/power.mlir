// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf | %FileCheck %s --check-prefix=SCF
// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts | %FileCheck %s --check-prefix=LLVM
// Entry points tested by power.cpp.

// SCF-DAG: func.func private @rcgmp_mpz_pow(!rcgmp.mpz, i64, i32) -> !rcgmp.mpz
// SCF-DAG: func.func private @rcgmp_mpz_pow_atomic(!rcgmp.mpz<atomic>, i64, i32) -> !rcgmp.mpz<atomic>

// The power is one for exponent zero, the base for odd exponents, and the
// checked square for even ones, valid for exponents up to two and for bases
// of magnitude at most one.
// SCF-LABEL: func.func @pow_local
// SCF: %[[BITS:.*]] = rcgmp.bridge.mpz.to_bits %arg0
// SCF: %[[X:.*]] = arith.shrsi %[[BITS]]
// SCF: %[[SCALED:.*]] = arith.andi %[[BITS]]
// SCF: %[[SQUARE:.*]], %[[OVERFLOW:.*]] = rcgmp.bridge.mulsi_overflow %[[X]], %[[SCALED]]
// SCF: %[[SQUARED:.*]] = arith.ori %[[SQUARE]]
// SCF: %[[ODD:.*]] = arith.trunci %arg1 : i64 to i1
// SCF: %[[POWER:.*]] = arith.select %[[ODD]], %[[BITS]], %[[SQUARED]]
// SCF: %[[ZERO:.*]] = arith.cmpi eq, %arg1
// SCF: %[[TAGGED:.*]] = arith.select %[[ZERO]], %{{.*}}, %[[POWER]]
// SCF: arith.cmpi ule
// SCF: arith.cmpi ule, %arg1
// SCF: arith.cmpi eq, %arg1
// SCF: arith.xori %[[OVERFLOW]]
// SCF: scf.if
// SCF-NEXT: rcgmp.bridge.mpz.from_bits %[[TAGGED]]
// SCF: rcgmp.bridge.mpz.is_unique
// SCF: func.call @rcgmp_mpz_pow(%arg0, %arg1, %{{.*}}) : (!rcgmp.mpz, i64, i32)
// LLVM-LABEL: llvm.func @pow_local
// LLVM: "llvm.intr.smul.with.overflow"
// LLVM: llvm.call @rcgmp_mpz_pow
func.func @pow_local(%x: !rcgmp.mpz, %n: i64) -> !rcgmp.mpz {
  %r = rcgmp.ipowi %x, %n : !rcgmp.mpz, i64
  return %r : !rcgmp.mpz
}
// SCF-LABEL: func.func @pow_atomic
// SCF: func.call @rcgmp_mpz_pow_atomic
// LLVM-LABEL: llvm.func @pow_atomic
// LLVM: llvm.load {{.*}} atomic acquire
// LLVM: llvm.call @rcgmp_mpz_pow_atomic
func.func @pow_atomic(%x: !rcgmp.mpz<atomic>, %n: i64) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.ipowi %x, %n : !rcgmp.mpz<atomic>, i64
  return %r : !rcgmp.mpz<atomic>
}

// A constant exponent unrolls into checked products, left to right: 5 is
// squared, squared again, and multiplied by the base.
// SCF-LABEL: func.func @pow_by_5
// SCF: %[[BITS:.*]] = rcgmp.bridge.mpz.to_bits %arg0
// SCF: %[[X:.*]] = arith.shrsi %[[BITS]]
// SCF: %[[SCALED:.*]] = arith.andi %[[BITS]]
// SCF: %[[S1:.*]] = arith.shrsi %[[SCALED]]
// SCF: %[[P1:.*]], %{{.*}} = rcgmp.bridge.mulsi_overflow %[[S1]], %[[SCALED]]
// SCF: %[[S2:.*]] = arith.shrsi %[[P1]]
// SCF: %[[P2:.*]], %{{.*}} = rcgmp.bridge.mulsi_overflow %[[S2]], %[[P1]]
// SCF: %[[P3:.*]], %{{.*}} = rcgmp.bridge.mulsi_overflow %[[X]], %[[P2]]
// SCF-NOT: mulsi_overflow
// SCF: %[[TAGGED:.*]] = arith.ori %[[P3]]
// SCF: scf.if
// SCF-NEXT: rcgmp.bridge.mpz.from_bits %[[TAGGED]]
// SCF: func.call @rcgmp_mpz_pow
func.func @pow_by_5(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %n = arith.constant 5 : i64
  %r = rcgmp.ipowi %x, %n : !rcgmp.mpz, i64
  return %r : !rcgmp.mpz
}
// 61 is the largest exponent that keeps a base of magnitude two immediate.
// SCF-LABEL: func.func @pow_by_61
// SCF-COUNT-9: rcgmp.bridge.mulsi_overflow
// SCF-NOT: mulsi_overflow
// SCF: func.call @rcgmp_mpz_pow
func.func @pow_by_61(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %n = arith.constant 61 : i64
  %r = rcgmp.ipowi %x, %n : !rcgmp.mpz, i64
  return %r : !rcgmp.mpz
}
// Beyond it, only bases of magnitude at most one stay immediate.
// SCF-LABEL: func.func @pow_by_62
// SCF-COUNT-1: rcgmp.bridge.mulsi_overflow
// SCF-NOT: mulsi_overflow
// SCF: func.call @rcgmp_mpz_pow
func.func @pow_by_62(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %n = arith.constant 62 : i64
  %r = rcgmp.ipowi %x, %n : !rcgmp.mpz, i64
  return %r : !rcgmp.mpz
}
func.func @pow_by_2(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %n = arith.constant 2 : i8
  %r = rcgmp.ipowi %x, %n : !rcgmp.mpz, i8
  return %r : !rcgmp.mpz
}
func.func @pow_by_3_atomic(%x: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %n = arith.constant 3 : i64
  %r = rcgmp.ipowi %x, %n : !rcgmp.mpz<atomic>, i64
  return %r : !rcgmp.mpz<atomic>
}

// Exponents are unsigned: a narrow one is zero-extended.
// SCF-LABEL: func.func @pow_narrow
// SCF: %[[EXPONENT:.*]] = arith.extui %arg1 : i8 to i64
// SCF: func.call @rcgmp_mpz_pow(%arg0, %[[EXPONENT]], %{{.*}})
func.func @pow_narrow(%x: !rcgmp.mpz, %n: i8) -> !rcgmp.mpz {
  %r = rcgmp.ipowi %x, %n : !rcgmp.mpz, i8
  return %r : !rcgmp.mpz
}

// A wide exponent saturates to the largest word of the same parity.
// SCF-LABEL: func.func @pow_wide
// SCF: %[[MAX:.*]] = arith.constant 18446744073709551615 : i128
// SCF: %[[CLAMPED:.*]] = arith.minui %arg1, %[[MAX]]
// SCF: %[[SATURATED:.*]] = arith.trunci %[[CLAMPED]] : i128 to i64
// SCF: %[[LOW:.*]] = arith.trunci %arg1 : i128 to i64
// SCF: %[[PARITY:.*]] = arith.ori %[[LOW]], %{{.*}} : i64
// SCF: %[[EXPONENT:.*]] = arith.andi %[[SATURATED]], %[[PARITY]]
// SCF: func.call @rcgmp_mpz_pow(%arg0, %[[EXPONENT]], %{{.*}})
func.func @pow_wide(%x: !rcgmp.mpz, %n: i128) -> !rcgmp.mpz {
  %r = rcgmp.ipowi %x, %n : !rcgmp.mpz, i128
  return %r : !rcgmp.mpz
}
