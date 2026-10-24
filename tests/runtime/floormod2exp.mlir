// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf | %FileCheck %s --check-prefix=SCF
// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts | %FileCheck %s --check-prefix=LLVM
// Entry points tested by floormod2exp.cpp.

// SCF-DAG: func.func private @rcgmp_mpz_fdiv_r_2exp(!rcgmp.mpz, i64, i32) -> !rcgmp.mpz
// SCF-DAG: func.func private @rcgmp_mpz_fdiv_r_2exp_atomic(!rcgmp.mpz<atomic>, i64, i32) -> !rcgmp.mpz<atomic>

// The tagged bits keep their n + 2 low bits, all of them from n = 62 on, and
// the result is immediate unless its sign bit is set.
// SCF-LABEL: func.func @mod_local
// SCF: %[[BITS:.*]] = rcgmp.bridge.mpz.to_bits %arg0
// SCF: %[[LIMIT:.*]] = arith.constant 62 : i64
// SCF: %[[SHIFT:.*]] = arith.minui %arg1, %[[LIMIT]]
// SCF: %[[FOUR:.*]] = arith.constant 4 : i64
// SCF: %[[POWER:.*]] = arith.shli %[[FOUR]], %[[SHIFT]]
// SCF: %[[ONE:.*]] = arith.constant 1 : i64
// SCF: %[[MASK:.*]] = arith.subi %[[POWER]], %[[ONE]]
// SCF: %[[TAGGED:.*]] = arith.andi %[[BITS]], %[[MASK]]
// SCF: %[[ZERO:.*]] = arith.constant 0 : i64
// SCF: %[[FITS:.*]] = arith.cmpi sge, %[[TAGGED]], %[[ZERO]]
// SCF: arith.andi %[[FITS]]
// SCF: scf.if
// SCF-NEXT: rcgmp.bridge.mpz.from_bits %[[TAGGED]]
// Only heap integers fail the fast path with an amount up to 61, and their
// remainder comes from the low limb, after which they are released.
// SCF: %[[SIXTY_ONE:.*]] = arith.constant 61 : i64
// SCF: %[[INLINE:.*]] = arith.cmpi ule, %arg1, %[[SIXTY_ONE]]
// SCF: scf.if %[[INLINE]]
// SCF: %[[SIZE:.*]] = rcgmp.bridge.mpz.size %arg0
// SCF: %[[LIMB:.*]] = rcgmp.bridge.mpz.limb %arg0[%{{.*}}]
// SCF: %[[NEGATIVE:.*]] = arith.cmpi slt, %[[SIZE]]
// SCF: %[[NEGATED:.*]] = arith.subi %{{.*}}, %[[LIMB]]
// SCF: %[[LOW:.*]] = arith.select %[[NEGATIVE]], %[[NEGATED]], %[[LIMB]]
// SCF: %[[POWER:.*]] = arith.shli %{{.*}}, %arg1
// SCF: %[[LOW_MASK:.*]] = arith.subi %[[POWER]]
// SCF: %[[KEPT:.*]] = arith.andi %[[LOW]], %[[LOW_MASK]]
// SCF: %[[SCALED:.*]] = arith.shli %[[KEPT]]
// SCF: %[[RESULT:.*]] = arith.ori %[[SCALED]]
// SCF: rcgmp.bridge.mpz.free %arg0
// SCF: rcgmp.bridge.mpz.dec_ref %arg0
// SCF: rcgmp.bridge.mpz.from_bits %[[RESULT]]
// SCF: } else {
// SCF: rcgmp.bridge.mpz.is_unique
// SCF: func.call @rcgmp_mpz_fdiv_r_2exp(%arg0, %arg1, %{{.*}}) : (!rcgmp.mpz, i64, i32)
// LLVM-LABEL: llvm.func @mod_local
// LLVM: llvm.intr.umin
// LLVM: llvm.call @rcgmp_mpz_fdiv_r_2exp
func.func @mod_local(%x: !rcgmp.mpz, %n: i64) -> !rcgmp.mpz {
  %r = rcgmp.floormod2expi %x, %n : !rcgmp.mpz, i64
  return %r : !rcgmp.mpz
}
// SCF-LABEL: func.func @mod_atomic
// SCF: func.call @rcgmp_mpz_fdiv_r_2exp_atomic
// LLVM-LABEL: llvm.func @mod_atomic
// LLVM: llvm.load {{.*}} atomic acquire
// LLVM: llvm.call @rcgmp_mpz_fdiv_r_2exp_atomic
func.func @mod_atomic(%x: !rcgmp.mpz<atomic>, %n: i64) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.floormod2expi %x, %n : !rcgmp.mpz<atomic>, i64
  return %r : !rcgmp.mpz<atomic>
}

// Constant amounts below 62, which keep every remainder immediate, and from
// 62 on, which keep only those of nonnegative integers.
func.func @mod_by_0(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %n = arith.constant 0 : i64
  %r = rcgmp.floormod2expi %x, %n : !rcgmp.mpz, i64
  return %r : !rcgmp.mpz
}
func.func @mod_by_32(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %n = arith.constant 32 : i64
  %r = rcgmp.floormod2expi %x, %n : !rcgmp.mpz, i64
  return %r : !rcgmp.mpz
}
func.func @mod_by_61(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %n = arith.constant 61 : i64
  %r = rcgmp.floormod2expi %x, %n : !rcgmp.mpz, i64
  return %r : !rcgmp.mpz
}
func.func @mod_by_62(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %n = arith.constant 62 : i64
  %r = rcgmp.floormod2expi %x, %n : !rcgmp.mpz, i64
  return %r : !rcgmp.mpz
}
func.func @mod_by_64(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %n = arith.constant 64 : i8
  %r = rcgmp.floormod2expi %x, %n : !rcgmp.mpz, i8
  return %r : !rcgmp.mpz
}
func.func @mod_by_130_atomic(%x: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %n = arith.constant 130 : i64
  %r = rcgmp.floormod2expi %x, %n : !rcgmp.mpz<atomic>, i64
  return %r : !rcgmp.mpz<atomic>
}

// Amounts are unsigned: a narrow one is zero-extended.
// SCF-LABEL: func.func @mod_narrow
// SCF: %[[AMOUNT:.*]] = arith.extui %arg1 : i8 to i64
// SCF: arith.minui %[[AMOUNT]]
// SCF: func.call @rcgmp_mpz_fdiv_r_2exp(%arg0, %[[AMOUNT]], %{{.*}})
func.func @mod_narrow(%x: !rcgmp.mpz, %n: i8) -> !rcgmp.mpz {
  %r = rcgmp.floormod2expi %x, %n : !rcgmp.mpz, i8
  return %r : !rcgmp.mpz
}

// A wide amount saturates to the largest word.
// SCF-LABEL: func.func @mod_wide
// SCF: %[[MAX:.*]] = arith.constant 18446744073709551615 : i128
// SCF: %[[CLAMPED:.*]] = arith.minui %arg1, %[[MAX]]
// SCF: %[[AMOUNT:.*]] = arith.trunci %[[CLAMPED]] : i128 to i64
// SCF: func.call @rcgmp_mpz_fdiv_r_2exp(%arg0, %[[AMOUNT]], %{{.*}})
func.func @mod_wide(%x: !rcgmp.mpz, %n: i128) -> !rcgmp.mpz {
  %r = rcgmp.floormod2expi %x, %n : !rcgmp.mpz, i128
  return %r : !rcgmp.mpz
}
