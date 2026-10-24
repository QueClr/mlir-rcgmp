// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf | %FileCheck %s --check-prefix=SCF
// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts | %FileCheck %s --check-prefix=LLVM
// Immediates divide the larger of 4*|x| and 4*|y| by the smaller when it is
// more than 2^8 times larger, loop over their odd parts, replacing the larger
// by the odd part of the difference, and restore the shared power of two.
// SCF-LABEL: func.func @gcd_local
// SCF: %[[X:.*]] = rcgmp.bridge.mpz.to_bits %arg0
// SCF: %[[Y:.*]] = rcgmp.bridge.mpz.to_bits %arg1
// SCF: %[[SCALED:.*]] = scf.if
// SCF: math.absi
// SCF: math.absi
// SCF: %[[REDUCED:.*]]:2 = scf.if
// SCF: arith.remui
// SCF: } else {
// SCF: %[[BOTH:.*]] = arith.ori %[[REDUCED]]#0, %[[REDUCED]]#1
// SCF: %[[TOP:.*]] = arith.ori %[[BOTH]]
// SCF: %[[SHIFT:.*]] = math.cttz %[[TOP]]
// SCF: %[[ODD:.*]]:3 = scf.while
// SCF: arith.subi
// SCF: scf.condition
// SCF: arith.minui
// SCF: math.cttz
// SCF: arith.shrui
// SCF: scf.yield
// SCF: arith.shli %[[ODD]]#1, %[[SHIFT]]
// SCF: } else {
// SCF: arith.cmpi sge, %[[SCALED]]
// SCF: %[[TAGGED:.*]] = arith.ori %[[SCALED]]
// SCF: scf.if
// SCF-NEXT: rcgmp.bridge.mpz.from_bits %[[TAGGED]]
// SCF: rcgmp.bridge.mpz.is_unique
// SCF: func.call @rcgmp_mpz_gcd(
// LLVM-LABEL: llvm.func @gcd_local
// LLVM: llvm.intr.abs
// LLVM: llvm.urem
// LLVM: llvm.intr.cttz
// LLVM: llvm.call @rcgmp_mpz_gcd(
func.func @gcd_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.gcdi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
// SCF-LABEL: func.func @gcd_atomic
// SCF: func.call @rcgmp_mpz_gcd_atomic
// LLVM-LABEL: llvm.func @gcd_atomic
// LLVM: llvm.load {{.*}} atomic acquire
// LLVM: llvm.call @rcgmp_mpz_gcd_atomic
func.func @gcd_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.gcdi %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
