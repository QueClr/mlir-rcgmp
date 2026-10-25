// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf | %FileCheck %s --check-prefix=SCF
// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts | %FileCheck %s --check-prefix=LLVM
// Entry points compared with GMP by floating.cpp.

// An immediate converts inline; the runtime rounds a heap integer.
// SCF-DAG: func.func private @rcgmp_mpz_to_f64(!rcgmp.mpz) -> f64
// SCF-DAG: func.func private @rcgmp_mpz_to_f32_atomic(!rcgmp.mpz<atomic>) -> f32
// SCF-DAG: func.func private @rcgmp_mpz_from_f64(f64) -> !rcgmp.mpz
// SCF-DAG: func.func private @rcgmp_mpz_from_f64_atomic(f64) -> !rcgmp.mpz<atomic>
// SCF-LABEL: func.func @sitofp_f64_local
// SCF: %[[BITS:.*]] = rcgmp.bridge.mpz.to_bits %{{.*}}
// SCF: %[[VALUE:.*]] = arith.shrsi %[[BITS]], %{{.*}} : i64
// SCF: %[[FAST:.*]] = arith.sitofp %[[VALUE]] : i64 to f64
// SCF: scf.if %{{.*}} -> (f64) {
// SCF-NEXT: scf.yield %[[FAST]] : f64
// SCF-NEXT: } else {
// SCF-NEXT: func.call @rcgmp_mpz_to_f64
// LLVM-LABEL: llvm.func @sitofp_f32_atomic
// LLVM: llvm.sitofp %{{.*}} : i64 to f32
// LLVM: llvm.call @rcgmp_mpz_to_f32_atomic
func.func @sitofp_f64_local(%x: !rcgmp.mpz) -> f64 {
  %f = rcgmp.sitofp %x : !rcgmp.mpz to f64
  return %f : f64
}
func.func @sitofp_f64_atomic(%x: !rcgmp.mpz<atomic>) -> f64 {
  %f = rcgmp.sitofp %x : !rcgmp.mpz<atomic> to f64
  return %f : f64
}
func.func @sitofp_f32_local(%x: !rcgmp.mpz) -> f32 {
  %f = rcgmp.sitofp %x : !rcgmp.mpz to f32
  return %f : f32
}
func.func @sitofp_f32_atomic(%x: !rcgmp.mpz<atomic>) -> f32 {
  %f = rcgmp.sitofp %x : !rcgmp.mpz<atomic> to f32
  return %f : f32
}

// A magnitude below 2^61 truncates to an immediate inline; the comparison
// fails for a NaN, which the runtime rejects with any other value.
// SCF-LABEL: func.func @fptosi_f64_local
// SCF: %[[BOUND:.*]] = arith.constant 2.305843009213694E+18 : f64
// SCF: %[[ABS:.*]] = math.absf %[[F:.*]] : f64
// SCF: %[[SMALL:.*]] = arith.cmpf olt, %[[ABS]], %[[BOUND]] : f64
// SCF: %[[LIKELY:.*]] = rcgmp.bridge.expect %[[SMALL]], true
// SCF: scf.if %[[LIKELY]] -> (!rcgmp.mpz) {
// SCF: %[[INT:.*]] = arith.fptosi %[[F]] : f64 to i64
// SCF: rcgmp.bridge.mpz.from_bits
// SCF: } else {
// SCF: func.call @rcgmp_mpz_from_f64(%[[F]])
// A single-precision value widens exactly first.
// SCF-LABEL: func.func @fptosi_f32_local
// SCF: arith.extf %{{.*}} : f32 to f64
// LLVM-LABEL: llvm.func @fptosi_f64_atomic
// LLVM: llvm.call @rcgmp_mpz_from_f64_atomic
func.func @fptosi_f64_local(%f: f64) -> !rcgmp.mpz {
  %x = rcgmp.fptosi %f : f64 to !rcgmp.mpz
  return %x : !rcgmp.mpz
}
func.func @fptosi_f64_atomic(%f: f64) -> !rcgmp.mpz<atomic> {
  %x = rcgmp.fptosi %f : f64 to !rcgmp.mpz<atomic>
  return %x : !rcgmp.mpz<atomic>
}
func.func @fptosi_f32_local(%f: f32) -> !rcgmp.mpz {
  %x = rcgmp.fptosi %f : f32 to !rcgmp.mpz
  return %x : !rcgmp.mpz
}
