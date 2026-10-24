// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf | %FileCheck %s --check-prefix=SCF
// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts | %FileCheck %s --check-prefix=LLVM
// Entry points compared with GMP by floating.cpp.

// An immediate converts inline; the runtime rounds a heap integer.
// SCF-DAG: func.func private @rcgmp_mpz_to_f64(!rcgmp.mpz) -> f64
// SCF-DAG: func.func private @rcgmp_mpz_to_f32_atomic(!rcgmp.mpz<atomic>) -> f32
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

