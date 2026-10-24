// RUN: %rcgmp-opt %s | %rcgmp-opt | %FileCheck %s

// CHECK-LABEL: func.func @conversions
// CHECK: rcgmp.sitofp %{{.*}} : !rcgmp.mpz to f64
// CHECK: rcgmp.sitofp %{{.*}} : <atomic> to f32
func.func @conversions(%x: !rcgmp.mpz, %y: !rcgmp.mpz<atomic>) -> (f64, f32) {
  %a = rcgmp.sitofp %x : !rcgmp.mpz to f64
  %b = rcgmp.sitofp %y : !rcgmp.mpz<atomic> to f32
  rcgmp.release %x : !rcgmp.mpz
  rcgmp.release %y : !rcgmp.mpz<atomic>
  return %a, %b : f64, f32
}
