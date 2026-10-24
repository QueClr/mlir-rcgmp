// RUN: %rcgmp-opt %s | %rcgmp-opt | %FileCheck %s

// CHECK-LABEL: func.func @conversions
// CHECK: rcgmp.sitofp %{{.*}} : !rcgmp.mpz to f64
// CHECK: rcgmp.sitofp %{{.*}} : <atomic> to f32
// CHECK: rcgmp.fptosi %{{.*}} : f64 to !rcgmp.mpz
// CHECK: rcgmp.fptosi %{{.*}} : f32 to <atomic>
func.func @conversions(%x: !rcgmp.mpz, %y: !rcgmp.mpz<atomic>, %f: f64,
                       %g: f32) -> (f64, f32, !rcgmp.mpz, !rcgmp.mpz<atomic>) {
  %a = rcgmp.sitofp %x : !rcgmp.mpz to f64
  %b = rcgmp.sitofp %y : !rcgmp.mpz<atomic> to f32
  %c = rcgmp.fptosi %f : f64 to !rcgmp.mpz
  %d = rcgmp.fptosi %g : f32 to !rcgmp.mpz<atomic>
  rcgmp.release %x : !rcgmp.mpz
  rcgmp.release %y : !rcgmp.mpz<atomic>
  return %a, %b, %c, %d : f64, f32, !rcgmp.mpz, !rcgmp.mpz<atomic>
}
