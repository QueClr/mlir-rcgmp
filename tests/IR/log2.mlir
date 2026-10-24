// RUN: %rcgmp-opt %s | %rcgmp-opt | %FileCheck %s

// CHECK-LABEL: func.func @logarithms
// CHECK: rcgmp.log2i %{{.*}} : !rcgmp.mpz to i64
// CHECK: rcgmp.log2i %{{.*}} : <atomic> to i8
// CHECK: rcgmp.log2i %{{.*}} : !rcgmp.mpz to i128
func.func @logarithms(%x: !rcgmp.mpz, %y: !rcgmp.mpz<atomic>) {
  %a = rcgmp.log2i %x : !rcgmp.mpz to i64
  %b = rcgmp.log2i %y : !rcgmp.mpz<atomic> to i8
  %c = rcgmp.log2i %x : !rcgmp.mpz to i128
  rcgmp.release %x : !rcgmp.mpz
  rcgmp.release %y : !rcgmp.mpz<atomic>
  return
}
