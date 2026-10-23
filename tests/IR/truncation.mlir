// RUN: %rcgmp-opt %s | %rcgmp-opt | %FileCheck %s

// CHECK-LABEL: func.func @truncations
// CHECK: rcgmp.trunci %{{.*}} : !rcgmp.mpz to i32
// CHECK: rcgmp.trunci %{{.*}} overflow<nsw, nuw> : <atomic> to i8
// CHECK: %{{.*}}, %{{.*}} = rcgmp.trunci_overflow %{{.*}} : !rcgmp.mpz to i128
// CHECK: %{{.*}}, %{{.*}} = rcgmp.trunci_overflow %{{.*}} unsigned : !rcgmp.mpz to i1
func.func @truncations(%x: !rcgmp.mpz, %y: !rcgmp.mpz<atomic>) {
  %a = rcgmp.trunci %x : !rcgmp.mpz to i32
  %b = rcgmp.trunci %y overflow<nsw, nuw> : !rcgmp.mpz<atomic> to i8
  %c, %co = rcgmp.trunci_overflow %x : !rcgmp.mpz to i128
  %d, %do = rcgmp.trunci_overflow %x unsigned : !rcgmp.mpz to i1
  rcgmp.release %x : !rcgmp.mpz
  rcgmp.release %y : !rcgmp.mpz<atomic>
  return
}
