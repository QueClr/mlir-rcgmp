// RUN: %rcgmp-opt %s | %rcgmp-opt | %FileCheck %s

// CHECK-LABEL: func.func @hashes
// CHECK: rcgmp.hashi %{{.*}} : !rcgmp.mpz
// CHECK: rcgmp.hashi %{{.*}} : <atomic>
func.func @hashes(%x: !rcgmp.mpz, %y: !rcgmp.mpz<atomic>) -> (i64, i64) {
  %a = rcgmp.hashi %x : !rcgmp.mpz
  %b = rcgmp.hashi %y : !rcgmp.mpz<atomic>
  rcgmp.release %x : !rcgmp.mpz
  rcgmp.release %y : !rcgmp.mpz<atomic>
  return %a, %b : i64, i64
}
