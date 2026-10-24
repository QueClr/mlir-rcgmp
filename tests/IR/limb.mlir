// RUN: %rcgmp-opt %s | %rcgmp-opt | %FileCheck %s

// CHECK-LABEL: func.func @limbs
// CHECK: rcgmp.limb_count %{{.*}} : !rcgmp.mpz
// CHECK: rcgmp.limb_count %{{.*}} : <atomic>
// CHECK: rcgmp.limb %{{.*}}[%{{.*}}] : !rcgmp.mpz to i64
// CHECK: rcgmp.limb %{{.*}}[%{{.*}}] : <atomic> to i64
func.func @limbs(%x: !rcgmp.mpz, %y: !rcgmp.mpz<atomic>, %i: index)
    -> (index, index, i64, i64) {
  %a = rcgmp.limb_count %x : !rcgmp.mpz
  %b = rcgmp.limb_count %y : !rcgmp.mpz<atomic>
  %c = rcgmp.limb %x[%i] : !rcgmp.mpz to i64
  %d = rcgmp.limb %y[%i] : !rcgmp.mpz<atomic> to i64
  rcgmp.release %x : !rcgmp.mpz
  rcgmp.release %y : !rcgmp.mpz<atomic>
  return %a, %b, %c, %d : index, index, i64, i64
}
