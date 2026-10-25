// RUN: %rcgmp-opt %s -canonicalize | %FileCheck %s
// Entry points tested by zero-remainder.cpp, which are canonicalized before
// lowering: remainders by plus or minus 2^k that are only compared with zero
// take the low k bits. Each returns its i1 result zero-extended to i32, which
// C receives as an int32_t.

// CHECK-LABEL: func.func @even
// CHECK: rcgmp.floormod2expi
// CHECK-NOT: rcgmp.remsi
func.func @even(%x: !rcgmp.mpz) -> i32 {
  %zero = rcgmp.constant 0 : !rcgmp.mpz
  %two = rcgmp.constant 2 : !rcgmp.mpz
  %r = rcgmp.remsi %x, %two : !rcgmp.mpz
  %t = rcgmp.cmpi eq, %r, %zero : !rcgmp.mpz
  rcgmp.release %r : !rcgmp.mpz
  %e = arith.extui %t : i1 to i32
  return %e : i32
}

// CHECK-LABEL: func.func @not_multiple_atomic
// CHECK: rcgmp.floormod2expi
// CHECK-NOT: rcgmp.floormodsi
func.func @not_multiple_atomic(%x: !rcgmp.mpz<atomic>) -> i32 {
  %zero = rcgmp.constant 0 : !rcgmp.mpz<atomic>
  %divisor = rcgmp.constant -1180591620717411303424 : !rcgmp.mpz<atomic>
  %r = rcgmp.floormodsi %x, %divisor : !rcgmp.mpz<atomic>
  %t = rcgmp.cmpi ne, %zero, %r : !rcgmp.mpz<atomic>
  rcgmp.release %r : !rcgmp.mpz<atomic>
  %e = arith.extui %t : i1 to i32
  return %e : i32
}
