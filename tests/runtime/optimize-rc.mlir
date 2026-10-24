// RUN: %rcgmp-opt %s -rcgmp-optimize-rc | %FileCheck %s
// Entry points lowered after -rcgmp-optimize-rc and checked by optimize-rc.cpp.

// CHECK-LABEL: func.func @increment(
// CHECK-NOT: rcgmp.retain
func.func @increment(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %one = rcgmp.constant 1 : !rcgmp.mpz
  rcgmp.retain %x : !rcgmp.mpz
  %s = rcgmp.addi %x, %one : !rcgmp.mpz
  rcgmp.release %x : !rcgmp.mpz
  return %s : !rcgmp.mpz
}

// CHECK-LABEL: func.func @increment_if(
// CHECK-NOT: rcgmp.retain
func.func @increment_if(%x: !rcgmp.mpz, %c: i32) -> !rcgmp.mpz {
  %one = rcgmp.constant 1 : !rcgmp.mpz
  %zero = arith.constant 0 : i32
  %taken = arith.cmpi ne, %c, %zero : i32
  rcgmp.retain %x : !rcgmp.mpz
  %r = scf.if %taken -> !rcgmp.mpz {
    %s = rcgmp.addi %x, %one : !rcgmp.mpz
    rcgmp.release %x : !rcgmp.mpz
    scf.yield %s : !rcgmp.mpz
  } else {
    rcgmp.release %x : !rcgmp.mpz
    scf.yield %x : !rcgmp.mpz
  }
  return %r : !rcgmp.mpz
}

// The comparison reads x after the addition, which must not reuse it.
// CHECK-LABEL: func.func @increment_compare(
// CHECK: rcgmp.retain
func.func @increment_compare(%x: !rcgmp.mpz) -> i32 {
  %one = rcgmp.constant 1 : !rcgmp.mpz
  rcgmp.retain %x : !rcgmp.mpz
  %s = rcgmp.addi %x, %one : !rcgmp.mpz
  %less = rcgmp.cmpi slt, %x, %s : !rcgmp.mpz
  rcgmp.release %s : !rcgmp.mpz
  rcgmp.release %x : !rcgmp.mpz
  %r = arith.extui %less : i1 to i32
  return %r : i32
}
