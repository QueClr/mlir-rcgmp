// RUN: %rcgmp-opt %s | %rcgmp-opt | %FileCheck %s
// RUN: %rcgmp-opt %s -mlir-print-op-generic | %FileCheck %s --check-prefix=GENERIC

// CHECK-LABEL: func.func @arms
// CHECK: rcgmp.match %{{.*}} : !rcgmp.mpz -> i64 {
// CHECK-NEXT: case 0, -1 {
// CHECK: at_least 340282366920938463463374607431768211456 (%[[K:.*]]) {
// CHECK-NEXT: rcgmp.release %[[K]]
// CHECK: default (%[[M:.*]]) {
// CHECK-NEXT: rcgmp.release %[[M]]
// GENERIC: arms = [#rcgmp.arm<case, [false, true]>, #rcgmp.arm<at_least, [340282366920938463463374607431768211456 : i130]>, #rcgmp.arm<default, []>]
func.func @arms(%n: !rcgmp.mpz, %a: i64, %b: i64) -> i64 {
  %r = rcgmp.match %n : !rcgmp.mpz -> i64 {
    case 0, -1 {
      rcgmp.yield %a : i64
    }
    at_least 340282366920938463463374607431768211456 (%k) {
      rcgmp.release %k : !rcgmp.mpz
      rcgmp.yield %b : i64
    }
    default (%m) {
      rcgmp.release %m : !rcgmp.mpz
      rcgmp.yield %a : i64
    }
  }
  return %r : i64
}

// Arms without results need no yield.
// CHECK-LABEL: func.func @no_results
// CHECK: rcgmp.match %{{.*}} : !rcgmp.mpz<atomic> {
// CHECK-NEXT: case 3 {
// CHECK-NEXT: }
// CHECK-NEXT: at_least 4 {
// CHECK-NEXT: }
// CHECK-NEXT: }
func.func @no_results(%n: !rcgmp.mpz<atomic>) {
  rcgmp.match %n : !rcgmp.mpz<atomic> {
    case 3 {
    }
    at_least 4 {
    }
  }
  return
}
