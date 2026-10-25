// RUN: %rcgmp-opt %s -inline | %FileCheck %s

// A callee using rcgmp operations, including rcgmp.match and its regions, is
// inlined into its caller.
// CHECK-LABEL: func.func @caller
// CHECK-NOT: call
// CHECK: rcgmp.match %{{.*}} : !rcgmp.mpz -> !rcgmp.mpz {
// CHECK: case 0 {
// CHECK: at_least 1 (%[[D:.*]]) {
// CHECK-NEXT: rcgmp.yield %[[D]] : !rcgmp.mpz
// CHECK: rcgmp.addi
// CHECK-NOT: call
func.func private @pred(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.match %x : !rcgmp.mpz -> !rcgmp.mpz {
    case 0 {
      %z = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.yield %z : !rcgmp.mpz
    }
    at_least 1 (%d) {
      rcgmp.yield %d : !rcgmp.mpz
    }
  }
  return %r : !rcgmp.mpz
}

func.func private @double(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  rcgmp.retain %x : !rcgmp.mpz
  %r = rcgmp.addi %x, %x : !rcgmp.mpz
  return %r : !rcgmp.mpz
}

func.func @caller(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %p = func.call @pred(%x) : (!rcgmp.mpz) -> !rcgmp.mpz
  %r = func.call @double(%p) : (!rcgmp.mpz) -> !rcgmp.mpz
  return %r : !rcgmp.mpz
}
