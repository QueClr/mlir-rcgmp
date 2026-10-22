// RUN: %rcgmp-opt %s | %rcgmp-opt | %FileCheck %s

// CHECK-LABEL: func.func @unary
// CHECK-SAME: (%[[X:.*]]: !rcgmp.mpz, %[[Y:.*]]: !rcgmp.mpz<atomic>)
// CHECK: %[[NOT:.*]] = rcgmp.noti %[[X]] : !rcgmp.mpz
// CHECK: %[[NEG:.*]] = rcgmp.negi %[[NOT]] : !rcgmp.mpz
// CHECK: rcgmp.absi %[[NEG]] : !rcgmp.mpz
// CHECK: rcgmp.negi %[[Y]] : <atomic>
func.func @unary(%x: !rcgmp.mpz, %y: !rcgmp.mpz<atomic>) -> (!rcgmp.mpz, !rcgmp.mpz<atomic>) {
  %not = rcgmp.noti %x : !rcgmp.mpz
  %neg = rcgmp.negi %not : !rcgmp.mpz
  %abs = rcgmp.absi %neg : !rcgmp.mpz
  %atomic = rcgmp.negi %y : !rcgmp.mpz<atomic>
  return %abs, %atomic : !rcgmp.mpz, !rcgmp.mpz<atomic>
}
