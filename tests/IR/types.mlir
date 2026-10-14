// RUN: %rcgmp-opt %s | %rcgmp-opt | %FileCheck %s

// CHECK-LABEL: func.func @nonatomic(
// CHECK-SAME: %[[VALUE:.*]]: !rcgmp.mpz) -> !rcgmp.mpz {
// CHECK: return %[[VALUE]] : !rcgmp.mpz
func.func @nonatomic(%value: !rcgmp.mpz) -> !rcgmp.mpz {
  return %value : !rcgmp.mpz
}

// CHECK-LABEL: func.func @atomic(
// CHECK-SAME: %[[VALUE:.*]]: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
// CHECK: return %[[VALUE]] : !rcgmp.mpz<atomic>
func.func @atomic(%value: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  return %value : !rcgmp.mpz<atomic>
}
