// RUN: %rcgmp-opt %s -convert-to-llvm | %FileCheck %s
// RUN: %rcgmp-opt %s -convert-to-llvm='dynamic' | %FileCheck %s

// CHECK-LABEL: llvm.func @identity(
// CHECK-SAME: %[[VALUE:.*]]: !llvm.ptr) -> !llvm.ptr {
// CHECK-NEXT: llvm.return %[[VALUE]] : !llvm.ptr
func.func @identity(%value: !rcgmp.mpz) -> !rcgmp.mpz {
  return %value : !rcgmp.mpz
}

// CHECK-LABEL: llvm.func @atomic_identity(
// CHECK-SAME: %[[VALUE:.*]]: !llvm.ptr) -> !llvm.ptr {
// CHECK-NEXT: llvm.return %[[VALUE]] : !llvm.ptr
func.func @atomic_identity(%value: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  return %value : !rcgmp.mpz<atomic>
}

// CHECK-LABEL: llvm.func @external(!llvm.ptr) -> !llvm.ptr
func.func private @external(!rcgmp.mpz) -> !rcgmp.mpz

// CHECK-LABEL: llvm.func @call(
// CHECK-SAME: %[[VALUE:.*]]: !llvm.ptr) -> !llvm.ptr {
// CHECK-NEXT: %[[RESULT:.*]] = llvm.call @external(%[[VALUE]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK-NEXT: llvm.return %[[RESULT]] : !llvm.ptr
func.func @call(%value: !rcgmp.mpz) -> !rcgmp.mpz {
  %result = func.call @external(%value) : (!rcgmp.mpz) -> !rcgmp.mpz
  return %result : !rcgmp.mpz
}

// CHECK-LABEL: llvm.func @branch(
// CHECK-SAME: %[[COND:.*]]: i1, %[[LEFT:.*]]: !llvm.ptr, %[[RIGHT:.*]]: !llvm.ptr) -> !llvm.ptr {
// CHECK-NEXT: llvm.cond_br %[[COND]], ^bb1(%[[LEFT]] : !llvm.ptr), ^bb1(%[[RIGHT]] : !llvm.ptr)
// CHECK: ^bb1(%[[RESULT:.*]]: !llvm.ptr):
// CHECK-NEXT: llvm.return %[[RESULT]] : !llvm.ptr
func.func @branch(%condition: i1, %left: !rcgmp.mpz, %right: !rcgmp.mpz) -> !rcgmp.mpz {
  cf.cond_br %condition, ^merge(%left : !rcgmp.mpz), ^merge(%right : !rcgmp.mpz)
^merge(%value: !rcgmp.mpz):
  return %value : !rcgmp.mpz
}
