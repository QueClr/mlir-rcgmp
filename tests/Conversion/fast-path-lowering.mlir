// RUN: %rcgmp-test-opt %s -allow-unregistered-dialect -test-rcgmp-fast-path-lowering | %FileCheck %s
// RUN: %rcgmp-test-opt %s -allow-unregistered-dialect -test-rcgmp-fast-path-lowering -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts | %FileCheck %s --check-prefix=LLVM

// Only consuming operations with an mpz result pass a reuse mask.
// CHECK-DAG: func.func private @test_negate(!rcgmp.mpz) -> !rcgmp.mpz
// CHECK-DAG: func.func private @test_select(!rcgmp.mpz, i1, !rcgmp.mpz, i32) -> !rcgmp.mpz
// CHECK-DAG: func.func private @test_equal_word_atomic(!rcgmp.mpz<atomic>, i64) -> i32
// CHECK-DAG: func.func private @test_truncate(!rcgmp.mpz) -> i32

// A borrowing operation with an mpz result checks no uniqueness.
// CHECK-LABEL: func.func @negate
// CHECK-SAME: (%[[X:.*]]: !rcgmp.mpz)
// CHECK: %[[BITS:.*]] = rcgmp.bridge.mpz.to_bits %[[X]]
// CHECK: %[[DIFF:.*]], %{{.*}} = rcgmp.bridge.subsi_extended %{{.*}}, %[[BITS]]
// CHECK: scf.if %{{.*}} -> (!rcgmp.mpz) {
// CHECK-NEXT: %[[FAST:.*]] = rcgmp.bridge.mpz.from_bits %[[DIFF]]
// CHECK-NEXT: scf.yield %[[FAST]]
// CHECK-NEXT: } else {
// CHECK-NEXT: %[[CALL:.*]] = func.call @test_negate(%[[X]]) : (!rcgmp.mpz) -> !rcgmp.mpz
// CHECK-NEXT: scf.yield %[[CALL]]
func.func @negate(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = "test.negate"(%x) : (!rcgmp.mpz) -> !rcgmp.mpz
  return %r : !rcgmp.mpz
}

// Bit i of the reuse mask follows the i-th mpz operand, skipping other operands.
// CHECK-LABEL: func.func @select
// CHECK-SAME: (%[[X:.*]]: !rcgmp.mpz, %[[C:.*]]: i1, %[[Y:.*]]: !rcgmp.mpz)
// CHECK: %[[XBITS:.*]] = rcgmp.bridge.mpz.to_bits %[[X]]
// CHECK: %[[YBITS:.*]] = rcgmp.bridge.mpz.to_bits %[[Y]]
// CHECK: %[[FAST:.*]] = arith.select %[[C]], %[[XBITS]], %[[YBITS]]
// CHECK: scf.if %{{.*}} -> (!rcgmp.mpz) {
// CHECK: rcgmp.bridge.mpz.from_bits %[[FAST]]
// CHECK: } else {
// CHECK: %[[XUNIQUE:.*]] = scf.if
// CHECK: rcgmp.bridge.mpz.is_unique %[[X]]
// CHECK: %[[YUNIQUE:.*]] = scf.if
// CHECK: rcgmp.bridge.mpz.is_unique %[[Y]]
// CHECK: %[[XFLAG:.*]] = arith.extui %[[XUNIQUE]] : i1 to i32
// CHECK: %[[YFLAG:.*]] = arith.extui %[[YUNIQUE]] : i1 to i32
// CHECK: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK: %[[SHIFTED:.*]] = arith.shli %[[YFLAG]], %[[ONE]] overflow<nsw, nuw>
// CHECK: %[[MASK:.*]] = arith.ori %[[XFLAG]], %[[SHIFTED]]
// CHECK: func.call @test_select(%[[X]], %[[C]], %[[Y]], %[[MASK]])
func.func @select(%x: !rcgmp.mpz, %c: i1, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = "test.select"(%x, %c, %y) : (!rcgmp.mpz, i1, !rcgmp.mpz) -> !rcgmp.mpz
  return %r : !rcgmp.mpz
}

// A borrowing operation checks no uniqueness, passes non-mpz operands through,
// and converts the runtime result.
// CHECK-LABEL: func.func @equal_word
// CHECK-SAME: (%[[X:.*]]: !rcgmp.mpz<atomic>, %[[W:.*]]: i64)
// CHECK: %[[BITS:.*]] = rcgmp.bridge.mpz.to_bits %[[X]]
// CHECK: %[[PAYLOAD:.*]] = arith.shrsi %[[BITS]]
// CHECK: %[[EQUAL:.*]] = arith.cmpi eq, %[[PAYLOAD]], %[[W]]
// CHECK: scf.if %{{.*}} -> (i1) {
// CHECK-NEXT: scf.yield %[[EQUAL]] : i1
// CHECK-NEXT: } else {
// CHECK-NOT: rcgmp.bridge.mpz.is_unique
// CHECK: %[[CALL:.*]] = func.call @test_equal_word_atomic(%[[X]], %[[W]])
// CHECK: %[[SLOW:.*]] = arith.cmpi ne, %[[CALL]]
// CHECK: scf.yield %[[SLOW]] : i1
// LLVM-LABEL: llvm.func @equal_word
// LLVM-NOT: llvm.load
// LLVM: llvm.call @test_equal_word_atomic
func.func @equal_word(%x: !rcgmp.mpz<atomic>, %w: i64) -> i1 {
  %r = "test.equal_word"(%x, %w) : (!rcgmp.mpz<atomic>, i64) -> i1
  return %r : i1
}

// A consuming operation with a fixed-width result checks no uniqueness.
// CHECK-LABEL: func.func @truncate
// CHECK-SAME: (%[[X:.*]]: !rcgmp.mpz)
// CHECK: %[[FAST:.*]] = arith.trunci %{{.*}} : i64 to i32
// CHECK: scf.if %{{.*}} -> (i32) {
// CHECK-NEXT: scf.yield %[[FAST]] : i32
// CHECK-NEXT: } else {
// CHECK-NEXT: %[[CALL:.*]] = func.call @test_truncate(%[[X]]) : (!rcgmp.mpz) -> i32
// CHECK-NEXT: scf.yield %[[CALL]] : i32
// LLVM-LABEL: llvm.func @truncate
// LLVM: llvm.call @test_truncate(%{{.*}}) : (!llvm.ptr) -> i32
func.func @truncate(%x: !rcgmp.mpz) -> i32 {
  %r = "test.truncate"(%x) : (!rcgmp.mpz) -> i32
  return %r : i32
}
