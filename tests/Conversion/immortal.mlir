// RUN: %rcgmp-opt %s -split-input-file -convert-to-llvm | %FileCheck %s

// A compatible existing object is used as is.
// CHECK: llvm.mlir.global private constant @__rcgmp_mpz_2000000000000000(dense<[0, 1, 1, 2305843009213693952]> : tensor<4xi64>) {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<4 x i64>
// CHECK-NOT: llvm.mlir.global
// CHECK-LABEL: llvm.func @existing
// CHECK: llvm.mlir.addressof @__rcgmp_mpz_2000000000000000 : !llvm.ptr
llvm.mlir.global private constant @__rcgmp_mpz_2000000000000000(dense<[0, 1, 1, 2305843009213693952]> : tensor<4xi64>) {alignment = 16 : i64} : !llvm.array<4 x i64>
func.func @existing() -> !rcgmp.mpz {
  %r = rcgmp.bridge.mpz.immortal 2305843009213693952 : i63 : !rcgmp.mpz
  return %r : !rcgmp.mpz
}

// -----

// The object depends only on the signed value, not on the attribute width.
// CHECK-DAG: llvm.mlir.global private unnamed_addr constant @__rcgmp_mpz_2000000000000000(dense<[0, 1, 1, 2305843009213693952]> : tensor<4xi64>)
// CHECK-DAG: llvm.mlir.global private unnamed_addr constant @__rcgmp_mpz_neg_8000000000000000(dense<[0, -1, 1, -9223372036854775808]> : tensor<4xi64>)
// CHECK-NOT: llvm.mlir.global
// CHECK-LABEL: llvm.func @widths
// CHECK: llvm.mlir.addressof @__rcgmp_mpz_2000000000000000 : !llvm.ptr
// CHECK: llvm.mlir.addressof @__rcgmp_mpz_2000000000000000 : !llvm.ptr
// CHECK: llvm.mlir.addressof @__rcgmp_mpz_neg_8000000000000000 : !llvm.ptr
func.func @widths() -> (!rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz) {
  %wide = rcgmp.bridge.mpz.immortal 2305843009213693952 : i128 : !rcgmp.mpz
  %narrow = rcgmp.bridge.mpz.immortal 2305843009213693952 : i63 : !rcgmp.mpz
  %min = rcgmp.bridge.mpz.immortal -9223372036854775808 : i64 : !rcgmp.mpz
  return %wide, %narrow, %min : !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz
}

// -----

// An immortal object has count zero and a unique one has count one. Atomic
// counts use a relaxed load for the immortal test and an acquire load for the
// uniqueness test.
// CHECK-LABEL: llvm.func @tests
// CHECK: %[[LC:.*]] = llvm.load %arg0 : !llvm.ptr -> i64
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i64)
// CHECK: llvm.icmp "eq" %[[LC]], %[[ZERO]]
// CHECK: %[[LU:.*]] = llvm.load %arg0 : !llvm.ptr -> i64
// CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i64)
// CHECK: llvm.icmp "eq" %[[LU]], %[[ONE]]
// CHECK: %[[AC:.*]] = llvm.load %arg1 atomic monotonic {alignment = 8 : i64} : !llvm.ptr -> i64
// CHECK: %[[AZERO:.*]] = llvm.mlir.constant(0 : i64)
// CHECK: llvm.icmp "eq" %[[AC]], %[[AZERO]]
// CHECK: %[[AU:.*]] = llvm.load %arg1 atomic acquire {alignment = 8 : i64} : !llvm.ptr -> i64
// CHECK: %[[AONE:.*]] = llvm.mlir.constant(1 : i64)
// CHECK: llvm.icmp "eq" %[[AU]], %[[AONE]]
func.func @tests(%x: !rcgmp.mpz, %y: !rcgmp.mpz<atomic>) -> (i1, i1, i1, i1) {
  %a = rcgmp.bridge.mpz.is_immortal %x : !rcgmp.mpz
  %b = rcgmp.bridge.mpz.is_unique %x : !rcgmp.mpz
  %c = rcgmp.bridge.mpz.is_immortal %y : !rcgmp.mpz<atomic>
  %d = rcgmp.bridge.mpz.is_unique %y : !rcgmp.mpz<atomic>
  return %a, %b, %c, %d : i1, i1, i1, i1
}
