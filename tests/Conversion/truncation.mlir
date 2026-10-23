// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf | %FileCheck %s

// An nsw result of at most 62 bits fits only an immediate, so the heap is not
// read.
// CHECK-LABEL: func.func @immediate_only
// CHECK: %[[BITS:.*]] = rcgmp.bridge.mpz.to_bits %arg0
// CHECK: %[[X:.*]] = arith.shrsi %[[BITS]]
// CHECK: arith.trunci %[[X]] overflow<nsw> : i64 to i32
// CHECK-NOT: scf.if
// CHECK: return
func.func @immediate_only(%x: !rcgmp.mpz) -> i32 {
  %v = rcgmp.trunci %x overflow<nsw> : !rcgmp.mpz to i32
  return %v : i32
}

// A 64-bit nsw result may come from the heap.
// CHECK-LABEL: func.func @nsw_word
// CHECK: scf.if
// CHECK: rcgmp.bridge.mpz.limb
func.func @nsw_word(%x: !rcgmp.mpz) -> i64 {
  %v = rcgmp.trunci %x overflow<nsw> : !rcgmp.mpz to i64
  return %v : i64
}

// A wide result reads a limb only below the limb count, and negates the
// magnitude of a negative integer.
// CHECK-LABEL: func.func @wide
// CHECK: } else {
// CHECK: %[[SIZE:.*]] = rcgmp.bridge.mpz.size %arg0
// CHECK: %[[NEGATIVE:.*]] = arith.cmpi slt, %[[SIZE]]
// CHECK: rcgmp.bridge.mpz.limb %arg0[%{{.*}}]
// CHECK: %[[PRESENT:.*]] = arith.cmpi ugt
// CHECK: scf.if %[[PRESENT]] -> (i64)
// CHECK: rcgmp.bridge.mpz.limb %arg0[%{{.*}}]
// CHECK: %[[MAGNITUDE:.*]] = arith.ori {{.*}} : i128
// CHECK: %[[NEGATED:.*]] = arith.subi %{{.*}}, %[[MAGNITUDE]] : i128
// CHECK: arith.select %[[NEGATIVE]], %[[NEGATED]], %[[MAGNITUDE]] : i128
func.func @wide(%x: !rcgmp.mpz) -> i128 {
  %v = rcgmp.trunci %x : !rcgmp.mpz to i128
  return %v : i128
}

// An unsigned immediate overflows when extending the truncation back changes
// it; a heap integer overflows when negative, longer than the result, or at
// least two to the power of the width.
// CHECK-LABEL: func.func @unsigned_overflow
// CHECK: %[[X:.*]] = arith.shrsi
// CHECK: %[[V:.*]] = arith.trunci %[[X]] : i64 to i32
// CHECK: %[[BACK:.*]] = arith.extui %[[V]] : i32 to i64
// CHECK: %[[OVER:.*]] = arith.cmpi ne, %[[BACK]], %[[X]]
// CHECK: scf.yield %[[V]], %[[OVER]]
// CHECK: arith.cmpi ugt
// CHECK: arith.constant 4294967296 : i64
// CHECK: arith.cmpi uge
func.func @unsigned_overflow(%x: !rcgmp.mpz) -> (i32, i1) {
  %v, %o = rcgmp.trunci_overflow %x unsigned : !rcgmp.mpz to i32
  return %v, %o : i32, i1
}
