// RUN: %rcgmp-opt %s -convert-to-llvm -reconcile-unrealized-casts | %FileCheck %s
// CHECK-LABEL: llvm.func @checked_add_i8
// CHECK: %[[PAIR:.*]] = "llvm.intr.sadd.with.overflow"{{.*}} : (i8, i8) -> !llvm.struct<(i8, i1)>
// CHECK: llvm.extractvalue %[[PAIR]][0]
// CHECK: llvm.extractvalue %[[PAIR]][1]
// Pack the wrapped sum in bits 0..7 and the overflow flag in bit 8.
func.func @checked_add_i8(%lhs: i32, %rhs: i32) -> i32 {
  %l = arith.trunci %lhs : i32 to i8
  %r = arith.trunci %rhs : i32 to i8
  %sum, %overflow = rcgmp.bridge.addsi_extended %l, %r : i8, i1
  %s = arith.extui %sum : i8 to i32
  %o = arith.extui %overflow : i1 to i32
  %eight = arith.constant 8 : i32
  %flag = arith.shli %o, %eight : i32
  %packed = arith.ori %s, %flag : i32
  return %packed : i32
}
