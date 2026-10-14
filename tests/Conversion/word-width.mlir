// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm='dynamic' -reconcile-unrealized-casts | %FileCheck %s
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi64>>, #dlti.dl_entry<index, 16 : i64>>} {
  // CHECK-LABEL: llvm.func @make
  // CHECK: llvm.alloca {{.*}} x i32
  // CHECK: llvm.mlir.constant(536870911 : i64)
  // CHECK: llvm.mlir.constant(-536870912 : i64)
  // CHECK: llvm.trunc {{.*}} : i64 to i32
  // CHECK: llvm.inttoptr {{.*}} : i32 to !llvm.ptr
  func.func @make(%x: i64) -> !rcgmp.mpz {
    %r = rcgmp.from_int %x : i64 to !rcgmp.mpz
    return %r : !rcgmp.mpz
  }
  // CHECK-LABEL: llvm.func @retain
  // CHECK: llvm.ptrtoint {{.*}} : !llvm.ptr to i32
  // CHECK: llvm.atomicrmw add {{.*}} monotonic : !llvm.ptr, i32
  func.func @retain(%x: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
    %r = rcgmp.retain %x : !rcgmp.mpz<atomic>
    return %r : !rcgmp.mpz<atomic>
  }
  // CHECK-LABEL: llvm.func @add
  // CHECK: "llvm.intr.sadd.with.overflow"{{.*}} : (i32, i32) -> !llvm.struct<(i32, i1)>
  // CHECK: llvm.inttoptr {{.*}} : i32 to !llvm.ptr
  // CHECK: llvm.load {{.*}} atomic acquire {{.*}} : !llvm.ptr -> i32
  func.func @add(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
    %r = rcgmp.addi %x, %y : !rcgmp.mpz<atomic>
    return %r : !rcgmp.mpz<atomic>
  }
}
