// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm='dynamic' -reconcile-unrealized-casts | %FileCheck %s
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi64>>, #dlti.dl_entry<index, 16 : i64>>} {
  // CHECK-DAG: llvm.mlir.global private unnamed_addr constant @__rcgmp_mpz_20000000(dense<[0, 1, 1, 536870912]> : tensor<4xi32>) {addr_space = 0 : i32, alignment = 4 : i64} : !llvm.array<4 x i32>
  // CHECK-DAG: llvm.mlir.global private unnamed_addr constant @__rcgmp_mpz_neg_20000001(dense<[0, -1, 1, 536870913]> : tensor<4xi32>) {addr_space = 0 : i32, alignment = 4 : i64} : !llvm.array<4 x i32>
  // CHECK-DAG: llvm.mlir.global private unnamed_addr constant @__rcgmp_mpz_80000000(dense<[0, 1, 1, -2147483648]> : tensor<4xi32>) {addr_space = 0 : i32, alignment = 4 : i64} : !llvm.array<4 x i32>
  // CHECK-DAG: llvm.mlir.global private unnamed_addr constant @__rcgmp_mpz_neg_100000000(dense<[0, -2, 2, 0, 1]> : tensor<5xi32>) {addr_space = 0 : i32, alignment = 4 : i64} : !llvm.array<5 x i32>
  // CHECK-DAG: llvm.mlir.global private unnamed_addr constant @__rcgmp_mpz_300000002(dense<[0, 2, 2, 2, 3]> : tensor<5xi32>) {addr_space = 0 : i32, alignment = 4 : i64} : !llvm.array<5 x i32>
  // CHECK-DAG: llvm.mlir.global private unnamed_addr constant @__rcgmp_mpz_2000000000000000(dense<[0, 2, 2, 0, 536870912]> : tensor<5xi32>) {addr_space = 0 : i32, alignment = 4 : i64} : !llvm.array<5 x i32>
  // CHECK-DAG: llvm.func @rcgmp_mpz_shl(!llvm.ptr, i32, i32) -> !llvm.ptr
  // CHECK-DAG: llvm.func @rcgmp_mpz_shr(!llvm.ptr, i32, i32) -> !llvm.ptr
  // CHECK-LABEL: llvm.func @constants
  // CHECK: llvm.mlir.constant(2147483645 : i32)
  // CHECK: llvm.mlir.addressof @__rcgmp_mpz_20000000
  // CHECK: llvm.mlir.constant(-2147483647 : i32)
  // CHECK: llvm.mlir.addressof @__rcgmp_mpz_neg_20000001
  func.func @constants() -> (!rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz) {
    %max = rcgmp.constant 536870911 : !rcgmp.mpz
    %above = rcgmp.constant 536870912 : !rcgmp.mpz
    %min = rcgmp.constant -536870912 : !rcgmp.mpz
    %below = rcgmp.constant -536870913 : !rcgmp.mpz
    return %max, %above, %min, %below : !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz
  }
  // Limbs follow the handle width, so 2^61 needs two of them.
  // CHECK-LABEL: llvm.func @limbs
  // CHECK: llvm.mlir.addressof @__rcgmp_mpz_80000000
  // CHECK: llvm.mlir.addressof @__rcgmp_mpz_neg_100000000
  // CHECK: llvm.mlir.addressof @__rcgmp_mpz_300000002
  // CHECK: llvm.mlir.addressof @__rcgmp_mpz_2000000000000000
  func.func @limbs() -> (!rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz) {
    %a = rcgmp.constant 2147483648 : !rcgmp.mpz
    %b = rcgmp.constant -4294967296 : !rcgmp.mpz
    %c = rcgmp.constant 0x300000002 : !rcgmp.mpz
    %d = rcgmp.constant 2305843009213693952 : !rcgmp.mpz
    return %a, %b, %c, %d : !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz
  }
  // Narrow attributes are sign-extended to the handle width.
  // CHECK-LABEL: llvm.func @negatives
  // CHECK: %[[A:.*]] = llvm.mlir.constant(-3 : i32)
  // CHECK: llvm.inttoptr %[[A]] : i32 to !llvm.ptr
  // CHECK: %[[B:.*]] = llvm.mlir.constant(-19 : i32)
  // CHECK: llvm.inttoptr %[[B]] : i32 to !llvm.ptr
  func.func @negatives() -> (!rcgmp.mpz, !rcgmp.mpz) {
    %a = rcgmp.constant -1 : !rcgmp.mpz
    %b = rcgmp.constant -5 : !rcgmp.mpz
    return %a, %b : !rcgmp.mpz, !rcgmp.mpz
  }
  // CHECK-LABEL: llvm.func @make
  // CHECK: llvm.alloca {{.*}} x i32
  // CHECK: llvm.mlir.constant(536870911 : i64)
  // CHECK: llvm.mlir.constant(-536870912 : i64)
  // CHECK: llvm.trunc {{.*}} : i64 to i32
  // CHECK: llvm.inttoptr {{.*}} : i32 to !llvm.ptr
  func.func @make(%x: i64) -> !rcgmp.mpz {
    %r = rcgmp.extsi %x : i64 to !rcgmp.mpz
    return %r : !rcgmp.mpz
  }
  // CHECK-LABEL: llvm.func @retain
  // CHECK: llvm.ptrtoint {{.*}} : !llvm.ptr to i32
  // CHECK: llvm.load {{.*}} atomic monotonic {{.*}} : !llvm.ptr -> i32
  // CHECK: llvm.atomicrmw add {{.*}} monotonic : !llvm.ptr, i32
  func.func @retain(%x: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
    rcgmp.retain %x : !rcgmp.mpz<atomic>
    return %x : !rcgmp.mpz<atomic>
  }
  // A unique value is freed without updating its count; otherwise an immortal
  // one is skipped and a counted one decremented.
  // CHECK-LABEL: llvm.func @release
  // CHECK: %[[COUNT:.*]] = llvm.load %arg0 : !llvm.ptr -> i32
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i32)
  // CHECK: llvm.icmp "eq" %[[COUNT]], %[[ONE]]
  // CHECK: llvm.cond_br
  // CHECK: llvm.call @rcgmp_mpz_free
  // CHECK: %[[AGAIN:.*]] = llvm.load %arg0 : !llvm.ptr -> i32
  // CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
  // CHECK: llvm.icmp "eq" %[[AGAIN]], %[[ZERO]]
  // CHECK: llvm.sub {{.*}} : i32
  func.func @release(%x: !rcgmp.mpz) {
    rcgmp.release %x : !rcgmp.mpz
    return
  }
  // CHECK-LABEL: llvm.func @add
  // CHECK: "llvm.intr.sadd.with.overflow"{{.*}} : (i32, i32) -> !llvm.struct<(i32, i1)>
  // CHECK: llvm.inttoptr {{.*}} : i32 to !llvm.ptr
  // CHECK: llvm.load {{.*}} atomic acquire {{.*}} : !llvm.ptr -> i32
  func.func @add(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
    %r = rcgmp.addi %x, %y : !rcgmp.mpz<atomic>
    return %r : !rcgmp.mpz<atomic>
  }
  // CHECK-LABEL: llvm.func @sub
  // CHECK: "llvm.intr.ssub.with.overflow"{{.*}} : (i32, i32) -> !llvm.struct<(i32, i1)>
  // CHECK: llvm.inttoptr {{.*}} : i32 to !llvm.ptr
  // CHECK: llvm.load {{.*}} atomic acquire {{.*}} : !llvm.ptr -> i32
  func.func @sub(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
    %r = rcgmp.subi %x, %y : !rcgmp.mpz<atomic>
    return %r : !rcgmp.mpz<atomic>
  }
  // CHECK-LABEL: llvm.func @mul
  // CHECK: "llvm.intr.smul.with.overflow"{{.*}} : (i32, i32) -> !llvm.struct<(i32, i1)>
  // CHECK: llvm.inttoptr {{.*}} : i32 to !llvm.ptr
  // CHECK: llvm.load {{.*}} atomic acquire {{.*}} : !llvm.ptr -> i32
  func.func @mul(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
    %r = rcgmp.muli %x, %y : !rcgmp.mpz<atomic>
    return %r : !rcgmp.mpz<atomic>
  }
  // The tagged minimum follows the handle width.
  // CHECK-LABEL: llvm.func @neg
  // CHECK: llvm.mlir.constant(-2147483647 : i32)
  // CHECK: llvm.icmp "ne"
  func.func @neg(%x: !rcgmp.mpz) -> !rcgmp.mpz {
    %r = rcgmp.negi %x : !rcgmp.mpz
    return %r : !rcgmp.mpz
  }
  // Amounts saturate or extend to the handle width, which the runtime receives,
  // and the fast path clamps them below it.
  // CHECK-LABEL: llvm.func @shl
  // CHECK: %[[MAX:.*]] = llvm.mlir.constant(4294967295 : i64)
  // CHECK: %[[SATURATED:.*]] = llvm.intr.umin(%arg1, %[[MAX]]) : (i64, i64) -> i64
  // CHECK: %[[AMOUNT:.*]] = llvm.trunc %[[SATURATED]] : i64 to i32
  // CHECK: llvm.mlir.constant(31 : i32)
  // CHECK: llvm.shl {{.*}} : i32
  // CHECK: llvm.call @rcgmp_mpz_shl(%arg0, %[[AMOUNT]], %{{.*}}) : (!llvm.ptr, i32, i32)
  func.func @shl(%x: !rcgmp.mpz, %n: i64) -> !rcgmp.mpz {
    %r = rcgmp.shli %x, %n : !rcgmp.mpz, i64
    return %r : !rcgmp.mpz
  }
  // CHECK-LABEL: llvm.func @shr
  // CHECK: %[[AMOUNT:.*]] = llvm.zext %arg1 : i8 to i32
  // CHECK: llvm.mlir.constant(31 : i32)
  // CHECK: llvm.ashr {{.*}} : i32
  // CHECK: llvm.call @rcgmp_mpz_shr(%arg0, %[[AMOUNT]], %{{.*}}) : (!llvm.ptr, i32, i32)
  func.func @shr(%x: !rcgmp.mpz, %n: i8) -> !rcgmp.mpz {
    %r = rcgmp.shrsi %x, %n : !rcgmp.mpz, i8
    return %r : !rcgmp.mpz
  }
  // The quotient's range check follows the handle width.
  // CHECK-LABEL: llvm.func @div
  // CHECK: llvm.sdiv {{.*}} : i32
  // CHECK: llvm.mlir.constant(536870911 : i32)
  // CHECK: llvm.icmp "sle"
  // CHECK: llvm.call @rcgmp_mpz_tdiv_q({{.*}}) : (!llvm.ptr, !llvm.ptr, i32)
  func.func @div(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
    %r = rcgmp.divsi %x, %y : !rcgmp.mpz
    return %r : !rcgmp.mpz
  }
  // Immediates have 30 signed bits, so only an nsw result of at most 30 bits
  // skips the heap.
  // CHECK-LABEL: llvm.func @narrow
  // CHECK: llvm.ashr {{.*}} : i32
  // CHECK: llvm.trunc {{.*}} : i32 to i16
  // CHECK-NEXT: llvm.return
  func.func @narrow(%x: !rcgmp.mpz) -> i16 {
    %v = rcgmp.trunci %x overflow<nsw> : !rcgmp.mpz to i16
    return %v : i16
  }
  // CHECK-LABEL: llvm.func @signed_word
  // CHECK: llvm.cond_br
  // CHECK: llvm.load {{.*}} : !llvm.ptr -> i32
  func.func @signed_word(%x: !rcgmp.mpz) -> i32 {
    %v = rcgmp.trunci %x overflow<nsw> : !rcgmp.mpz to i32
    return %v : i32
  }
  // Limbs and tagged offsets follow the handle width: 2^32 + 1 has two 32-bit
  // limbs, and the remainder of at_least 2 subtracts 8 from the tagged bits.
  // CHECK-LABEL: llvm.func @match
  // CHECK: llvm.mlir.constant(8 : i32)
  // CHECK: llvm.getelementptr inbounds %{{.*}}[1] : (!llvm.ptr) -> !llvm.ptr, i32
  // CHECK-NEXT: llvm.load %{{.*}} {alignment = 4 : i64} : !llvm.ptr -> i32
  // CHECK: llvm.getelementptr inbounds %{{.*}}[%{{.*}}] : (!llvm.ptr, i16) -> !llvm.ptr, i32
  func.func @match(%x: !rcgmp.mpz) -> !rcgmp.mpz {
    %r = rcgmp.match %x : !rcgmp.mpz -> !rcgmp.mpz {
      case 4294967297 {
        %c = rcgmp.constant 0 : !rcgmp.mpz
        rcgmp.yield %c : !rcgmp.mpz
      }
      at_least 2 (%k) {
        rcgmp.yield %k : !rcgmp.mpz
      }
      default (%m) {
        rcgmp.yield %m : !rcgmp.mpz
      }
    }
    return %r : !rcgmp.mpz
  }
}
