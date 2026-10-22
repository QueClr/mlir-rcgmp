// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf | %FileCheck %s
// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts | %FileCheck %s --check-prefix=LLVM

// Tables are module globals named by a digest of their contents.
// CHECK-DAG: memref.global "private" constant @[[TABLE:__rcgmp_table_[0-9a-f]+]] : memref<6x1xi8> = dense<{{\[}}[30], [10], [11], [10], [30], [20]]>
// CHECK-DAG: memref.global "private" constant @[[SLOTS:__rcgmp_table_[0-9a-f]+]] : memref<{{[0-9]+}}x2xi64>

func.func private @use(!rcgmp.mpz) -> i64

// Each path yields the dispatch index, the region argument, and the results of
// arms that only yield values known on selection. Immediates are selected by
// payload and keep their tag after subtracting four times the threshold.
// Negative heap integers have no arm, so they take the arm of their neighbors.
// CHECK-LABEL: func.func @successor
// CHECK-SAME: (%[[N:.*]]: !rcgmp.mpz, %[[A:.*]]: i64)
// CHECK: %[[BITS:.*]] = rcgmp.bridge.mpz.to_bits %[[N]] : !rcgmp.mpz to i64
// CHECK: %[[SELECTED:.*]]:3 = scf.if %{{.*}} -> (index, !rcgmp.mpz, i64) {
// CHECK: %[[PAYLOAD:.*]] = arith.shrsi %[[BITS]]
// CHECK: arith.cmpi slt, %[[PAYLOAD]]
// CHECK: scf.yield %{{.*}}, %{{.*}}, %[[A]] : index, i64, i64
// CHECK: ub.poison : i64
// CHECK: %[[TAGGED:.*]] = arith.subi %[[BITS]], %{{.*}}#1 : i64
// CHECK: rcgmp.bridge.mpz.from_bits %[[TAGGED]]
// CHECK: } else {
// CHECK: %[[SIZE:.*]] = rcgmp.bridge.mpz.size %[[N]]
// CHECK: arith.cmpi sgt, %[[SIZE]]
// CHECK: func.call @rcgmp_mpz_sub(%[[N]]
// CHECK: rcgmp.bridge.mpz.dec_ref %[[N]]
// CHECK: scf.index_switch %[[SELECTED]]#0 -> i64
// CHECK-NEXT: case 0 {
// CHECK-NEXT: func.call @use(%[[SELECTED]]#1)
// CHECK: default {
// CHECK-NEXT: scf.yield %[[SELECTED]]#2 : i64
func.func @successor(%n: !rcgmp.mpz, %a: i64) -> i64 {
  %r = rcgmp.match %n : !rcgmp.mpz -> i64 {
    case 0 {
      rcgmp.yield %a : i64
    }
    at_least 1 (%k) {
      %u = func.call @use(%k) : (!rcgmp.mpz) -> i64
      rcgmp.yield %u : i64
    }
  }
  return %r : i64
}

// Close boundaries clamp the payload into a table. Arms yielding integer
// constants become rows of the narrowest integers holding them, and since no
// arm needs a region, there is no dispatch after selection.
// CHECK-LABEL: func.func @table
// CHECK: %[[PAYLOAD:.*]] = arith.shrsi
// CHECK: %[[LOW:.*]] = arith.constant -1 : i64
// CHECK: %[[HIGH:.*]] = arith.constant 4 : i64
// CHECK: %[[MIN:.*]] = arith.minsi %[[PAYLOAD]], %[[HIGH]]
// CHECK: %[[CLAMPED:.*]] = arith.maxsi %[[MIN]], %[[LOW]]
// CHECK: %[[OFFSET:.*]] = arith.subi %[[CLAMPED]], %[[LOW]]
// CHECK: %[[INDEX:.*]] = arith.index_castui %[[OFFSET]]
// CHECK: %[[BASE:.*]] = memref.get_global @[[TABLE]]
// CHECK: %[[ENTRY:.*]] = memref.load %[[BASE]][%[[INDEX]], %{{.*}}]
// CHECK: arith.extsi %[[ENTRY]] : i8 to i64
// No heap integer selects a case, so the heap path loads only the size.
// CHECK: rcgmp.bridge.mpz.size
// CHECK-NOT: rcgmp.bridge.mpz.limb
// CHECK-NOT: scf.index_switch
// CHECK: return
func.func @table(%n: !rcgmp.mpz) -> i64 {
  %r = rcgmp.match %n : !rcgmp.mpz -> i64 {
    case 0, 2 {
      %c = arith.constant 10 : i64
      rcgmp.yield %c : i64
    }
    case 1 {
      %c = arith.constant 11 : i64
      rcgmp.yield %c : i64
    }
    at_least 4 {
      %c = arith.constant 20 : i64
      rcgmp.yield %c : i64
    }
    default {
      %c = arith.constant 30 : i64
      rcgmp.yield %c : i64
    }
  }
  return %r : i64
}

// Many distant keys over one arm hash to the only slot that may hold them,
// which stores the key and the results; a mismatching key selects the other
// arm.
// CHECK-LABEL: func.func @hashed
// CHECK: %[[PAYLOAD:.*]] = arith.shrsi
// CHECK: %[[PRODUCT:.*]] = arith.muli %[[PAYLOAD]], %{{.*}} : i64
// CHECK: %[[SLOT:.*]] = arith.shrui %[[PRODUCT]], %{{.*}} : i64
// CHECK: %[[INDEX:.*]] = arith.index_castui %[[SLOT]]
// CHECK: memref.get_global @[[SLOTS]]
// CHECK: %[[KEY:.*]] = memref.load %{{.*}}[%[[INDEX]], %{{.*}}]
// CHECK: %[[VALUE:.*]] = memref.load %{{.*}}[%[[INDEX]], %{{.*}}]
// CHECK: %[[HIT:.*]] = arith.cmpi eq, %[[KEY]], %[[PAYLOAD]]
// CHECK: arith.select %[[HIT]], %[[VALUE]], %{{.*}} : i64
func.func @hashed(%n: !rcgmp.mpz) -> i64 {
  %r = rcgmp.match %n : !rcgmp.mpz -> i64 {
    case 0 {
      %c = arith.constant 1 : i64
      rcgmp.yield %c : i64
    }
    case 7 {
      %c = arith.constant 2 : i64
      rcgmp.yield %c : i64
    }
    case 100 {
      %c = arith.constant 3 : i64
      rcgmp.yield %c : i64
    }
    case 1000 {
      %c = arith.constant 4 : i64
      rcgmp.yield %c : i64
    }
    case 65536 {
      %c = arith.constant 5 : i64
      rcgmp.yield %c : i64
    }
    case 1099511627776 {
      %c = arith.constant 6 : i64
      rcgmp.yield %c : i64
    }
    default {
      %c = arith.constant 0 : i64
      rcgmp.yield %c : i64
    }
  }
  return %r : i64
}

// A few points over one arm are compared for equality. Every heap integer
// selects the default, so the heap path reads nothing.
// CHECK-LABEL: func.func @points
// CHECK: %[[PAYLOAD:.*]] = arith.shrsi
// CHECK: arith.cmpi eq, %[[PAYLOAD]], %{{.*}}
// CHECK: arith.cmpi eq, %[[PAYLOAD]], %{{.*}}
// CHECK: } else {
// CHECK-NOT: rcgmp.bridge.mpz.size
// CHECK: return
func.func @points(%n: !rcgmp.mpz, %a: i64, %b: i64) -> i64 {
  %r = rcgmp.match %n : !rcgmp.mpz -> i64 {
    case 3, 9 {
      rcgmp.yield %a : i64
    }
    default {
      rcgmp.yield %b : i64
    }
  }
  return %r : i64
}

// Heap literals are selected by limb count, then by the limbs they share,
// compared at once, then by the limb where they diverge.
// CHECK-LABEL: func.func @limbs
// CHECK: %[[SIZE:.*]] = rcgmp.bridge.mpz.size %[[N:.*]] :
// CHECK: arith.cmpi eq, %[[SIZE]], %{{.*}}
// CHECK: %[[HIGH:.*]] = rcgmp.bridge.mpz.limb %[[N]][1]
// CHECK: arith.cmpi eq, %[[HIGH]], %{{.*}}
// CHECK: %[[LOW:.*]] = rcgmp.bridge.mpz.limb %[[N]][0]
// CHECK: arith.cmpi ult, %[[LOW]], %{{.*}}
// LLVM-LABEL: llvm.func @limbs
// LLVM: llvm.getelementptr inbounds %{{.*}}[1] : (!llvm.ptr) -> !llvm.ptr, i64
// LLVM-NEXT: llvm.load %{{.*}} {alignment = 8 : i64} : !llvm.ptr -> i64
// LLVM: llvm.getelementptr inbounds %{{.*}}[4] : (!llvm.ptr) -> !llvm.ptr, i64
// LLVM: llvm.getelementptr inbounds %{{.*}}[3] : (!llvm.ptr) -> !llvm.ptr, i64
func.func @limbs(%n: !rcgmp.mpz, %a: i64, %b: i64) -> i64 {
  %r = rcgmp.match %n : !rcgmp.mpz -> i64 {
    case 18446744073709551617, 18446744073709551618 {
      rcgmp.yield %a : i64
    }
    default {
      rcgmp.yield %b : i64
    }
  }
  return %r : i64
}

// Long shared limb runs compare the whole integer in the runtime.
// CHECK-LABEL: func.func @run
// CHECK: rcgmp.bridge.mpz.size
// CHECK-NOT: rcgmp.bridge.mpz.limb
// CHECK: func.call @rcgmp_mpz_cmp
func.func @run(%n: !rcgmp.mpz, %a: i64, %b: i64) -> i64 {
  %r = rcgmp.match %n : !rcgmp.mpz -> i64 {
    case 2582249878086908589655919172003011874329705792829223512830659356540647622016841194629645353280137831435903171972747493383 {
      rcgmp.yield %a : i64
    }
    default {
      rcgmp.yield %b : i64
    }
  }
  return %r : i64
}

// Subtracting a negative threshold may leave the immediate range, so the arm
// subtracts it with an overflow check.
// CHECK-LABEL: func.func @negative_threshold
// CHECK: scf.index_switch
// CHECK: rcgmp.bridge.subsi_extended
// CHECK: func.call @rcgmp_mpz_sub
// CHECK: func.call @use
func.func @negative_threshold(%n: !rcgmp.mpz) -> i64 {
  %r = rcgmp.match %n : !rcgmp.mpz -> i64 {
    at_least -5 (%k) {
      %u = func.call @use(%k) : (!rcgmp.mpz) -> i64
      rcgmp.yield %u : i64
    }
  }
  return %r : i64
}
