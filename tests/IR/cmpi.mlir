// RUN: %rcgmp-opt %s | %rcgmp-opt | %FileCheck %s
// RUN: %rcgmp-opt %s -mlir-print-op-generic | %FileCheck %s --check-prefix=GENERIC

// CHECK-LABEL: func.func @predicates
// CHECK-SAME: (%[[X:.*]]: !rcgmp.mpz, %[[Y:.*]]: !rcgmp.mpz)
// CHECK: rcgmp.cmpi eq, %[[X]], %[[Y]] : !rcgmp.mpz
// CHECK: rcgmp.cmpi ne, %[[X]], %[[Y]] : !rcgmp.mpz
// CHECK: rcgmp.cmpi slt, %[[X]], %[[Y]] : !rcgmp.mpz
// CHECK: rcgmp.cmpi sle, %[[X]], %[[Y]] : !rcgmp.mpz
// CHECK: rcgmp.cmpi sgt, %[[X]], %[[Y]] : !rcgmp.mpz
// CHECK: rcgmp.cmpi sge, %[[X]], %[[X]] : !rcgmp.mpz
// GENERIC: <{predicate = 0 : i64}>
// GENERIC: <{predicate = 1 : i64}>
// GENERIC: <{predicate = 2 : i64}>
// GENERIC: <{predicate = 3 : i64}>
// GENERIC: <{predicate = 4 : i64}>
// GENERIC: <{predicate = 5 : i64}>
func.func @predicates(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> (i1, i1, i1, i1, i1, i1) {
  %eq = rcgmp.cmpi eq, %x, %y : !rcgmp.mpz
  %ne = rcgmp.cmpi ne, %x, %y : !rcgmp.mpz
  %slt = rcgmp.cmpi slt, %x, %y : !rcgmp.mpz
  %sle = rcgmp.cmpi sle, %x, %y : !rcgmp.mpz
  %sgt = rcgmp.cmpi sgt, %x, %y : !rcgmp.mpz
  %sge = rcgmp.cmpi sge, %x, %x : !rcgmp.mpz
  return %eq, %ne, %slt, %sle, %sgt, %sge : i1, i1, i1, i1, i1, i1
}

// CHECK-LABEL: func.func @atomic
// CHECK: rcgmp.cmpi slt, %{{.*}}, %{{.*}} : <atomic>
func.func @atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> i1 {
  %r = rcgmp.cmpi slt, %x, %y : !rcgmp.mpz<atomic>
  return %r : i1
}
