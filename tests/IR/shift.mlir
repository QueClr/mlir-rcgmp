// RUN: %rcgmp-opt %s | %rcgmp-opt | %FileCheck %s

// CHECK-LABEL: func.func @shifts
// CHECK-SAME: (%[[X:.*]]: !rcgmp.mpz, %[[N:.*]]: i8, %[[Y:.*]]: !rcgmp.mpz<atomic>, %[[M:.*]]: i128)
// CHECK: %[[L:.*]] = rcgmp.shli %[[X]], %[[N]] : !rcgmp.mpz, i8
// CHECK: rcgmp.shrsi %[[L]], %[[N]] : !rcgmp.mpz, i8
// CHECK: rcgmp.shrsi %[[Y]], %[[M]] : <atomic>, i128
func.func @shifts(%x: !rcgmp.mpz, %n: i8, %y: !rcgmp.mpz<atomic>, %m: i128) -> (!rcgmp.mpz, !rcgmp.mpz<atomic>) {
  %l = rcgmp.shli %x, %n : !rcgmp.mpz, i8
  %r = rcgmp.shrsi %l, %n : !rcgmp.mpz, i8
  %a = rcgmp.shrsi %y, %m : !rcgmp.mpz<atomic>, i128
  return %r, %a : !rcgmp.mpz, !rcgmp.mpz<atomic>
}
