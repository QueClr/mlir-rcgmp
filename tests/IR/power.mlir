// RUN: %rcgmp-opt %s | %rcgmp-opt | %FileCheck %s

// CHECK-LABEL: func.func @powers
// CHECK-SAME: (%[[X:.*]]: !rcgmp.mpz, %[[N:.*]]: i8, %[[Y:.*]]: !rcgmp.mpz<atomic>, %[[M:.*]]: i128)
// CHECK: %[[P:.*]] = rcgmp.ipowi %[[X]], %[[N]] : !rcgmp.mpz, i8
// CHECK: rcgmp.ipowi %[[Y]], %[[M]] : <atomic>, i128
func.func @powers(%x: !rcgmp.mpz, %n: i8, %y: !rcgmp.mpz<atomic>, %m: i128) -> (!rcgmp.mpz, !rcgmp.mpz<atomic>) {
  %p = rcgmp.ipowi %x, %n : !rcgmp.mpz, i8
  %a = rcgmp.ipowi %y, %m : !rcgmp.mpz<atomic>, i128
  return %p, %a : !rcgmp.mpz, !rcgmp.mpz<atomic>
}
