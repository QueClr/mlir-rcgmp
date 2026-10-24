// RUN: %rcgmp-opt %s | %rcgmp-opt | %FileCheck %s

// CHECK-LABEL: func.func @bits
// CHECK: rcgmp.testbiti %{{.*}}, %{{.*}} : !rcgmp.mpz, i64
// CHECK: rcgmp.testbiti %{{.*}}, %{{.*}} : <atomic>, i8
// CHECK: rcgmp.testbiti %{{.*}}, %{{.*}} : !rcgmp.mpz, i128
func.func @bits(%x: !rcgmp.mpz, %y: !rcgmp.mpz<atomic>, %n: i64, %m: i8,
                %w: i128) -> (i1, i1, i1) {
  %a = rcgmp.testbiti %x, %n : !rcgmp.mpz, i64
  %b = rcgmp.testbiti %y, %m : !rcgmp.mpz<atomic>, i8
  %c = rcgmp.testbiti %x, %w : !rcgmp.mpz, i128
  rcgmp.release %x : !rcgmp.mpz
  rcgmp.release %y : !rcgmp.mpz<atomic>
  return %a, %b, %c : i1, i1, i1
}
