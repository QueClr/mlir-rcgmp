// RUN: %rcgmp-opt %s -canonicalize | %FileCheck %s

// Reinterpreting a handle as bits and back, either way, is the original.
// CHECK-LABEL: func.func @round_trips
// CHECK-SAME: (%[[BITS:.*]]: i64, %[[HANDLE:.*]]: !rcgmp.mpz)
// CHECK-NOT: rcgmp.bridge
// CHECK: return %[[BITS]], %[[HANDLE]]
func.func @round_trips(%bits: i64, %handle: !rcgmp.mpz) -> (i64, !rcgmp.mpz) {
  %h = rcgmp.bridge.mpz.from_bits %bits : i64 to !rcgmp.mpz
  %b = rcgmp.bridge.mpz.to_bits %h : !rcgmp.mpz to i64
  %c = rcgmp.bridge.mpz.to_bits %handle : !rcgmp.mpz to i64
  %d = rcgmp.bridge.mpz.from_bits %c : i64 to !rcgmp.mpz
  return %b, %d : i64, !rcgmp.mpz
}

// Constructed values have a known tag.
// CHECK-LABEL: func.func @known_tags
// CHECK-DAG: %[[TRUE:.*]] = arith.constant true
// CHECK-DAG: %[[FALSE:.*]] = arith.constant false
// CHECK-NOT: is_immediate
// CHECK: return %[[TRUE]], %[[FALSE]], %[[FALSE]], %[[TRUE]], %[[FALSE]], %[[TRUE]]
func.func @known_tags(%v: i32, %w: i64, %bits: i64) -> (i1, i1, i1, i1, i1, i1) {
  %imm = rcgmp.bridge.mpz.from_imm %v : i32 to !rcgmp.mpz
  %heap = rcgmp.bridge.mpz.promote %w : i64 to !rcgmp.mpz
  %immortal = rcgmp.bridge.mpz.immortal 2305843009213693952 : i63 : !rcgmp.mpz
  %odd = arith.constant 21 : i64
  %even = arith.constant 8 : i64
  %one = arith.constant 1 : i64
  %tagged = arith.ori %bits, %one : i64
  %a = rcgmp.bridge.mpz.from_bits %odd : i64 to !rcgmp.mpz
  %b = rcgmp.bridge.mpz.from_bits %even : i64 to !rcgmp.mpz
  %c = rcgmp.bridge.mpz.from_bits %tagged : i64 to !rcgmp.mpz
  %t0 = rcgmp.bridge.mpz.is_immediate %imm : !rcgmp.mpz
  %t1 = rcgmp.bridge.mpz.is_immediate %heap : !rcgmp.mpz
  %t2 = rcgmp.bridge.mpz.is_immediate %immortal : !rcgmp.mpz
  %t3 = rcgmp.bridge.mpz.is_immediate %a : !rcgmp.mpz
  %t4 = rcgmp.bridge.mpz.is_immediate %b : !rcgmp.mpz
  %t5 = rcgmp.bridge.mpz.is_immediate %c : !rcgmp.mpz
  return %t0, %t1, %t2, %t3, %t4, %t5 : i1, i1, i1, i1, i1, i1
}

// A test is known in the then region of a conjunction requiring it, and in
// the else region of the test itself, but not in the else region of a
// conjunction.
// CHECK-LABEL: func.func @enclosing_tests
// CHECK-DAG: %[[TRUE:.*]] = arith.constant true
// CHECK-DAG: %[[FALSE:.*]] = arith.constant false
// CHECK: scf.if
// CHECK-NEXT: func.call @use(%[[TRUE]])
// CHECK-NEXT: } else {
// CHECK-NEXT: %[[UNKNOWN:.*]] = rcgmp.bridge.mpz.is_immediate %arg0
// CHECK-NEXT: func.call @use(%[[UNKNOWN]])
// CHECK: scf.if
// CHECK: } else {
// CHECK-NEXT: scf.if
// CHECK-NEXT: func.call @use(%[[FALSE]])
func.func private @use(i1)
func.func @enclosing_tests(%x: !rcgmp.mpz, %y: !rcgmp.mpz, %c: i1) {
  %tx = rcgmp.bridge.mpz.is_immediate %x : !rcgmp.mpz
  %ty = rcgmp.bridge.mpz.is_immediate %y : !rcgmp.mpz
  %both = arith.andi %tx, %ty : i1
  %all = arith.andi %both, %c : i1
  scf.if %all {
    %again = rcgmp.bridge.mpz.is_immediate %x : !rcgmp.mpz
    func.call @use(%again) : (i1) -> ()
  } else {
    %again = rcgmp.bridge.mpz.is_immediate %x : !rcgmp.mpz
    func.call @use(%again) : (i1) -> ()
  }
  scf.if %tx {
    func.call @use(%c) : (i1) -> ()
  } else {
    scf.if %c {
      %again = rcgmp.bridge.mpz.is_immediate %x : !rcgmp.mpz
      func.call @use(%again) : (i1) -> ()
    }
  }
  return
}

// Fast paths test several handles as ((bits & ...) & 1) == 1, which also
// requires each one to be immediate.
// CHECK-LABEL: func.func @combined_test
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: scf.if
// CHECK-NEXT: func.call @use(%[[TRUE]])
func.func @combined_test(%x: !rcgmp.mpz, %y: !rcgmp.mpz, %c: i1) {
  %a = rcgmp.bridge.mpz.to_bits %x : !rcgmp.mpz to i64
  %b = rcgmp.bridge.mpz.to_bits %y : !rcgmp.mpz to i64
  %ab = arith.andi %a, %b : i64
  %one = arith.constant 1 : i64
  %tag = arith.andi %ab, %one : i64
  %both = arith.cmpi eq, %tag, %one : i64
  %valid = arith.andi %both, %c : i1
  scf.if %valid {
    %again = rcgmp.bridge.mpz.is_immediate %y : !rcgmp.mpz
    func.call @use(%again) : (i1) -> ()
  }
  return
}
