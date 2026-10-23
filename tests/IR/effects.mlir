// RUN: %rcgmp-opt %s -cse -canonicalize | %FileCheck %s

// Each extsi creates its own reference, so equal inputs are not merged.
// CHECK-LABEL: func.func @distinct_allocations
// CHECK: %[[A:.*]] = rcgmp.extsi
// CHECK: %[[B:.*]] = rcgmp.extsi
// CHECK: rcgmp.release %[[A]]
// CHECK: rcgmp.release %[[B]]
func.func @distinct_allocations(%i: i64) {
  %a = rcgmp.extsi %i : i64 to !rcgmp.mpz
  %b = rcgmp.extsi %i : i64 to !rcgmp.mpz
  rcgmp.release %a : !rcgmp.mpz
  rcgmp.release %b : !rcgmp.mpz
  return
}

// An unused allocation is dead; reference-count updates are not.
// CHECK-LABEL: func.func @dead_code
// CHECK-NOT: rcgmp.extsi
// CHECK: rcgmp.retain %[[X:.*]] :
// CHECK-NEXT: rcgmp.release %[[X]]
// CHECK-NEXT: rcgmp.release %[[X]]
func.func @dead_code(%i: i64, %x: !rcgmp.mpz) {
  %unused = rcgmp.extsi %i : i64 to !rcgmp.mpz
  rcgmp.retain %x : !rcgmp.mpz
  rcgmp.release %x : !rcgmp.mpz
  rcgmp.release %x : !rcgmp.mpz
  return
}

// Uniqueness tests only read the count; a count update between them keeps
// both.
// CHECK-LABEL: func.func @uniqueness
// CHECK: rcgmp.bridge.mpz.is_unique
// CHECK-NEXT: rcgmp.bridge.mpz.inc_ref
// CHECK-NEXT: rcgmp.bridge.mpz.is_unique
// CHECK-NOT: rcgmp.bridge.mpz.is_unique
func.func @uniqueness(%x: !rcgmp.mpz) -> (i1, i1, i1, i1) {
  %a = rcgmp.bridge.mpz.is_unique %x : !rcgmp.mpz
  %b = rcgmp.bridge.mpz.is_unique %x : !rcgmp.mpz
  %overflow = rcgmp.bridge.mpz.inc_ref %x : !rcgmp.mpz
  %c = rcgmp.bridge.mpz.is_unique %x : !rcgmp.mpz
  return %a, %b, %overflow, %c : i1, i1, i1, i1
}

// Comparisons only read their borrowed operands: unused ones are erased, and
// equal ones merge unless a count update intervenes.
// CHECK-LABEL: func.func @comparisons
// CHECK: %[[A:.*]] = rcgmp.cmpi slt, %[[X:.*]], %[[Y:.*]] :
// CHECK-NEXT: rcgmp.retain %[[X]]
// CHECK-NEXT: %[[C:.*]] = rcgmp.cmpi slt, %[[X]], %[[Y]] :
// CHECK-NEXT: return %[[A]], %[[A]], %[[C]]
func.func @comparisons(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> (i1, i1, i1) {
  %unused = rcgmp.cmpi eq, %x, %y : !rcgmp.mpz
  %a = rcgmp.cmpi slt, %x, %y : !rcgmp.mpz
  %b = rcgmp.cmpi slt, %x, %y : !rcgmp.mpz
  rcgmp.retain %x : !rcgmp.mpz
  %c = rcgmp.cmpi slt, %x, %y : !rcgmp.mpz
  return %a, %b, %c : i1, i1, i1
}
