// RUN: %rcgmp-opt %s -cse -canonicalize | %FileCheck %s

// Each from_int creates its own reference, so equal inputs are not merged.
// CHECK-LABEL: func.func @distinct_allocations
// CHECK: %[[A:.*]] = rcgmp.from_int
// CHECK: %[[B:.*]] = rcgmp.from_int
// CHECK: rcgmp.release %[[A]]
// CHECK: rcgmp.release %[[B]]
func.func @distinct_allocations(%i: i64) {
  %a = rcgmp.from_int %i : i64 to !rcgmp.mpz
  %b = rcgmp.from_int %i : i64 to !rcgmp.mpz
  rcgmp.release %a : !rcgmp.mpz
  rcgmp.release %b : !rcgmp.mpz
  return
}

// An unused allocation is dead; reference-count updates are not.
// CHECK-LABEL: func.func @dead_code
// CHECK-NOT: rcgmp.from_int
// CHECK: rcgmp.retain %[[X:.*]] :
// CHECK-NEXT: rcgmp.release %[[X]]
// CHECK-NEXT: rcgmp.release %[[X]]
func.func @dead_code(%i: i64, %x: !rcgmp.mpz) {
  %unused = rcgmp.from_int %i : i64 to !rcgmp.mpz
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
