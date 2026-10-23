// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -canonicalize | %FileCheck %s

// Canonicalization rewrites trunci(shrsi) to trunci(shrui) and keeps the
// truncation's overflow flags, which the logical shift falsifies for negative
// integers, so the lowered truncation carries none.
// CHECK-LABEL: func.func @nsw
// CHECK: %[[X:.*]] = arith.shrui
// CHECK: arith.trunci %[[X]] : i64 to i62
func.func @nsw(%x: !rcgmp.mpz) -> i62 {
  %v = rcgmp.trunci %x overflow<nsw> : !rcgmp.mpz to i62
  return %v : i62
}
