// RUN: %rcgmp-opt %s -split-input-file -verify-diagnostics

func.func @index_result(%x: !rcgmp.mpz) -> index {
  // expected-error@+1 {{result #0 must be signless integer}}
  %r = "rcgmp.trunci"(%x) : (!rcgmp.mpz) -> index
  return %r : index
}

// -----

func.func @wide_overflow(%x: !rcgmp.mpz) -> i8 {
  // expected-error@+1 {{result #1 must be 1-bit signless integer}}
  %r, %o = "rcgmp.trunci_overflow"(%x) : (!rcgmp.mpz) -> (i8, i8)
  return %r : i8
}
