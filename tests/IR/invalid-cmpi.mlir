// RUN: %rcgmp-opt %s -split-input-file -verify-diagnostics

// Integers are signed, so there are no unsigned predicates.
func.func @unsigned_predicate(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> i1 {
  // expected-error@+1 {{expected string or keyword containing one of the following enum values for attribute 'predicate' [eq, ne, slt, sle, sgt, sge]}}
  %r = rcgmp.cmpi ult, %x, %y : !rcgmp.mpz
  return %r : i1
}

// -----

func.func @mixed_counting(%x: !rcgmp.mpz, %y: !rcgmp.mpz<atomic>) -> i1 {
  // expected-error@+1 {{requires all operands to have the same type}}
  %r = "rcgmp.cmpi"(%x, %y) <{predicate = 0 : i64}> : (!rcgmp.mpz, !rcgmp.mpz<atomic>) -> i1
  return %r : i1
}
