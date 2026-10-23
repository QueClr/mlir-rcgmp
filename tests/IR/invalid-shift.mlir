// RUN: %rcgmp-opt %s -split-input-file -verify-diagnostics

func.func @mixed_counting(%x: !rcgmp.mpz, %n: i64) -> !rcgmp.mpz<atomic> {
  // expected-error@+1 {{all of {value, result} have same type}}
  %r = "rcgmp.shli"(%x, %n) : (!rcgmp.mpz, i64) -> !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}

// -----

func.func @index_amount(%x: !rcgmp.mpz, %n: index) -> !rcgmp.mpz {
  // expected-error@+1 {{operand #1 must be signless integer}}
  %r = "rcgmp.shrsi"(%x, %n) : (!rcgmp.mpz, index) -> !rcgmp.mpz
  return %r : !rcgmp.mpz
}
