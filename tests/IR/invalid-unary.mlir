// RUN: %rcgmp-opt %s -verify-diagnostics

func.func @mixed_counting(%x: !rcgmp.mpz) -> !rcgmp.mpz<atomic> {
  // expected-error@+1 {{requires the same type for all operands and results}}
  %r = "rcgmp.negi"(%x) : (!rcgmp.mpz) -> !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
