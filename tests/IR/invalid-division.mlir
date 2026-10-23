// RUN: %rcgmp-opt %s -split-input-file -verify-diagnostics

func.func @mixed_counting(%x: !rcgmp.mpz, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz {
  // expected-error@+1 {{requires the same type for all operands and results}}
  %r = "rcgmp.divsi"(%x, %y) : (!rcgmp.mpz, !rcgmp.mpz<atomic>) -> !rcgmp.mpz
  return %r : !rcgmp.mpz
}

// -----

func.func @mixed_results(%x: !rcgmp.mpz, %y: !rcgmp.mpz) {
  // expected-error@+1 {{requires the same type for all operands and results}}
  %q, %r = "rcgmp.divremsi"(%x, %y) : (!rcgmp.mpz, !rcgmp.mpz) -> (!rcgmp.mpz, !rcgmp.mpz<atomic>)
  return
}
