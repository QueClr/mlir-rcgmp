// RUN: %rcgmp-test-opt %s -allow-unregistered-dialect -test-rcgmp-fast-path-lowering -split-input-file -verify-diagnostics

// A runtime declaration without the reuse mask does not match the call.
func.func private @test_select(!rcgmp.mpz, i1, !rcgmp.mpz) -> !rcgmp.mpz

func.func @select(%x: !rcgmp.mpz, %c: i1, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  // expected-error @+1 {{failed to legalize operation 'test.select'}}
  %r = "test.select"(%x, %c, %y) : (!rcgmp.mpz, i1, !rcgmp.mpz) -> !rcgmp.mpz
  return %r : !rcgmp.mpz
}

// -----

// A runtime declaration returning the operation result type instead of the
// runtime result type does not match the call.
func.func private @test_equal_word(!rcgmp.mpz, i64) -> i1

func.func @equal_word(%x: !rcgmp.mpz, %w: i64) -> i1 {
  // expected-error @+1 {{failed to legalize operation 'test.equal_word'}}
  %r = "test.equal_word"(%x, %w) : (!rcgmp.mpz, i64) -> i1
  return %r : i1
}
