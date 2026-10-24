// RUN: %rcgmp-opt %s -split-input-file -verify-diagnostics

func.func @half_result(%x: !rcgmp.mpz) -> f16 {
  // expected-error @below {{result #0 must be 32-bit float or 64-bit float}}
  %f = rcgmp.sitofp %x : !rcgmp.mpz to f16
  return %f : f16
}

