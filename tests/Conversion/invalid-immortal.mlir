// RUN: %rcgmp-opt %s -split-input-file -convert-to-llvm -verify-diagnostics
llvm.func @__rcgmp_mpz_2000000000000000()
func.func @function() -> !rcgmp.mpz {
  // expected-error@+2 {{failed to legalize operation}}
  // expected-error@+1 {{incompatible definition of immortal integer __rcgmp_mpz_2000000000000000}}
  %r = rcgmp.bridge.mpz.immortal 2305843009213693952 : i63 : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
// -----
llvm.mlir.global private @__rcgmp_mpz_2000000000000000(dense<[0, 1, 1, 2305843009213693952]> : tensor<4xi64>) {alignment = 8 : i64} : !llvm.array<4 x i64>
func.func @mutable() -> !rcgmp.mpz {
  // expected-error@+2 {{failed to legalize operation}}
  // expected-error@+1 {{incompatible definition of immortal integer __rcgmp_mpz_2000000000000000}}
  %r = rcgmp.bridge.mpz.immortal 2305843009213693952 : i63 : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
// -----
llvm.mlir.global internal constant @__rcgmp_mpz_2000000000000000(dense<[0, 1, 1, 2305843009213693952]> : tensor<4xi64>) {alignment = 8 : i64} : !llvm.array<4 x i64>
func.func @internal() -> !rcgmp.mpz {
  // expected-error@+2 {{failed to legalize operation}}
  // expected-error@+1 {{incompatible definition of immortal integer __rcgmp_mpz_2000000000000000}}
  %r = rcgmp.bridge.mpz.immortal 2305843009213693952 : i63 : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
// -----
llvm.mlir.global private constant @__rcgmp_mpz_2000000000000000(dense<[0, 1, 1, 2305843009213693952]> : tensor<4xi64>) {alignment = 1 : i64} : !llvm.array<4 x i64>
func.func @underaligned() -> !rcgmp.mpz {
  // expected-error@+2 {{failed to legalize operation}}
  // expected-error@+1 {{incompatible definition of immortal integer __rcgmp_mpz_2000000000000000}}
  %r = rcgmp.bridge.mpz.immortal 2305843009213693952 : i63 : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
// -----
llvm.mlir.global private constant @__rcgmp_mpz_2000000000000000(dense<[1, 1, 1, 2305843009213693952]> : tensor<4xi64>) {alignment = 8 : i64} : !llvm.array<4 x i64>
func.func @counted() -> !rcgmp.mpz {
  // expected-error@+2 {{failed to legalize operation}}
  // expected-error@+1 {{incompatible definition of immortal integer __rcgmp_mpz_2000000000000000}}
  %r = rcgmp.bridge.mpz.immortal 2305843009213693952 : i63 : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
