// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts -o /dev/null
// Shifts compared against GMP by shift.cpp and chain.cpp.
func.func @shli_local(%x: !rcgmp.mpz, %n: i64) -> !rcgmp.mpz {
  %r = rcgmp.shli %x, %n : !rcgmp.mpz, i64
  return %r : !rcgmp.mpz
}
func.func @shli_atomic(%x: !rcgmp.mpz<atomic>, %n: i64) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.shli %x, %n : !rcgmp.mpz<atomic>, i64
  return %r : !rcgmp.mpz<atomic>
}
func.func @shrsi_local(%x: !rcgmp.mpz, %n: i64) -> !rcgmp.mpz {
  %r = rcgmp.shrsi %x, %n : !rcgmp.mpz, i64
  return %r : !rcgmp.mpz
}
func.func @shrsi_atomic(%x: !rcgmp.mpz<atomic>, %n: i64) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.shrsi %x, %n : !rcgmp.mpz<atomic>, i64
  return %r : !rcgmp.mpz<atomic>
}
