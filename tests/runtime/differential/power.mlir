// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts -o /dev/null
// Powers compared against GMP by power.cpp.
func.func @ipowi_local(%x: !rcgmp.mpz, %n: i64) -> !rcgmp.mpz {
  %r = rcgmp.ipowi %x, %n : !rcgmp.mpz, i64
  return %r : !rcgmp.mpz
}
func.func @ipowi_atomic(%x: !rcgmp.mpz<atomic>, %n: i64) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.ipowi %x, %n : !rcgmp.mpz<atomic>, i64
  return %r : !rcgmp.mpz<atomic>
}
