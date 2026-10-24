// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts -o /dev/null
// Remainders by powers of two compared against GMP by floormod2exp.cpp.
func.func @floormod2expi_local(%x: !rcgmp.mpz, %n: i64) -> !rcgmp.mpz {
  %r = rcgmp.floormod2expi %x, %n : !rcgmp.mpz, i64
  return %r : !rcgmp.mpz
}
func.func @floormod2expi_atomic(%x: !rcgmp.mpz<atomic>, %n: i64) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.floormod2expi %x, %n : !rcgmp.mpz<atomic>, i64
  return %r : !rcgmp.mpz<atomic>
}
