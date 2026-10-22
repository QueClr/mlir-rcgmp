// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts -o /dev/null
// Unary operations compared against GMP by unary.cpp and chain.cpp.
func.func @noti_local(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.noti %x : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @noti_atomic(%x: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.noti %x : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @negi_local(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.negi %x : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @negi_atomic(%x: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.negi %x : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @absi_local(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.absi %x : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @absi_atomic(%x: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.absi %x : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
