// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts -o /dev/null
// Reference counting checked by counting.cpp and used throughout the harness.
func.func @retain_local(%x: !rcgmp.mpz {rcgmp.borrowed}) -> !rcgmp.mpz {
  rcgmp.retain %x : !rcgmp.mpz
  return %x : !rcgmp.mpz
}
func.func @retain_atomic(%x: !rcgmp.mpz<atomic> {rcgmp.borrowed}) -> !rcgmp.mpz<atomic> {
  rcgmp.retain %x : !rcgmp.mpz<atomic>
  return %x : !rcgmp.mpz<atomic>
}
func.func @release_local(%x: !rcgmp.mpz) {
  rcgmp.release %x : !rcgmp.mpz
  return
}
func.func @release_atomic(%x: !rcgmp.mpz<atomic>) {
  rcgmp.release %x : !rcgmp.mpz<atomic>
  return
}
