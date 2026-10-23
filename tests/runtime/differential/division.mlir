// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts -o /dev/null
// Divisions compared against GMP by division.cpp.
func.func @divsi_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.divsi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @divsi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.divsi %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @remsi_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.remsi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @remsi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.remsi %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @floordivsi_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.floordivsi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @floordivsi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.floordivsi %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @floormodsi_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.floormodsi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @floormodsi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.floormodsi %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @divremsi_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> (!rcgmp.mpz, !rcgmp.mpz) {
  %q, %r = rcgmp.divremsi %x, %y : !rcgmp.mpz
  return %q, %r : !rcgmp.mpz, !rcgmp.mpz
}
func.func @divremsi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> (!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>) {
  %q, %r = rcgmp.divremsi %x, %y : !rcgmp.mpz<atomic>
  return %q, %r : !rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>
}
func.func @floordivmodsi_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> (!rcgmp.mpz, !rcgmp.mpz) {
  %q, %r = rcgmp.floordivmodsi %x, %y : !rcgmp.mpz
  return %q, %r : !rcgmp.mpz, !rcgmp.mpz
}
func.func @floordivmodsi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> (!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>) {
  %q, %r = rcgmp.floordivmodsi %x, %y : !rcgmp.mpz<atomic>
  return %q, %r : !rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>
}
