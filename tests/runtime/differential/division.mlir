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
func.func @edivsi_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.edivsi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @edivsi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.edivsi %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @emodsi_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.emodsi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @emodsi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.emodsi %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @edivmodsi_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> (!rcgmp.mpz, !rcgmp.mpz) {
  %q, %r = rcgmp.edivmodsi %x, %y : !rcgmp.mpz
  return %q, %r : !rcgmp.mpz, !rcgmp.mpz
}
func.func @edivmodsi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> (!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>) {
  %q, %r = rcgmp.edivmodsi %x, %y : !rcgmp.mpz<atomic>
  return %q, %r : !rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>
}
func.func @divsi_undefined_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.divsi %x, %y divzero<undefined> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @divsi_undefined_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.divsi %x, %y divzero<undefined> : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @remsi_undefined_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.remsi %x, %y divzero<undefined> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @remsi_undefined_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.remsi %x, %y divzero<undefined> : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @floordivsi_undefined_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.floordivsi %x, %y divzero<undefined> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @floordivsi_undefined_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.floordivsi %x, %y divzero<undefined> : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @floormodsi_undefined_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.floormodsi %x, %y divzero<undefined> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @floormodsi_undefined_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.floormodsi %x, %y divzero<undefined> : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @divremsi_undefined_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> (!rcgmp.mpz, !rcgmp.mpz) {
  %q, %r = rcgmp.divremsi %x, %y divzero<undefined> : !rcgmp.mpz
  return %q, %r : !rcgmp.mpz, !rcgmp.mpz
}
func.func @divremsi_undefined_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> (!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>) {
  %q, %r = rcgmp.divremsi %x, %y divzero<undefined> : !rcgmp.mpz<atomic>
  return %q, %r : !rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>
}
func.func @floordivmodsi_undefined_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> (!rcgmp.mpz, !rcgmp.mpz) {
  %q, %r = rcgmp.floordivmodsi %x, %y divzero<undefined> : !rcgmp.mpz
  return %q, %r : !rcgmp.mpz, !rcgmp.mpz
}
func.func @floordivmodsi_undefined_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> (!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>) {
  %q, %r = rcgmp.floordivmodsi %x, %y divzero<undefined> : !rcgmp.mpz<atomic>
  return %q, %r : !rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>
}
func.func @edivsi_undefined_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.edivsi %x, %y divzero<undefined> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @edivsi_undefined_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.edivsi %x, %y divzero<undefined> : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @emodsi_undefined_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.emodsi %x, %y divzero<undefined> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @emodsi_undefined_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.emodsi %x, %y divzero<undefined> : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @edivmodsi_undefined_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> (!rcgmp.mpz, !rcgmp.mpz) {
  %q, %r = rcgmp.edivmodsi %x, %y divzero<undefined> : !rcgmp.mpz
  return %q, %r : !rcgmp.mpz, !rcgmp.mpz
}
func.func @edivmodsi_undefined_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> (!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>) {
  %q, %r = rcgmp.edivmodsi %x, %y divzero<undefined> : !rcgmp.mpz<atomic>
  return %q, %r : !rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>
}
func.func @divsi_lean_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.divsi %x, %y divzero<lean> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @divsi_lean_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.divsi %x, %y divzero<lean> : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @remsi_lean_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.remsi %x, %y divzero<lean> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @remsi_lean_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.remsi %x, %y divzero<lean> : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @floordivsi_lean_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.floordivsi %x, %y divzero<lean> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @floordivsi_lean_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.floordivsi %x, %y divzero<lean> : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @floormodsi_lean_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.floormodsi %x, %y divzero<lean> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @floormodsi_lean_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.floormodsi %x, %y divzero<lean> : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @divremsi_lean_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> (!rcgmp.mpz, !rcgmp.mpz) {
  %q, %r = rcgmp.divremsi %x, %y divzero<lean> : !rcgmp.mpz
  return %q, %r : !rcgmp.mpz, !rcgmp.mpz
}
func.func @divremsi_lean_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> (!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>) {
  %q, %r = rcgmp.divremsi %x, %y divzero<lean> : !rcgmp.mpz<atomic>
  return %q, %r : !rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>
}
func.func @floordivmodsi_lean_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> (!rcgmp.mpz, !rcgmp.mpz) {
  %q, %r = rcgmp.floordivmodsi %x, %y divzero<lean> : !rcgmp.mpz
  return %q, %r : !rcgmp.mpz, !rcgmp.mpz
}
func.func @floordivmodsi_lean_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> (!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>) {
  %q, %r = rcgmp.floordivmodsi %x, %y divzero<lean> : !rcgmp.mpz<atomic>
  return %q, %r : !rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>
}
func.func @edivsi_lean_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.edivsi %x, %y divzero<lean> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @edivsi_lean_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.edivsi %x, %y divzero<lean> : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @emodsi_lean_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.emodsi %x, %y divzero<lean> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @emodsi_lean_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.emodsi %x, %y divzero<lean> : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @edivmodsi_lean_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> (!rcgmp.mpz, !rcgmp.mpz) {
  %q, %r = rcgmp.edivmodsi %x, %y divzero<lean> : !rcgmp.mpz
  return %q, %r : !rcgmp.mpz, !rcgmp.mpz
}
func.func @edivmodsi_lean_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> (!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>) {
  %q, %r = rcgmp.edivmodsi %x, %y divzero<lean> : !rcgmp.mpz<atomic>
  return %q, %r : !rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>
}
