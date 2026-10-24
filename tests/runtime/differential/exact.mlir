// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts -o /dev/null
// Exact quotients compared against GMP by exact.cpp.
func.func @divsi_exact_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.divsi %x, %y exact : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @divsi_exact_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.divsi %x, %y exact : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @divsi_exact_undefined_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.divsi %x, %y exact divzero<undefined> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @divsi_exact_undefined_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.divsi %x, %y exact divzero<undefined> : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @divsi_exact_lean_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.divsi %x, %y exact divzero<lean> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @divsi_exact_lean_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.divsi %x, %y exact divzero<lean> : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @floordivsi_exact_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.floordivsi %x, %y exact : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @floordivsi_exact_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.floordivsi %x, %y exact : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @floordivsi_exact_undefined_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.floordivsi %x, %y exact divzero<undefined> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @floordivsi_exact_undefined_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.floordivsi %x, %y exact divzero<undefined> : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @floordivsi_exact_lean_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.floordivsi %x, %y exact divzero<lean> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @floordivsi_exact_lean_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.floordivsi %x, %y exact divzero<lean> : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @edivsi_exact_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.edivsi %x, %y exact : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @edivsi_exact_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.edivsi %x, %y exact : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @edivsi_exact_undefined_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.edivsi %x, %y exact divzero<undefined> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @edivsi_exact_undefined_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.edivsi %x, %y exact divzero<undefined> : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @edivsi_exact_lean_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.edivsi %x, %y exact divzero<lean> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @edivsi_exact_lean_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.edivsi %x, %y exact divzero<lean> : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
