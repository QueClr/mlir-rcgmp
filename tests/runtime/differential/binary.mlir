// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts -o /dev/null
// Binary operations compared against GMP by binary.cpp and chain.cpp.
func.func @addi_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.addi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @addi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.addi %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @subi_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.subi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @subi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.subi %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @monusi_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.monusi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @monusi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.monusi %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @muli_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.muli %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @muli_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.muli %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @andi_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.andi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @andi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.andi %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @xori_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.xori %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @xori_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.xori %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @ori_local(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.ori %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @ori_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.ori %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
