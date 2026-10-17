// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts -o /dev/null
// Entry points compared against GMP by differential.c.
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
func.func @retain_local(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  rcgmp.retain %x : !rcgmp.mpz
  return %x : !rcgmp.mpz
}
func.func @retain_atomic(%x: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
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
// from_int entry points take one to three i64 words, least significant first,
// and truncate them to the operand width, so that C passes only i64 arguments.
func.func @from_signed_i61_local(%w0: i64) -> !rcgmp.mpz {
  %x = arith.trunci %w0 : i64 to i61
  %r = rcgmp.from_int %x : i61 to !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @from_unsigned_i61_local(%w0: i64) -> !rcgmp.mpz {
  %x = arith.trunci %w0 : i64 to i61
  %r = rcgmp.from_int %x unsigned : i61 to !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @from_signed_i62_local(%w0: i64) -> !rcgmp.mpz {
  %x = arith.trunci %w0 : i64 to i62
  %r = rcgmp.from_int %x : i62 to !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @from_unsigned_i62_local(%w0: i64) -> !rcgmp.mpz {
  %x = arith.trunci %w0 : i64 to i62
  %r = rcgmp.from_int %x unsigned : i62 to !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @from_signed_i63_local(%w0: i64) -> !rcgmp.mpz {
  %x = arith.trunci %w0 : i64 to i63
  %r = rcgmp.from_int %x : i63 to !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @from_unsigned_i63_local(%w0: i64) -> !rcgmp.mpz {
  %x = arith.trunci %w0 : i64 to i63
  %r = rcgmp.from_int %x unsigned : i63 to !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @from_signed_i64_local(%w0: i64) -> !rcgmp.mpz {
  %r = rcgmp.from_int %w0 : i64 to !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @from_unsigned_i64_local(%w0: i64) -> !rcgmp.mpz {
  %r = rcgmp.from_int %w0 unsigned : i64 to !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @from_signed_i65_local(%w0: i64, %w1: i64) -> !rcgmp.mpz {
  %x0 = arith.extui %w0 : i64 to i128
  %c1 = arith.constant 64 : i128
  %e1 = arith.extui %w1 : i64 to i128
  %s1 = arith.shli %e1, %c1 : i128
  %x1 = arith.ori %x0, %s1 : i128
  %x = arith.trunci %x1 : i128 to i65
  %r = rcgmp.from_int %x : i65 to !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @from_unsigned_i65_local(%w0: i64, %w1: i64) -> !rcgmp.mpz {
  %x0 = arith.extui %w0 : i64 to i128
  %c1 = arith.constant 64 : i128
  %e1 = arith.extui %w1 : i64 to i128
  %s1 = arith.shli %e1, %c1 : i128
  %x1 = arith.ori %x0, %s1 : i128
  %x = arith.trunci %x1 : i128 to i65
  %r = rcgmp.from_int %x unsigned : i65 to !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @from_signed_i128_local(%w0: i64, %w1: i64) -> !rcgmp.mpz {
  %x0 = arith.extui %w0 : i64 to i128
  %c1 = arith.constant 64 : i128
  %e1 = arith.extui %w1 : i64 to i128
  %s1 = arith.shli %e1, %c1 : i128
  %x1 = arith.ori %x0, %s1 : i128
  %r = rcgmp.from_int %x1 : i128 to !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @from_unsigned_i128_local(%w0: i64, %w1: i64) -> !rcgmp.mpz {
  %x0 = arith.extui %w0 : i64 to i128
  %c1 = arith.constant 64 : i128
  %e1 = arith.extui %w1 : i64 to i128
  %s1 = arith.shli %e1, %c1 : i128
  %x1 = arith.ori %x0, %s1 : i128
  %r = rcgmp.from_int %x1 unsigned : i128 to !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @from_signed_i129_local(%w0: i64, %w1: i64, %w2: i64) -> !rcgmp.mpz {
  %x0 = arith.extui %w0 : i64 to i192
  %c1 = arith.constant 64 : i192
  %e1 = arith.extui %w1 : i64 to i192
  %s1 = arith.shli %e1, %c1 : i192
  %x1 = arith.ori %x0, %s1 : i192
  %c2 = arith.constant 128 : i192
  %e2 = arith.extui %w2 : i64 to i192
  %s2 = arith.shli %e2, %c2 : i192
  %x2 = arith.ori %x1, %s2 : i192
  %x = arith.trunci %x2 : i192 to i129
  %r = rcgmp.from_int %x : i129 to !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @from_unsigned_i129_local(%w0: i64, %w1: i64, %w2: i64) -> !rcgmp.mpz {
  %x0 = arith.extui %w0 : i64 to i192
  %c1 = arith.constant 64 : i192
  %e1 = arith.extui %w1 : i64 to i192
  %s1 = arith.shli %e1, %c1 : i192
  %x1 = arith.ori %x0, %s1 : i192
  %c2 = arith.constant 128 : i192
  %e2 = arith.extui %w2 : i64 to i192
  %s2 = arith.shli %e2, %c2 : i192
  %x2 = arith.ori %x1, %s2 : i192
  %x = arith.trunci %x2 : i192 to i129
  %r = rcgmp.from_int %x unsigned : i129 to !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @from_signed_i64_atomic(%w0: i64) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.from_int %w0 : i64 to !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @from_unsigned_i64_atomic(%w0: i64) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.from_int %w0 unsigned : i64 to !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @from_signed_i129_atomic(%w0: i64, %w1: i64, %w2: i64) -> !rcgmp.mpz<atomic> {
  %x0 = arith.extui %w0 : i64 to i192
  %c1 = arith.constant 64 : i192
  %e1 = arith.extui %w1 : i64 to i192
  %s1 = arith.shli %e1, %c1 : i192
  %x1 = arith.ori %x0, %s1 : i192
  %c2 = arith.constant 128 : i192
  %e2 = arith.extui %w2 : i64 to i192
  %s2 = arith.shli %e2, %c2 : i192
  %x2 = arith.ori %x1, %s2 : i192
  %x = arith.trunci %x2 : i192 to i129
  %r = rcgmp.from_int %x : i129 to !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @from_unsigned_i129_atomic(%w0: i64, %w1: i64, %w2: i64) -> !rcgmp.mpz<atomic> {
  %x0 = arith.extui %w0 : i64 to i192
  %c1 = arith.constant 64 : i192
  %e1 = arith.extui %w1 : i64 to i192
  %s1 = arith.shli %e1, %c1 : i192
  %x1 = arith.ori %x0, %s1 : i192
  %c2 = arith.constant 128 : i192
  %e2 = arith.extui %w2 : i64 to i192
  %s2 = arith.shli %e2, %c2 : i192
  %x2 = arith.ori %x1, %s2 : i192
  %x = arith.trunci %x2 : i192 to i129
  %r = rcgmp.from_int %x unsigned : i129 to !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
