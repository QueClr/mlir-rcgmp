// RUN: %rcgmp-opt %s -split-input-file -verify-diagnostics
func.func @not_narrowest() -> !rcgmp.mpz {
  // expected-error@+1 {{requires a signless integer attribute of the narrowest width holding its value}}
  %r = "rcgmp.constant"() <{value = 5 : i64}> : () -> !rcgmp.mpz
  return %r : !rcgmp.mpz
}
// -----
func.func @index_value() -> !rcgmp.mpz {
  // expected-error@+1 {{requires a signless integer attribute of the narrowest width holding its value}}
  %r = "rcgmp.constant"() <{value = 0 : index}> : () -> !rcgmp.mpz
  return %r : !rcgmp.mpz
}
// -----
func.func @negative_not_narrowest() -> !rcgmp.mpz {
  // expected-error@+1 {{requires a signless integer attribute of the narrowest width holding its value}}
  %r = "rcgmp.constant"() <{value = -8 : i5}> : () -> !rcgmp.mpz
  return %r : !rcgmp.mpz
}
// -----
func.func @unsigned_value() -> !rcgmp.mpz {
  // expected-error@+1 {{requires a signless integer attribute of the narrowest width holding its value}}
  %r = "rcgmp.constant"() <{value = 5 : ui4}> : () -> !rcgmp.mpz
  return %r : !rcgmp.mpz
}
// -----
func.func @zero_width() -> !rcgmp.mpz {
  // expected-error@+1 {{requires a signless integer attribute of the narrowest width holding its value}}
  %r = "rcgmp.constant"() <{value = 0 : i0}> : () -> !rcgmp.mpz
  return %r : !rcgmp.mpz
}
// -----
func.func @missing_value() -> !rcgmp.mpz {
  // expected-error@+1 {{expected integer value}}
  %r = rcgmp.constant : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
// -----
func.func @boolean_value() -> !rcgmp.mpz {
  // expected-error@+1 {{expected integer value}}
  %r = rcgmp.constant true : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
