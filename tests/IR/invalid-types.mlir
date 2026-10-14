// RUN: %rcgmp-opt %s -split-input-file -verify-diagnostics

// expected-error @+1 {{expected 'atomic'}}
func.func private @explicit_nonatomic(!rcgmp.mpz<nonatomic>)

// -----

// expected-error @+1 {{expected 'atomic'}}
func.func private @empty_mode(!rcgmp.mpz<>)

// -----

// expected-error @+1 {{expected '>'}}
func.func private @extra_mode(!rcgmp.mpz<atomic, atomic>)

// -----

// expected-error @+1 {{type `mpq` in dialect `rcgmp`}}
func.func private @unsupported_type(!rcgmp.mpq)

// -----

func.func @mismatched_mode(%value: !rcgmp.mpz) -> !rcgmp.mpz<atomic> {
  // expected-error @+1 {{type of return operand 0}}
  return %value : !rcgmp.mpz
}
