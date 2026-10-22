// RUN: %rcgmp-opt %s -split-input-file -verify-diagnostics

func.func @default_not_last(%n: !rcgmp.mpz) {
  // expected-error@+1 {{requires the 'default' arm to be last}}
  rcgmp.match %n : !rcgmp.mpz {
    default {
    }
    case 0 {
    }
  }
  return
}

// -----

func.func @case_argument(%n: !rcgmp.mpz) {
  // expected-error@+1 {{arm 0 takes at most 0 arguments}}
  rcgmp.match %n : !rcgmp.mpz {
    case 0 (%k) {
    }
  }
  return
}

// -----

func.func @argument_type(%n: !rcgmp.mpz) {
  // expected-error@+1 {{arm 0 argument must have the scrutinee type}}
  "rcgmp.match"(%n) <{arms = [#rcgmp.arm<default, []>]}> ({
  ^bb0(%m: !rcgmp.mpz<atomic>):
    rcgmp.yield
  }) : (!rcgmp.mpz) -> ()
  return
}

// -----

func.func @yield_types(%n: !rcgmp.mpz, %x: i32) -> i64 {
  // expected-error@+1 {{arm 0 must yield values of the result types}}
  %r = rcgmp.match %n : !rcgmp.mpz -> i64 {
    case 0 {
      rcgmp.yield %x : i32
    }
  }
  return %r : i64
}

// -----

func.func @duplicate(%n: !rcgmp.mpz) {
  // expected-error@+1 {{lists 2 more than once}}
  rcgmp.match %n : !rcgmp.mpz {
    case 2, 2 {
    }
  }
  return
}

// -----

func.func @at_least_values(%n: !rcgmp.mpz) {
  // expected-error@+1 {{an 'at_least' arm requires exactly one value}}
  "rcgmp.match"(%n) <{arms = [#rcgmp.arm<at_least, [1 : i2, 2 : i3]>]}> ({
    rcgmp.yield
  }) : (!rcgmp.mpz) -> ()
  return
}

// -----

func.func @wide_value(%n: !rcgmp.mpz) {
  // expected-error@+1 {{requires signless integer values of the narrowest width holding them}}
  "rcgmp.match"(%n) <{arms = [#rcgmp.arm<case, [1 : i64]>]}> ({
    rcgmp.yield
  }) : (!rcgmp.mpz) -> ()
  return
}

// -----

func.func @no_arms(%n: !rcgmp.mpz) {
  // expected-error@+1 {{requires at least one arm}}
  rcgmp.match %n : !rcgmp.mpz {
  }
  return
}

// -----

func.func @unknown_arm(%n: !rcgmp.mpz) {
  rcgmp.match %n : !rcgmp.mpz {
    // expected-error@+1 {{expected 'case', 'at_least', or 'default'}}
    otherwise {
    }
  }
  return
}
