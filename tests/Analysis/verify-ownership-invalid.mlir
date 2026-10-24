// RUN: %rcgmp-opt %s -split-input-file -verify-diagnostics -allow-unregistered-dialect -rcgmp-verify-ownership

func.func @use_after_consume(%x: !rcgmp.mpz) {
  // expected-note @below {{last reference consumed here}}
  %y = rcgmp.negi %x : !rcgmp.mpz
  rcgmp.release %y : !rcgmp.mpz
  // expected-error @below {{operand #0 consumes a reference it does not own}}
  rcgmp.release %x : !rcgmp.mpz
  return
}

// -----

// Each operand position needs its own reference.
func.func @aliased_operands(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  // expected-error @below {{operand #1 consumes a reference it does not own}}
  // expected-note @below {{last reference consumed here}}
  %y = rcgmp.addi %x, %x : !rcgmp.mpz
  return %y : !rcgmp.mpz
}

// -----

func.func @borrow_after_consume(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> i1 {
  // expected-note @below {{last reference consumed here}}
  rcgmp.release %x : !rcgmp.mpz
  // expected-error @below {{operand #0 borrows a value after its last reference was consumed}}
  %lt = rcgmp.cmpi slt, %x, %y : !rcgmp.mpz
  rcgmp.release %y : !rcgmp.mpz
  return %lt : i1
}

// -----

func.func @retain_after_consume(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  // expected-note @below {{last reference consumed here}}
  rcgmp.release %x : !rcgmp.mpz
  // expected-error @below {{operand #0 borrows a value after its last reference was consumed}}
  rcgmp.retain %x : !rcgmp.mpz
  return %x : !rcgmp.mpz
}

// -----

// expected-note @below {{owned value defined here}}
func.func @leak_argument(%x: !rcgmp.mpz) {
  // expected-error @below {{leaves its region without consuming an owned value}}
  return
}

// -----

// expected-note @below {{owned value defined here}}
func.func @leak_retained(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  rcgmp.retain %x : !rcgmp.mpz
  // expected-error @below {{leaves its region without consuming an owned value}}
  return %x : !rcgmp.mpz
}

// -----

func.func @leak_in_region(%c: i1, %input: i64) {
  scf.if %c {
    // expected-note @below {{owned value defined here}}
    %x = rcgmp.extsi %input : i64 to !rcgmp.mpz
    // expected-error @below {{leaves its region without consuming an owned value}}
    scf.yield
  }
  return
}

// -----

// The implicit yield of the empty branch takes the location of the scf.if.
func.func @branches_disagree(%c: i1, %x: !rcgmp.mpz) {
  // expected-error @below {{owns different references than another path to the same successor}}
  scf.if %c {
    // expected-note @below {{consumed here}}
    rcgmp.release %x : !rcgmp.mpz
  } else {
  }
  return
}

// -----

// The body must consume the same references on every iteration.
func.func @loop_consumes_outer(%n: index, %x: !rcgmp.mpz) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    // expected-note @below {{consumed here}}
    rcgmp.release %x : !rcgmp.mpz
    // expected-error @below {{owns different references than another path to the same successor}}
    scf.yield
  }
  return
}

// -----

func.func @blocks_disagree(%c: i1, %x: !rcgmp.mpz) {
  cf.cond_br %c, ^left, ^right
^left:
  // expected-note @below {{consumed here}}
  rcgmp.release %x : !rcgmp.mpz
  cf.br ^join
^right:
  // expected-error @below {{owns different references than another path to the same successor}}
  cf.br ^join
^join:
  return
}

// -----

func.func @match_leaks_arm_argument(%n: !rcgmp.mpz) {
  rcgmp.match %n : !rcgmp.mpz {
    // expected-note @below {{owned value defined here}}
    at_least 0 (%k) {
      // expected-error @below {{leaves its region without consuming an owned value}}
      rcgmp.yield
    }
    default {
    }
  }
  return
}

// -----

func.func @unknown_operation(%x: !rcgmp.mpz) {
  // expected-error @below {{cannot determine whether operand #0 is consumed or borrowed}}
  "test.unknown"(%x) : (!rcgmp.mpz) -> ()
  return
}
