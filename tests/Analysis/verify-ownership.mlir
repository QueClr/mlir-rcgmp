// RUN: %rcgmp-opt %s -rcgmp-verify-ownership | %FileCheck %s
// Valid ownership passes through unchanged.

// Retaining adds a reference to the same value, which each operand position
// then consumes.
// CHECK-LABEL: func.func @straight_line
func.func @straight_line(%input: i64) -> !rcgmp.mpz {
  %x = rcgmp.extsi %input : i64 to !rcgmp.mpz
  rcgmp.retain %x : !rcgmp.mpz
  %y = rcgmp.addi %x, %x : !rcgmp.mpz
  rcgmp.retain %y : !rcgmp.mpz
  rcgmp.release %y : !rcgmp.mpz
  return %y : !rcgmp.mpz
}

// Borrowing operations leave the reference in place.
// CHECK-LABEL: func.func @borrows
func.func @borrows(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> i1 {
  %lt = rcgmp.cmpi slt, %x, %y : !rcgmp.mpz
  %bit = rcgmp.testbiti %x, %lt : !rcgmp.mpz, i1
  rcgmp.release %x : !rcgmp.mpz
  rcgmp.release %y : !rcgmp.mpz
  return %bit : i1
}

// Constants may be consumed any number of times, or not at all.
// CHECK-LABEL: func.func @constants
func.func @constants() -> !rcgmp.mpz {
  %big = rcgmp.constant -340282366920938463463374607431768211456 : !rcgmp.mpz
  %unused = rcgmp.constant 7 : !rcgmp.mpz
  %square = rcgmp.muli %big, %big : !rcgmp.mpz
  rcgmp.release %big : !rcgmp.mpz
  return %square : !rcgmp.mpz
}

// A borrowed argument must be retained before a reference to it is consumed.
// CHECK-LABEL: func.func @borrowed
func.func @borrowed(%x: !rcgmp.mpz<atomic> {rcgmp.borrowed}) -> !rcgmp.mpz<atomic> {
  rcgmp.retain %x : !rcgmp.mpz<atomic>
  rcgmp.retain %x : !rcgmp.mpz<atomic>
  %r = rcgmp.muli %x, %x : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}

// Calls consume their arguments, except those the callee borrows, and return
// owned results. Borrowing a reference twice does not consume it.
// CHECK-LABEL: func.func @calls
func.func @calls(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %copy = call @borrow_local(%x) : (!rcgmp.mpz) -> !rcgmp.mpz
  call @borrow_both(%x, %x) : (!rcgmp.mpz, !rcgmp.mpz) -> ()
  %r = call @consume(%x, %copy) : (!rcgmp.mpz, !rcgmp.mpz) -> !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func private @borrow_local(!rcgmp.mpz {rcgmp.borrowed}) -> !rcgmp.mpz
func.func private @borrow_both(!rcgmp.mpz {rcgmp.borrowed}, !rcgmp.mpz {rcgmp.borrowed})
func.func private @consume(!rcgmp.mpz, !rcgmp.mpz) -> !rcgmp.mpz

// Both branches consume the outer references, and each yields a new one.
// CHECK-LABEL: func.func @if_else
func.func @if_else(%c: i1, %x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = scf.if %c -> !rcgmp.mpz {
    rcgmp.release %x : !rcgmp.mpz
    scf.yield %y : !rcgmp.mpz
  } else {
    %s = rcgmp.subi %x, %y : !rcgmp.mpz
    scf.yield %s : !rcgmp.mpz
  }
  return %r : !rcgmp.mpz
}

// A region may borrow, or retain and consume, an outer value.
// CHECK-LABEL: func.func @if_borrow
func.func @if_borrow(%c: i1, %x: !rcgmp.mpz) -> !rcgmp.mpz {
  scf.if %c {
    rcgmp.retain %x : !rcgmp.mpz
    rcgmp.release %x : !rcgmp.mpz
  }
  return %x : !rcgmp.mpz
}

// Loops carry references through their iteration arguments.
// CHECK-LABEL: func.func @loops
func.func @loops(%n: index, %x: !rcgmp.mpz) -> !rcgmp.mpz {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %shift = arith.constant 1 : i64
  %one = rcgmp.constant 1 : !rcgmp.mpz
  %sum = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %x) -> !rcgmp.mpz {
    %next = rcgmp.addi %acc, %one : !rcgmp.mpz
    scf.yield %next : !rcgmp.mpz
  }
  %r = scf.while (%v = %sum) : (!rcgmp.mpz) -> !rcgmp.mpz {
    %small = rcgmp.cmpi slt, %v, %one : !rcgmp.mpz
    scf.condition(%small) %v : !rcgmp.mpz
  } do {
  ^bb0(%v: !rcgmp.mpz):
    %next = rcgmp.shli %v, %shift : !rcgmp.mpz, i64
    scf.yield %next : !rcgmp.mpz
  }
  return %r : !rcgmp.mpz
}

// Block arguments receive forwarded references, which paths to the same block
// must agree on.
// CHECK-LABEL: func.func @blocks
func.func @blocks(%c: i1, %x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  cf.cond_br %c, ^left, ^right
^left:
  rcgmp.release %y : !rcgmp.mpz
  cf.br ^join(%x : !rcgmp.mpz)
^right:
  %s = rcgmp.addi %x, %y : !rcgmp.mpz
  cf.br ^join(%s : !rcgmp.mpz)
^join(%r: !rcgmp.mpz):
  return %r : !rcgmp.mpz
}

// A path that never returns may still own references.
// CHECK-LABEL: func.func @unreachable
func.func @unreachable(%c: i1, %x: !rcgmp.mpz) -> !rcgmp.mpz {
  cf.cond_br %c, ^dead, ^live
^dead:
  llvm.unreachable
^live:
  return %x : !rcgmp.mpz
}

// The match consumes its scrutinee, and arm arguments own a new reference.
// CHECK-LABEL: func.func @match
func.func @match(%n: !rcgmp.mpz, %a: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.match %n : !rcgmp.mpz -> !rcgmp.mpz {
    case 0 {
      rcgmp.yield %a : !rcgmp.mpz
    }
    at_least 1 (%k) {
      %s = rcgmp.addi %k, %a : !rcgmp.mpz
      rcgmp.yield %s : !rcgmp.mpz
    }
    default (%m) {
      rcgmp.release %a : !rcgmp.mpz
      rcgmp.yield %m : !rcgmp.mpz
    }
  }
  return %r : !rcgmp.mpz
}

// A reconstructed handle owns no reference: it may be borrowed or retained.
// CHECK-LABEL: func.func @views
func.func @views(%bits: i64, %x: !rcgmp.mpz) -> !rcgmp.mpz {
  %v = rcgmp.bridge.mpz.from_bits %bits : i64 to !rcgmp.mpz
  %lt = rcgmp.cmpi slt, %v, %x : !rcgmp.mpz
  rcgmp.retain %v : !rcgmp.mpz
  %r = rcgmp.addi %v, %x : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
