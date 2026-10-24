// RUN: %rcgmp-opt %s -rcgmp-optimize-rc | %FileCheck %s

func.func private @use(!rcgmp.mpz)
func.func private @use_i1(i1)

// A retain cancels a release when no use comes between them.
// CHECK-LABEL: func.func @adjacent
// CHECK-NEXT: rcgmp.release %arg0
// CHECK-NEXT: return
func.func @adjacent(%x: !rcgmp.mpz) {
  rcgmp.retain %x : !rcgmp.mpz
  rcgmp.release %x : !rcgmp.mpz
  rcgmp.release %x : !rcgmp.mpz
  return
}

// The addition receives the last reference, so it may reuse the storage.
// CHECK-LABEL: func.func @transfer
// CHECK-NEXT: %[[S:.*]] = rcgmp.addi %arg0, %arg1
// CHECK-NEXT: return %[[S]]
func.func @transfer(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  rcgmp.retain %x : !rcgmp.mpz
  %s = rcgmp.addi %x, %y : !rcgmp.mpz
  rcgmp.release %x : !rcgmp.mpz
  return %s : !rcgmp.mpz
}

// Borrowing uses, here in a loop, keep a reference without the retain.
// CHECK-LABEL: func.func @borrowed
// CHECK-NOT: rcgmp.retain
// CHECK: scf.for
// CHECK: rcgmp.cmpi
// CHECK: rcgmp.release %arg0
// CHECK-NOT: rcgmp.release
func.func @borrowed(%x: !rcgmp.mpz, %y: !rcgmp.mpz, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  rcgmp.retain %x : !rcgmp.mpz
  scf.for %i = %c0 to %n step %c1 {
    %b = rcgmp.cmpi slt, %x, %y : !rcgmp.mpz
    func.call @use_i1(%b) : (i1) -> ()
  }
  rcgmp.release %x : !rcgmp.mpz
  rcgmp.release %x : !rcgmp.mpz
  return
}

// A comparison after the addition still reads the value, so the addition must
// not receive the last reference.
// CHECK-LABEL: func.func @read_after_consumer
// CHECK: rcgmp.retain
// CHECK: rcgmp.addi
// CHECK: rcgmp.cmpi
// CHECK: rcgmp.release
func.func @read_after_consumer(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  rcgmp.retain %x : !rcgmp.mpz
  %s = rcgmp.addi %x, %y : !rcgmp.mpz
  %b = rcgmp.cmpi slt, %x, %s : !rcgmp.mpz
  func.call @use_i1(%b) : (i1) -> ()
  rcgmp.release %x : !rcgmp.mpz
  return %s : !rcgmp.mpz
}

// A call may consume its argument. One call can take the last reference, but
// not one followed by another.
// CHECK-LABEL: func.func @calls
// CHECK-NEXT: call @use(%arg0)
// CHECK-NEXT: rcgmp.retain %arg1
// CHECK-NEXT: call @use(%arg1)
// CHECK-NEXT: call @use(%arg1)
// CHECK-NEXT: rcgmp.release %arg1
func.func @calls(%x: !rcgmp.mpz, %y: !rcgmp.mpz) {
  rcgmp.retain %x : !rcgmp.mpz
  func.call @use(%x) : (!rcgmp.mpz) -> ()
  rcgmp.release %x : !rcgmp.mpz
  rcgmp.retain %y : !rcgmp.mpz
  func.call @use(%y) : (!rcgmp.mpz) -> ()
  func.call @use(%y) : (!rcgmp.mpz) -> ()
  rcgmp.release %y : !rcgmp.mpz
  return
}

// A retain before a branch moves into both arms, where it cancels each arm's
// release; the then arm's addition receives the last reference.
// CHECK-LABEL: func.func @into_branches
// CHECK-NEXT: scf.if
// CHECK-NEXT: rcgmp.addi %arg0, %arg1
// CHECK-NEXT: scf.yield
// CHECK-NEXT: } else {
// CHECK-NEXT: rcgmp.release %arg1
// CHECK-NEXT: scf.yield %arg0
// CHECK-NOT: rcgmp.retain
func.func @into_branches(%x: !rcgmp.mpz, %y: !rcgmp.mpz, %c: i1)
    -> !rcgmp.mpz {
  rcgmp.retain %x : !rcgmp.mpz
  %r = scf.if %c -> !rcgmp.mpz {
    %s = rcgmp.addi %x, %y : !rcgmp.mpz
    rcgmp.release %x : !rcgmp.mpz
    scf.yield %s : !rcgmp.mpz
  } else {
    rcgmp.release %y : !rcgmp.mpz
    rcgmp.release %x : !rcgmp.mpz
    scf.yield %x : !rcgmp.mpz
  }
  return %r : !rcgmp.mpz
}

// A release after a match moves into its arms, where one cancels a retain and
// the other remains.
// CHECK-LABEL: func.func @into_arms
// CHECK: case 0 {
// CHECK-NEXT: %[[S:.*]] = rcgmp.addi %arg1, %arg2
// CHECK-NEXT: rcgmp.yield %[[S]]
// CHECK: default {
// CHECK-NEXT: %[[Z:.*]] = rcgmp.constant 0
// CHECK-NEXT: rcgmp.release %arg2
// CHECK-NEXT: rcgmp.release %arg1
// CHECK-NEXT: rcgmp.yield %[[Z]]
// CHECK-NOT: rcgmp.release
func.func @into_arms(%n: !rcgmp.mpz, %x: !rcgmp.mpz, %y: !rcgmp.mpz)
    -> !rcgmp.mpz {
  %r = rcgmp.match %n : !rcgmp.mpz -> !rcgmp.mpz {
    case 0 {
      rcgmp.retain %x : !rcgmp.mpz
      %s = rcgmp.addi %x, %y : !rcgmp.mpz
      rcgmp.yield %s : !rcgmp.mpz
    }
    default {
      %z = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.release %y : !rcgmp.mpz
      rcgmp.yield %z : !rcgmp.mpz
    }
  }
  rcgmp.release %x : !rcgmp.mpz
  return %r : !rcgmp.mpz
}

// A branch consumes the value on one path, so the comparison after it needs
// the retained reference; the retain moves only into the arm that cancels it.
// CHECK-LABEL: func.func @consumed_in_branch
// CHECK: scf.if
// CHECK-NEXT: rcgmp.retain %arg0
// CHECK-NEXT: call @use(%arg0)
// CHECK-NEXT: } else {
// CHECK-NEXT: }
// CHECK-NEXT: rcgmp.cmpi
// CHECK: rcgmp.release %arg0
func.func @consumed_in_branch(%x: !rcgmp.mpz, %y: !rcgmp.mpz, %c: i1) {
  rcgmp.retain %x : !rcgmp.mpz
  scf.if %c {
    func.call @use(%x) : (!rcgmp.mpz) -> ()
  } else {
    rcgmp.release %x : !rcgmp.mpz
  }
  %b = rcgmp.cmpi slt, %x, %y : !rcgmp.mpz
  func.call @use_i1(%b) : (i1) -> ()
  rcgmp.release %x : !rcgmp.mpz
  return
}

// The retain does not move into a branch without an arm that cancels it, nor
// into a loop, which may run its body any number of times.
// CHECK-LABEL: func.func @kept
// CHECK-NEXT: rcgmp.retain %arg0
// CHECK-NEXT: scf.if
// CHECK-NEXT: call @use(%arg0)
// CHECK: rcgmp.retain %arg0
// CHECK-NEXT: scf.for
func.func @kept(%x: !rcgmp.mpz, %c: i1, %n: index) {
  rcgmp.retain %x : !rcgmp.mpz
  scf.if %c {
    func.call @use(%x) : (!rcgmp.mpz) -> ()
  } else {
    func.call @use(%x) : (!rcgmp.mpz) -> ()
  }
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  rcgmp.retain %x : !rcgmp.mpz
  scf.for %i = %c0 to %n step %c1 {
    rcgmp.retain %x : !rcgmp.mpz
    func.call @use(%x) : (!rcgmp.mpz) -> ()
  }
  rcgmp.release %x : !rcgmp.mpz
  rcgmp.release %x : !rcgmp.mpz
  return
}

// References are only cancelled within a block.
// CHECK-LABEL: func.func @blocks
// CHECK: rcgmp.retain
// CHECK: ^bb1:
// CHECK-NEXT: rcgmp.release
func.func @blocks(%x: !rcgmp.mpz) {
  rcgmp.retain %x : !rcgmp.mpz
  cf.br ^bb1
^bb1:
  rcgmp.release %x : !rcgmp.mpz
  rcgmp.release %x : !rcgmp.mpz
  return
}
