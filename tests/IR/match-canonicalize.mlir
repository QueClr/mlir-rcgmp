// RUN: %rcgmp-opt %s -canonicalize | %FileCheck %s

// Arms that earlier arms shadow are dropped, and so are case values that an
// earlier arm selects.
// CHECK-LABEL: func.func @unreachable
// CHECK: rcgmp.match
// CHECK-NEXT: at_least 10 {
// CHECK: case 3 {
// CHECK: default {
// CHECK-NOT: case
func.func @unreachable(%n: !rcgmp.mpz, %a: i64, %b: i64) -> i64 {
  %r = rcgmp.match %n : !rcgmp.mpz -> i64 {
    at_least 10 {
      rcgmp.yield %a : i64
    }
    case 12, 3 {
      rcgmp.yield %b : i64
    }
    case 11 {
      rcgmp.yield %a : i64
    }
    default {
      rcgmp.yield %b : i64
    }
  }
  return %r : i64
}

// A match on the remainder of an at_least arm, as nested successor patterns
// produce, joins its parent: n = 2 is case 0 of the inner match, n >= 3 binds
// n - 3 directly, and the inner default would only see negative remainders.
// CHECK-LABEL: func.func @successors
// CHECK: rcgmp.match %{{.*}} : !rcgmp.mpz -> i64 {
// CHECK-NEXT: case 0 {
// CHECK: case 1 {
// CHECK: case 2 {
// CHECK: at_least 3 (%[[K:.*]]) {
// CHECK-NEXT: func.call @use(%[[K]])
// CHECK-NOT: rcgmp.match
func.func private @use(!rcgmp.mpz) -> i64
func.func @successors(%n: !rcgmp.mpz, %a: i64, %b: i64) -> i64 {
  %r = rcgmp.match %n : !rcgmp.mpz -> i64 {
    case 0 {
      rcgmp.yield %a : i64
    }
    at_least 1 (%m) {
      %s = rcgmp.match %m : !rcgmp.mpz -> i64 {
        case 0 {
          rcgmp.yield %b : i64
        }
        at_least 1 (%l) {
          %t = rcgmp.match %l : !rcgmp.mpz -> i64 {
            case 0 {
              rcgmp.yield %a : i64
            }
            at_least 1 (%k) {
              %u = func.call @use(%k) : (!rcgmp.mpz) -> i64
              rcgmp.yield %u : i64
            }
          }
          rcgmp.yield %t : i64
        }
      }
      rcgmp.yield %s : i64
    }
  }
  return %r : i64
}

// An inner default becomes at_least the parent threshold, binding the same
// remainder.
// CHECK-LABEL: func.func @inner_default
// CHECK: case 5 {
// CHECK: at_least 7 {
// CHECK: at_least 5 (%[[M:.*]]) {
// CHECK-NEXT: func.call @use(%[[M]])
func.func @inner_default(%n: !rcgmp.mpz, %a: i64) -> i64 {
  %r = rcgmp.match %n : !rcgmp.mpz -> i64 {
    at_least 5 (%m) {
      %s = rcgmp.match %m : !rcgmp.mpz -> i64 {
        case 0, -1 {
          rcgmp.yield %a : i64
        }
        at_least 2 {
          rcgmp.yield %a : i64
        }
        default (%k) {
          %u = func.call @use(%k) : (!rcgmp.mpz) -> i64
          rcgmp.yield %u : i64
        }
      }
      rcgmp.yield %s : i64
    }
    default {
      rcgmp.yield %a : i64
    }
  }
  return %r : i64
}

// A negative inner threshold binds a different remainder, so the matches stay
// apart.
// CHECK-LABEL: func.func @negative_inner_threshold
// CHECK: rcgmp.match
// CHECK: rcgmp.match
func.func @negative_inner_threshold(%n: !rcgmp.mpz, %a: i64) -> i64 {
  %r = rcgmp.match %n : !rcgmp.mpz -> i64 {
    at_least 5 (%m) {
      %s = rcgmp.match %m : !rcgmp.mpz -> i64 {
        at_least -1 (%k) {
          %u = func.call @use(%k) : (!rcgmp.mpz) -> i64
          rcgmp.yield %u : i64
        }
      }
      rcgmp.yield %s : i64
    }
    default {
      rcgmp.yield %a : i64
    }
  }
  return %r : i64
}

// An argument that is only released is dropped, so the match releases the
// scrutinee without computing it.
// CHECK-LABEL: func.func @released
// CHECK: at_least 1 {
// CHECK-NOT: rcgmp.release
// CHECK: default {
func.func @released(%n: !rcgmp.mpz, %a: i64) -> i64 {
  %r = rcgmp.match %n : !rcgmp.mpz -> i64 {
    at_least 1 (%k) {
      rcgmp.release %k : !rcgmp.mpz
      rcgmp.yield %a : i64
    }
    default (%m) {
      rcgmp.release %m : !rcgmp.mpz
      rcgmp.yield %a : i64
    }
  }
  return %r : i64
}

// A constant scrutinee selects its arm, whose argument is a constant.
// CHECK-LABEL: func.func @constant
// CHECK-NOT: rcgmp.match
// CHECK: %[[K:.*]] = rcgmp.constant 340282366920938463463374607431768211452 : !rcgmp.mpz
// CHECK: call @use(%[[K]])
func.func @constant() -> i64 {
  %c = rcgmp.constant 340282366920938463463374607431768211457 : !rcgmp.mpz
  %r = rcgmp.match %c : !rcgmp.mpz -> i64 {
    case 0 {
      %z = arith.constant 0 : i64
      rcgmp.yield %z : i64
    }
    at_least 5 (%k) {
      %u = func.call @use(%k) : (!rcgmp.mpz) -> i64
      rcgmp.yield %u : i64
    }
  }
  return %r : i64
}
