// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts -o /dev/null
// Constants, selected by index, cover both sides of the 64-bit immediate
// bounds and of the signed and unsigned word bounds, and two- and three-limb
// values; other indices yield zero. Differential.cppm lists the same values in
// this order.
func.func @constant_local(%index: index) -> !rcgmp.mpz {
  %r = scf.index_switch %index -> !rcgmp.mpz
  case 0 {
    %c = rcgmp.constant 0 : !rcgmp.mpz
    scf.yield %c : !rcgmp.mpz
  }
  case 1 {
    %c = rcgmp.constant -1 : !rcgmp.mpz
    scf.yield %c : !rcgmp.mpz
  }
  case 2 {
    %c = rcgmp.constant 2305843009213693951 : !rcgmp.mpz
    scf.yield %c : !rcgmp.mpz
  }
  case 3 {
    %c = rcgmp.constant -2305843009213693952 : !rcgmp.mpz
    scf.yield %c : !rcgmp.mpz
  }
  case 4 {
    %c = rcgmp.constant 2305843009213693952 : !rcgmp.mpz
    scf.yield %c : !rcgmp.mpz
  }
  case 5 {
    %c = rcgmp.constant -2305843009213693953 : !rcgmp.mpz
    scf.yield %c : !rcgmp.mpz
  }
  case 6 {
    %c = rcgmp.constant 9223372036854775807 : !rcgmp.mpz
    scf.yield %c : !rcgmp.mpz
  }
  case 7 {
    %c = rcgmp.constant -9223372036854775808 : !rcgmp.mpz
    scf.yield %c : !rcgmp.mpz
  }
  case 8 {
    %c = rcgmp.constant 18446744073709551615 : !rcgmp.mpz
    scf.yield %c : !rcgmp.mpz
  }
  case 9 {
    %c = rcgmp.constant -18446744073709551616 : !rcgmp.mpz
    scf.yield %c : !rcgmp.mpz
  }
  case 10 {
    %c = rcgmp.constant 340282366920938463463374607431768211455 : !rcgmp.mpz
    scf.yield %c : !rcgmp.mpz
  }
  case 11 {
    %c = rcgmp.constant -340282366920938463463374607431768211457 : !rcgmp.mpz
    scf.yield %c : !rcgmp.mpz
  }
  default {
    %c = rcgmp.constant 0 : !rcgmp.mpz
    scf.yield %c : !rcgmp.mpz
  }
  return %r : !rcgmp.mpz
}
func.func @constant_atomic(%index: index) -> !rcgmp.mpz<atomic> {
  %r = scf.index_switch %index -> !rcgmp.mpz<atomic>
  case 0 {
    %c = rcgmp.constant 0 : !rcgmp.mpz<atomic>
    scf.yield %c : !rcgmp.mpz<atomic>
  }
  case 1 {
    %c = rcgmp.constant -1 : !rcgmp.mpz<atomic>
    scf.yield %c : !rcgmp.mpz<atomic>
  }
  case 2 {
    %c = rcgmp.constant 2305843009213693951 : !rcgmp.mpz<atomic>
    scf.yield %c : !rcgmp.mpz<atomic>
  }
  case 3 {
    %c = rcgmp.constant -2305843009213693952 : !rcgmp.mpz<atomic>
    scf.yield %c : !rcgmp.mpz<atomic>
  }
  case 4 {
    %c = rcgmp.constant 2305843009213693952 : !rcgmp.mpz<atomic>
    scf.yield %c : !rcgmp.mpz<atomic>
  }
  case 5 {
    %c = rcgmp.constant -2305843009213693953 : !rcgmp.mpz<atomic>
    scf.yield %c : !rcgmp.mpz<atomic>
  }
  case 6 {
    %c = rcgmp.constant 9223372036854775807 : !rcgmp.mpz<atomic>
    scf.yield %c : !rcgmp.mpz<atomic>
  }
  case 7 {
    %c = rcgmp.constant -9223372036854775808 : !rcgmp.mpz<atomic>
    scf.yield %c : !rcgmp.mpz<atomic>
  }
  case 8 {
    %c = rcgmp.constant 18446744073709551615 : !rcgmp.mpz<atomic>
    scf.yield %c : !rcgmp.mpz<atomic>
  }
  case 9 {
    %c = rcgmp.constant -18446744073709551616 : !rcgmp.mpz<atomic>
    scf.yield %c : !rcgmp.mpz<atomic>
  }
  case 10 {
    %c = rcgmp.constant 340282366920938463463374607431768211455 : !rcgmp.mpz<atomic>
    scf.yield %c : !rcgmp.mpz<atomic>
  }
  case 11 {
    %c = rcgmp.constant -340282366920938463463374607431768211457 : !rcgmp.mpz<atomic>
    scf.yield %c : !rcgmp.mpz<atomic>
  }
  default {
    %c = rcgmp.constant 0 : !rcgmp.mpz<atomic>
    scf.yield %c : !rcgmp.mpz<atomic>
  }
  return %r : !rcgmp.mpz<atomic>
}
