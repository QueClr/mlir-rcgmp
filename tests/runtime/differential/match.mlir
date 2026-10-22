// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts -o /dev/null
// Matches compared against GMP by match.cpp. Each returns the index of the
// selected arm and its region argument, or zero for an arm without one.
func.func @match_nat_local(%n: !rcgmp.mpz) -> (i64, !rcgmp.mpz) {
  %r:2 = rcgmp.match %n : !rcgmp.mpz -> (i64, !rcgmp.mpz) {
    case 0 {
      %i = arith.constant 0 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    case 1 {
      %i = arith.constant 1 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    at_least 2 (%x) {
      %i = arith.constant 2 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
  }
  return %r#0, %r#1 : i64, !rcgmp.mpz
}
func.func @match_nat_atomic(%n: !rcgmp.mpz<atomic>) -> (i64, !rcgmp.mpz<atomic>) {
  %r:2 = rcgmp.match %n : !rcgmp.mpz<atomic> -> (i64, !rcgmp.mpz<atomic>) {
    case 0 {
      %i = arith.constant 0 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz<atomic>
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    case 1 {
      %i = arith.constant 1 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz<atomic>
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    at_least 2 (%x) {
      %i = arith.constant 2 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
  }
  return %r#0, %r#1 : i64, !rcgmp.mpz<atomic>
}
func.func @match_window_local(%n: !rcgmp.mpz) -> (i64, !rcgmp.mpz) {
  %r:2 = rcgmp.match %n : !rcgmp.mpz -> (i64, !rcgmp.mpz) {
    case 0 {
      %i = arith.constant 0 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    case 1, 3 {
      %i = arith.constant 1 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    case 2 {
      %i = arith.constant 2 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    case 5 {
      %i = arith.constant 3 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    at_least 7 (%x) {
      %i = arith.constant 4 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    default (%x) {
      %i = arith.constant 5 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
  }
  return %r#0, %r#1 : i64, !rcgmp.mpz
}
func.func @match_window_atomic(%n: !rcgmp.mpz<atomic>) -> (i64, !rcgmp.mpz<atomic>) {
  %r:2 = rcgmp.match %n : !rcgmp.mpz<atomic> -> (i64, !rcgmp.mpz<atomic>) {
    case 0 {
      %i = arith.constant 0 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz<atomic>
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    case 1, 3 {
      %i = arith.constant 1 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz<atomic>
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    case 2 {
      %i = arith.constant 2 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz<atomic>
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    case 5 {
      %i = arith.constant 3 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz<atomic>
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    at_least 7 (%x) {
      %i = arith.constant 4 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    default (%x) {
      %i = arith.constant 5 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
  }
  return %r#0, %r#1 : i64, !rcgmp.mpz<atomic>
}
func.func @match_sparse_local(%n: !rcgmp.mpz) -> (i64, !rcgmp.mpz) {
  %r:2 = rcgmp.match %n : !rcgmp.mpz -> (i64, !rcgmp.mpz) {
    case -1000 {
      %i = arith.constant 0 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    case 0 {
      %i = arith.constant 1 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    case 100 {
      %i = arith.constant 2 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    case 1000000 {
      %i = arith.constant 3 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    at_least 1099511627776 (%x) {
      %i = arith.constant 4 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    default (%x) {
      %i = arith.constant 5 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
  }
  return %r#0, %r#1 : i64, !rcgmp.mpz
}
func.func @match_sparse_atomic(%n: !rcgmp.mpz<atomic>) -> (i64, !rcgmp.mpz<atomic>) {
  %r:2 = rcgmp.match %n : !rcgmp.mpz<atomic> -> (i64, !rcgmp.mpz<atomic>) {
    case -1000 {
      %i = arith.constant 0 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz<atomic>
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    case 0 {
      %i = arith.constant 1 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz<atomic>
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    case 100 {
      %i = arith.constant 2 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz<atomic>
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    case 1000000 {
      %i = arith.constant 3 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz<atomic>
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    at_least 1099511627776 (%x) {
      %i = arith.constant 4 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    default (%x) {
      %i = arith.constant 5 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
  }
  return %r#0, %r#1 : i64, !rcgmp.mpz<atomic>
}
func.func @match_points_local(%n: !rcgmp.mpz) -> (i64, !rcgmp.mpz) {
  %r:2 = rcgmp.match %n : !rcgmp.mpz -> (i64, !rcgmp.mpz) {
    case 3 {
      %i = arith.constant 0 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    case 9 {
      %i = arith.constant 1 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    default (%x) {
      %i = arith.constant 2 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
  }
  return %r#0, %r#1 : i64, !rcgmp.mpz
}
func.func @match_points_atomic(%n: !rcgmp.mpz<atomic>) -> (i64, !rcgmp.mpz<atomic>) {
  %r:2 = rcgmp.match %n : !rcgmp.mpz<atomic> -> (i64, !rcgmp.mpz<atomic>) {
    case 3 {
      %i = arith.constant 0 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz<atomic>
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    case 9 {
      %i = arith.constant 1 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz<atomic>
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    default (%x) {
      %i = arith.constant 2 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
  }
  return %r#0, %r#1 : i64, !rcgmp.mpz<atomic>
}
func.func @match_negative_local(%n: !rcgmp.mpz) -> (i64, !rcgmp.mpz) {
  %r:2 = rcgmp.match %n : !rcgmp.mpz -> (i64, !rcgmp.mpz) {
    case -7 {
      %i = arith.constant 0 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    at_least -5 (%x) {
      %i = arith.constant 1 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    default (%x) {
      %i = arith.constant 2 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
  }
  return %r#0, %r#1 : i64, !rcgmp.mpz
}
func.func @match_negative_atomic(%n: !rcgmp.mpz<atomic>) -> (i64, !rcgmp.mpz<atomic>) {
  %r:2 = rcgmp.match %n : !rcgmp.mpz<atomic> -> (i64, !rcgmp.mpz<atomic>) {
    case -7 {
      %i = arith.constant 0 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz<atomic>
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    at_least -5 (%x) {
      %i = arith.constant 1 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    default (%x) {
      %i = arith.constant 2 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
  }
  return %r#0, %r#1 : i64, !rcgmp.mpz<atomic>
}
func.func @match_edges_local(%n: !rcgmp.mpz) -> (i64, !rcgmp.mpz) {
  %r:2 = rcgmp.match %n : !rcgmp.mpz -> (i64, !rcgmp.mpz) {
    case 2305843009213693951 {
      %i = arith.constant 0 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    case -2305843009213693952 {
      %i = arith.constant 1 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    at_least 2305843009213693952 (%x) {
      %i = arith.constant 2 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    case -2305843009213693953 {
      %i = arith.constant 3 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    default (%x) {
      %i = arith.constant 4 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
  }
  return %r#0, %r#1 : i64, !rcgmp.mpz
}
func.func @match_edges_atomic(%n: !rcgmp.mpz<atomic>) -> (i64, !rcgmp.mpz<atomic>) {
  %r:2 = rcgmp.match %n : !rcgmp.mpz<atomic> -> (i64, !rcgmp.mpz<atomic>) {
    case 2305843009213693951 {
      %i = arith.constant 0 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz<atomic>
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    case -2305843009213693952 {
      %i = arith.constant 1 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz<atomic>
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    at_least 2305843009213693952 (%x) {
      %i = arith.constant 2 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    case -2305843009213693953 {
      %i = arith.constant 3 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz<atomic>
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    default (%x) {
      %i = arith.constant 4 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
  }
  return %r#0, %r#1 : i64, !rcgmp.mpz<atomic>
}
func.func @match_limbs_local(%n: !rcgmp.mpz) -> (i64, !rcgmp.mpz) {
  %r:2 = rcgmp.match %n : !rcgmp.mpz -> (i64, !rcgmp.mpz) {
    case 18446744073709551616, 18446744073709551617 {
      %i = arith.constant 0 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    case 36893488147419103232 {
      %i = arith.constant 1 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    case 55340232221128654853 {
      %i = arith.constant 2 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    case -340282366920938463463374607431768211461 {
      %i = arith.constant 3 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    at_least 340282366920938463463374607431768211457 (%x) {
      %i = arith.constant 4 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    at_least 18446744073709551619 {
      %i = arith.constant 5 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    default (%x) {
      %i = arith.constant 6 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
  }
  return %r#0, %r#1 : i64, !rcgmp.mpz
}
func.func @match_limbs_atomic(%n: !rcgmp.mpz<atomic>) -> (i64, !rcgmp.mpz<atomic>) {
  %r:2 = rcgmp.match %n : !rcgmp.mpz<atomic> -> (i64, !rcgmp.mpz<atomic>) {
    case 18446744073709551616, 18446744073709551617 {
      %i = arith.constant 0 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz<atomic>
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    case 36893488147419103232 {
      %i = arith.constant 1 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz<atomic>
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    case 55340232221128654853 {
      %i = arith.constant 2 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz<atomic>
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    case -340282366920938463463374607431768211461 {
      %i = arith.constant 3 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz<atomic>
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    at_least 340282366920938463463374607431768211457 (%x) {
      %i = arith.constant 4 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    at_least 18446744073709551619 {
      %i = arith.constant 5 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz<atomic>
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    default (%x) {
      %i = arith.constant 6 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
  }
  return %r#0, %r#1 : i64, !rcgmp.mpz<atomic>
}
func.func @match_run_local(%n: !rcgmp.mpz) -> (i64, !rcgmp.mpz) {
  %r:2 = rcgmp.match %n : !rcgmp.mpz -> (i64, !rcgmp.mpz) {
    case 2582249878086908589655919172003011874329705792829223512830659356540647622016841194629645353280137831435903171972747493383 {
      %i = arith.constant 0 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    case 2582249878086908589655919172003011874329705792829223512830659356540647622016841194629645353280137831435903171972747493385 {
      %i = arith.constant 1 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    case -2582249878086908589655919172003011874329705792829223512830659356540647622016841194629645353280137831435903171972747493383 {
      %i = arith.constant 2 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    at_least 4149515568880992958512407863691161151012446232242436899995657329690652811412908146399707048947103794288197886611300789182395151075411775307886874834113963687061181803401509523685376 (%x) {
      %i = arith.constant 3 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    default {
      %i = arith.constant 4 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
  }
  return %r#0, %r#1 : i64, !rcgmp.mpz
}
func.func @match_run_atomic(%n: !rcgmp.mpz<atomic>) -> (i64, !rcgmp.mpz<atomic>) {
  %r:2 = rcgmp.match %n : !rcgmp.mpz<atomic> -> (i64, !rcgmp.mpz<atomic>) {
    case 2582249878086908589655919172003011874329705792829223512830659356540647622016841194629645353280137831435903171972747493383 {
      %i = arith.constant 0 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz<atomic>
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    case 2582249878086908589655919172003011874329705792829223512830659356540647622016841194629645353280137831435903171972747493385 {
      %i = arith.constant 1 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz<atomic>
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    case -2582249878086908589655919172003011874329705792829223512830659356540647622016841194629645353280137831435903171972747493383 {
      %i = arith.constant 2 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz<atomic>
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    at_least 4149515568880992958512407863691161151012446232242436899995657329690652811412908146399707048947103794288197886611300789182395151075411775307886874834113963687061181803401509523685376 (%x) {
      %i = arith.constant 3 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    default {
      %i = arith.constant 4 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz<atomic>
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
  }
  return %r#0, %r#1 : i64, !rcgmp.mpz<atomic>
}
func.func @match_shadow_local(%n: !rcgmp.mpz) -> (i64, !rcgmp.mpz) {
  %r:2 = rcgmp.match %n : !rcgmp.mpz -> (i64, !rcgmp.mpz) {
    at_least 10 (%x) {
      %i = arith.constant 0 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    case 12, 3 {
      %i = arith.constant 1 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    at_least 5 (%x) {
      %i = arith.constant 2 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
    case -3 {
      %i = arith.constant 3 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz
      rcgmp.yield %i, %x : i64, !rcgmp.mpz
    }
  }
  return %r#0, %r#1 : i64, !rcgmp.mpz
}
func.func @match_shadow_atomic(%n: !rcgmp.mpz<atomic>) -> (i64, !rcgmp.mpz<atomic>) {
  %r:2 = rcgmp.match %n : !rcgmp.mpz<atomic> -> (i64, !rcgmp.mpz<atomic>) {
    at_least 10 (%x) {
      %i = arith.constant 0 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    case 12, 3 {
      %i = arith.constant 1 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz<atomic>
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    at_least 5 (%x) {
      %i = arith.constant 2 : i64
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
    case -3 {
      %i = arith.constant 3 : i64
      %x = rcgmp.constant 0 : !rcgmp.mpz<atomic>
      rcgmp.yield %i, %x : i64, !rcgmp.mpz<atomic>
    }
  }
  return %r#0, %r#1 : i64, !rcgmp.mpz<atomic>
}
func.func @index_table_local(%n: !rcgmp.mpz) -> i64 {
  %r = rcgmp.match %n : !rcgmp.mpz -> i64 {
    case 0 {
      %i = arith.constant 0 : i64
      rcgmp.yield %i : i64
    }
    case 1 {
      %i = arith.constant 1 : i64
      rcgmp.yield %i : i64
    }
    case 2 {
      %i = arith.constant 2 : i64
      rcgmp.yield %i : i64
    }
    case 3 {
      %i = arith.constant 3 : i64
      rcgmp.yield %i : i64
    }
    case 4 {
      %i = arith.constant 4 : i64
      rcgmp.yield %i : i64
    }
    case 5 {
      %i = arith.constant 5 : i64
      rcgmp.yield %i : i64
    }
    case 6 {
      %i = arith.constant 6 : i64
      rcgmp.yield %i : i64
    }
    case 7 {
      %i = arith.constant 7 : i64
      rcgmp.yield %i : i64
    }
    case 8 {
      %i = arith.constant 8 : i64
      rcgmp.yield %i : i64
    }
    case 9 {
      %i = arith.constant 9 : i64
      rcgmp.yield %i : i64
    }
    at_least 10 (%x) {
      rcgmp.release %x : !rcgmp.mpz
      %i = arith.constant 10 : i64
      rcgmp.yield %i : i64
    }
    default {
      %i = arith.constant 11 : i64
      rcgmp.yield %i : i64
    }
  }
  return %r : i64
}
func.func @index_table_atomic(%n: !rcgmp.mpz<atomic>) -> i64 {
  %r = rcgmp.match %n : !rcgmp.mpz<atomic> -> i64 {
    case 0 {
      %i = arith.constant 0 : i64
      rcgmp.yield %i : i64
    }
    case 1 {
      %i = arith.constant 1 : i64
      rcgmp.yield %i : i64
    }
    case 2 {
      %i = arith.constant 2 : i64
      rcgmp.yield %i : i64
    }
    case 3 {
      %i = arith.constant 3 : i64
      rcgmp.yield %i : i64
    }
    case 4 {
      %i = arith.constant 4 : i64
      rcgmp.yield %i : i64
    }
    case 5 {
      %i = arith.constant 5 : i64
      rcgmp.yield %i : i64
    }
    case 6 {
      %i = arith.constant 6 : i64
      rcgmp.yield %i : i64
    }
    case 7 {
      %i = arith.constant 7 : i64
      rcgmp.yield %i : i64
    }
    case 8 {
      %i = arith.constant 8 : i64
      rcgmp.yield %i : i64
    }
    case 9 {
      %i = arith.constant 9 : i64
      rcgmp.yield %i : i64
    }
    at_least 10 (%x) {
      rcgmp.release %x : !rcgmp.mpz<atomic>
      %i = arith.constant 10 : i64
      rcgmp.yield %i : i64
    }
    default {
      %i = arith.constant 11 : i64
      rcgmp.yield %i : i64
    }
  }
  return %r : i64
}
func.func @index_hashed_local(%n: !rcgmp.mpz) -> i64 {
  %r = rcgmp.match %n : !rcgmp.mpz -> i64 {
    case -1000 {
      %i = arith.constant 0 : i64
      rcgmp.yield %i : i64
    }
    case 0 {
      %i = arith.constant 1 : i64
      rcgmp.yield %i : i64
    }
    case 7 {
      %i = arith.constant 2 : i64
      rcgmp.yield %i : i64
    }
    case 100 {
      %i = arith.constant 3 : i64
      rcgmp.yield %i : i64
    }
    case 1000 {
      %i = arith.constant 4 : i64
      rcgmp.yield %i : i64
    }
    case 4096 {
      %i = arith.constant 5 : i64
      rcgmp.yield %i : i64
    }
    case 20000 {
      %i = arith.constant 6 : i64
      rcgmp.yield %i : i64
    }
    case 32768 {
      %i = arith.constant 7 : i64
      rcgmp.yield %i : i64
    }
    default {
      %i = arith.constant 8 : i64
      rcgmp.yield %i : i64
    }
  }
  return %r : i64
}
func.func @index_hashed_atomic(%n: !rcgmp.mpz<atomic>) -> i64 {
  %r = rcgmp.match %n : !rcgmp.mpz<atomic> -> i64 {
    case -1000 {
      %i = arith.constant 0 : i64
      rcgmp.yield %i : i64
    }
    case 0 {
      %i = arith.constant 1 : i64
      rcgmp.yield %i : i64
    }
    case 7 {
      %i = arith.constant 2 : i64
      rcgmp.yield %i : i64
    }
    case 100 {
      %i = arith.constant 3 : i64
      rcgmp.yield %i : i64
    }
    case 1000 {
      %i = arith.constant 4 : i64
      rcgmp.yield %i : i64
    }
    case 4096 {
      %i = arith.constant 5 : i64
      rcgmp.yield %i : i64
    }
    case 20000 {
      %i = arith.constant 6 : i64
      rcgmp.yield %i : i64
    }
    case 32768 {
      %i = arith.constant 7 : i64
      rcgmp.yield %i : i64
    }
    default {
      %i = arith.constant 8 : i64
      rcgmp.yield %i : i64
    }
  }
  return %r : i64
}
func.func @index_hashed_wide_local(%n: !rcgmp.mpz) -> i64 {
  %r = rcgmp.match %n : !rcgmp.mpz -> i64 {
    case -1000 {
      %i = arith.constant 0 : i64
      rcgmp.yield %i : i64
    }
    case 0 {
      %i = arith.constant 1 : i64
      rcgmp.yield %i : i64
    }
    case 7 {
      %i = arith.constant 2 : i64
      rcgmp.yield %i : i64
    }
    case 65536 {
      %i = arith.constant 3 : i64
      rcgmp.yield %i : i64
    }
    case 2147483648 {
      %i = arith.constant 4 : i64
      rcgmp.yield %i : i64
    }
    case 1099511627776 {
      %i = arith.constant 5 : i64
      rcgmp.yield %i : i64
    }
    case -1125899906842624 {
      %i = arith.constant 6 : i64
      rcgmp.yield %i : i64
    }
    default {
      %i = arith.constant 7 : i64
      rcgmp.yield %i : i64
    }
  }
  return %r : i64
}
func.func @index_hashed_wide_atomic(%n: !rcgmp.mpz<atomic>) -> i64 {
  %r = rcgmp.match %n : !rcgmp.mpz<atomic> -> i64 {
    case -1000 {
      %i = arith.constant 0 : i64
      rcgmp.yield %i : i64
    }
    case 0 {
      %i = arith.constant 1 : i64
      rcgmp.yield %i : i64
    }
    case 7 {
      %i = arith.constant 2 : i64
      rcgmp.yield %i : i64
    }
    case 65536 {
      %i = arith.constant 3 : i64
      rcgmp.yield %i : i64
    }
    case 2147483648 {
      %i = arith.constant 4 : i64
      rcgmp.yield %i : i64
    }
    case 1099511627776 {
      %i = arith.constant 5 : i64
      rcgmp.yield %i : i64
    }
    case -1125899906842624 {
      %i = arith.constant 6 : i64
      rcgmp.yield %i : i64
    }
    default {
      %i = arith.constant 7 : i64
      rcgmp.yield %i : i64
    }
  }
  return %r : i64
}
func.func @index_mixed_local(%n: !rcgmp.mpz) -> i64 {
  %r = rcgmp.match %n : !rcgmp.mpz -> i64 {
    case 0 {
      %i = arith.constant 0 : i64
      rcgmp.yield %i : i64
    }
    case 1, 3 {
      %i = arith.constant 1 : i64
      rcgmp.yield %i : i64
    }
    at_least 5 (%x) {
      rcgmp.release %x : !rcgmp.mpz
      %i = arith.constant 2 : i64
      rcgmp.yield %i : i64
    }
    case 2 {
      %i = arith.constant 3 : i64
      rcgmp.yield %i : i64
    }
    default (%x) {
      rcgmp.release %x : !rcgmp.mpz
      %i = arith.constant 4 : i64
      rcgmp.yield %i : i64
    }
  }
  return %r : i64
}
func.func @index_mixed_atomic(%n: !rcgmp.mpz<atomic>) -> i64 {
  %r = rcgmp.match %n : !rcgmp.mpz<atomic> -> i64 {
    case 0 {
      %i = arith.constant 0 : i64
      rcgmp.yield %i : i64
    }
    case 1, 3 {
      %i = arith.constant 1 : i64
      rcgmp.yield %i : i64
    }
    at_least 5 (%x) {
      rcgmp.release %x : !rcgmp.mpz<atomic>
      %i = arith.constant 2 : i64
      rcgmp.yield %i : i64
    }
    case 2 {
      %i = arith.constant 3 : i64
      rcgmp.yield %i : i64
    }
    default (%x) {
      rcgmp.release %x : !rcgmp.mpz<atomic>
      %i = arith.constant 4 : i64
      rcgmp.yield %i : i64
    }
  }
  return %r : i64
}
