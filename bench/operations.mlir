// Entry points timed by operations.cpp.
func.func @addi(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.addi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @subi(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.subi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @monusi(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.monusi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
// The same result from a comparison and an exact subtraction.
func.func @guarded_subi(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %le = rcgmp.cmpi sle, %x, %y : !rcgmp.mpz
  %r = scf.if %le -> !rcgmp.mpz {
    rcgmp.release %x : !rcgmp.mpz
    rcgmp.release %y : !rcgmp.mpz
    %zero = rcgmp.constant 0 : !rcgmp.mpz
    scf.yield %zero : !rcgmp.mpz
  } else {
    %d = rcgmp.subi %x, %y : !rcgmp.mpz
    scf.yield %d : !rcgmp.mpz
  }
  return %r : !rcgmp.mpz
}
func.func @muli(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.muli %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @andi(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.andi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @ori(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.ori %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @xori(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.xori %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @noti(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.noti %x : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @negi(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.negi %x : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @absi(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.absi %x : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @cmpi(%x: !rcgmp.mpz {rcgmp.borrowed}, %y: !rcgmp.mpz {rcgmp.borrowed}) -> i32 {
  %r = rcgmp.cmpi slt, %x, %y : !rcgmp.mpz
  %e = arith.extui %r : i1 to i32
  return %e : i32
}
func.func @extsi(%v: i64) -> !rcgmp.mpz {
  %r = rcgmp.extsi %v : i64 to !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @retain(%x: !rcgmp.mpz {rcgmp.borrowed}) -> !rcgmp.mpz {
  rcgmp.retain %x : !rcgmp.mpz
  return %x : !rcgmp.mpz
}
func.func @release(%x: !rcgmp.mpz) {
  rcgmp.release %x : !rcgmp.mpz
  return
}
func.func @addi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.addi %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @subi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.subi %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @monusi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.monusi %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @muli_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.muli %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @andi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.andi %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @ori_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.ori %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @xori_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.xori %x, %y : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @noti_atomic(%x: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.noti %x : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @negi_atomic(%x: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.negi %x : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @absi_atomic(%x: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.absi %x : !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @cmpi_atomic(%x: !rcgmp.mpz<atomic> {rcgmp.borrowed}, %y: !rcgmp.mpz<atomic> {rcgmp.borrowed}) -> i32 {
  %r = rcgmp.cmpi slt, %x, %y : !rcgmp.mpz<atomic>
  %e = arith.extui %r : i1 to i32
  return %e : i32
}
func.func @extsi_atomic(%v: i64) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.extsi %v : i64 to !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @retain_atomic(%x: !rcgmp.mpz<atomic> {rcgmp.borrowed}) -> !rcgmp.mpz<atomic> {
  rcgmp.retain %x : !rcgmp.mpz<atomic>
  return %x : !rcgmp.mpz<atomic>
}
func.func @release_atomic(%x: !rcgmp.mpz<atomic>) {
  rcgmp.release %x : !rcgmp.mpz<atomic>
  return
}
func.func @successor(%n: !rcgmp.mpz) -> i64 {
  %r = rcgmp.match %n : !rcgmp.mpz -> i64 {
    case 0, 1 {
      %one = arith.constant 1 : i64
      rcgmp.yield %one : i64
    }
    at_least 2 (%k) {
      %bound = rcgmp.constant 1000 : !rcgmp.mpz
      %above = rcgmp.cmpi sgt, %k, %bound : !rcgmp.mpz
      rcgmp.release %k : !rcgmp.mpz
      %seven = arith.constant 7 : i64
      %three = arith.constant 3 : i64
      %v = arith.select %above, %seven, %three : i64
      rcgmp.yield %v : i64
    }
  }
  return %r : i64
}
func.func @int_cases(%n: !rcgmp.mpz) -> i64 {
  %r = rcgmp.match %n : !rcgmp.mpz -> i64 {
    at_least 0 (%k) {
      %bound = rcgmp.constant 1000 : !rcgmp.mpz
      %above = rcgmp.cmpi sgt, %k, %bound : !rcgmp.mpz
      rcgmp.release %k : !rcgmp.mpz
      %hi = arith.constant 7 : i64
      %lo = arith.constant 5 : i64
      %v = arith.select %above, %hi, %lo : i64
      rcgmp.yield %v : i64
    }
    at_most -1 (%k) {
      %bound = rcgmp.constant 1000 : !rcgmp.mpz
      %above = rcgmp.cmpi sgt, %k, %bound : !rcgmp.mpz
      rcgmp.release %k : !rcgmp.mpz
      %hi = arith.constant 5 : i64
      %lo = arith.constant 3 : i64
      %v = arith.select %above, %hi, %lo : i64
      rcgmp.yield %v : i64
    }
  }
  return %r : i64
}
func.func @int_cases_noti(%n: !rcgmp.mpz) -> i64 {
  %r = rcgmp.match %n : !rcgmp.mpz -> i64 {
    at_least 0 (%k) {
      %bound = rcgmp.constant 1000 : !rcgmp.mpz
      %above = rcgmp.cmpi sgt, %k, %bound : !rcgmp.mpz
      rcgmp.release %k : !rcgmp.mpz
      %hi = arith.constant 7 : i64
      %lo = arith.constant 5 : i64
      %v = arith.select %above, %hi, %lo : i64
      rcgmp.yield %v : i64
    }
    default (%m) {
      %k = rcgmp.noti %m : !rcgmp.mpz
      %bound = rcgmp.constant 1000 : !rcgmp.mpz
      %above = rcgmp.cmpi sgt, %k, %bound : !rcgmp.mpz
      rcgmp.release %k : !rcgmp.mpz
      %hi = arith.constant 5 : i64
      %lo = arith.constant 3 : i64
      %v = arith.select %above, %hi, %lo : i64
      rcgmp.yield %v : i64
    }
  }
  return %r : i64
}
func.func @shli(%x: !rcgmp.mpz, %n: i64) -> !rcgmp.mpz {
  %r = rcgmp.shli %x, %n : !rcgmp.mpz, i64
  return %r : !rcgmp.mpz
}
func.func @shrsi(%x: !rcgmp.mpz, %n: i64) -> !rcgmp.mpz {
  %r = rcgmp.shrsi %x, %n : !rcgmp.mpz, i64
  return %r : !rcgmp.mpz
}
func.func @shli_atomic(%x: !rcgmp.mpz<atomic>, %n: i64) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.shli %x, %n : !rcgmp.mpz<atomic>, i64
  return %r : !rcgmp.mpz<atomic>
}
func.func @shrsi_atomic(%x: !rcgmp.mpz<atomic>, %n: i64) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.shrsi %x, %n : !rcgmp.mpz<atomic>, i64
  return %r : !rcgmp.mpz<atomic>
}
func.func @floormod2expi(%x: !rcgmp.mpz, %n: i64) -> !rcgmp.mpz {
  %r = rcgmp.floormod2expi %x, %n : !rcgmp.mpz, i64
  return %r : !rcgmp.mpz
}
func.func @floormod2expi_atomic(%x: !rcgmp.mpz<atomic>, %n: i64) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.floormod2expi %x, %n : !rcgmp.mpz<atomic>, i64
  return %r : !rcgmp.mpz<atomic>
}
func.func @floormod2expi_32(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %n = arith.constant 32 : i64
  %r = rcgmp.floormod2expi %x, %n : !rcgmp.mpz, i64
  return %r : !rcgmp.mpz
}
func.func @floormod2expi_64(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %n = arith.constant 64 : i64
  %r = rcgmp.floormod2expi %x, %n : !rcgmp.mpz, i64
  return %r : !rcgmp.mpz
}
// The forms that compute the remainder without floormod2expi, which the
// canonicalizer would replace by it: a mask of 2^n - 1, and the Euclidean
// remainder by 2^n, for a variable and a constant n.
func.func @andi_mask(%x: !rcgmp.mpz, %n: i64) -> !rcgmp.mpz {
  %one = rcgmp.constant 1 : !rcgmp.mpz
  %power = rcgmp.shli %one, %n : !rcgmp.mpz, i64
  %mask = rcgmp.subi %power, %one : !rcgmp.mpz
  %r = rcgmp.andi %x, %mask : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @emodsi_power(%x: !rcgmp.mpz, %n: i64) -> !rcgmp.mpz {
  %one = rcgmp.constant 1 : !rcgmp.mpz
  %power = rcgmp.shli %one, %n : !rcgmp.mpz, i64
  %r = rcgmp.emodsi %x, %power : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @andi_32(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %mask = rcgmp.constant 4294967295 : !rcgmp.mpz
  %r = rcgmp.andi %x, %mask : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @andi_64(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %mask = rcgmp.constant 18446744073709551615 : !rcgmp.mpz
  %r = rcgmp.andi %x, %mask : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @emodsi_32(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %power = rcgmp.constant 4294967296 : !rcgmp.mpz
  %r = rcgmp.emodsi %x, %power : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @emodsi_64(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %power = rcgmp.constant 18446744073709551616 : !rcgmp.mpz
  %r = rcgmp.emodsi %x, %power : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @ipowi(%x: !rcgmp.mpz, %n: i64) -> !rcgmp.mpz {
  %r = rcgmp.ipowi %x, %n : !rcgmp.mpz, i64
  return %r : !rcgmp.mpz
}
func.func @ipowi_atomic(%x: !rcgmp.mpz<atomic>, %n: i64) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.ipowi %x, %n : !rcgmp.mpz<atomic>, i64
  return %r : !rcgmp.mpz<atomic>
}
func.func @fifth_ipowi(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %n = arith.constant 5 : i64
  %r = rcgmp.ipowi %x, %n : !rcgmp.mpz, i64
  return %r : !rcgmp.mpz
}
// A power of two, which canonicalization would turn into the shift below.
func.func @ipowi_two(%n: i64) -> !rcgmp.mpz {
  %two = rcgmp.constant 2 : !rcgmp.mpz
  %r = rcgmp.ipowi %two, %n : !rcgmp.mpz, i64
  return %r : !rcgmp.mpz
}
// The forms that compute powers without ipowi: a shift of one for powers of
// two, products for a fixed exponent, and a loop that squares and multiplies.
func.func @shli_one(%n: i64) -> !rcgmp.mpz {
  %one = rcgmp.constant 1 : !rcgmp.mpz
  %r = rcgmp.shli %one, %n : !rcgmp.mpz, i64
  return %r : !rcgmp.mpz
}
func.func @square_muli(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  rcgmp.retain %x : !rcgmp.mpz
  %r = rcgmp.muli %x, %x : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @fifth_muli(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  rcgmp.retain %x : !rcgmp.mpz
  rcgmp.retain %x : !rcgmp.mpz
  %x2 = rcgmp.muli %x, %x : !rcgmp.mpz
  rcgmp.retain %x2 : !rcgmp.mpz
  %x4 = rcgmp.muli %x2, %x2 : !rcgmp.mpz
  %r = rcgmp.muli %x4, %x : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @power_loop(%x: !rcgmp.mpz, %n: i64) -> !rcgmp.mpz {
  %one = rcgmp.constant 1 : !rcgmp.mpz
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %r:3 = scf.while (%acc = %one, %base = %x, %k = %n) : (!rcgmp.mpz, !rcgmp.mpz, i64) -> (!rcgmp.mpz, !rcgmp.mpz, i64) {
    %more = arith.cmpi ne, %k, %c0 : i64
    scf.condition(%more) %acc, %base, %k : !rcgmp.mpz, !rcgmp.mpz, i64
  } do {
  ^bb0(%acc: !rcgmp.mpz, %base: !rcgmp.mpz, %k: i64):
    %bit = arith.andi %k, %c1 : i64
    %odd = arith.cmpi ne, %bit, %c0 : i64
    %product = scf.if %odd -> !rcgmp.mpz {
      rcgmp.retain %base : !rcgmp.mpz
      %p = rcgmp.muli %acc, %base : !rcgmp.mpz
      scf.yield %p : !rcgmp.mpz
    } else {
      scf.yield %acc : !rcgmp.mpz
    }
    %rest = arith.shrui %k, %c1 : i64
    %done = arith.cmpi eq, %rest, %c0 : i64
    %square = scf.if %done -> !rcgmp.mpz {
      scf.yield %base : !rcgmp.mpz
    } else {
      rcgmp.retain %base : !rcgmp.mpz
      %s = rcgmp.muli %base, %base : !rcgmp.mpz
      scf.yield %s : !rcgmp.mpz
    }
    scf.yield %product, %square, %rest : !rcgmp.mpz, !rcgmp.mpz, i64
  }
  rcgmp.release %r#1 : !rcgmp.mpz
  return %r#0 : !rcgmp.mpz
}
func.func @divsi(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.divsi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @remsi(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.remsi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @floordivsi(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.floordivsi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @floormodsi(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.floormodsi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @edivsi(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.edivsi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @emodsi(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.emodsi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
// Euclidean division and remainder composed from floored ones: the quotient
// by a negative divisor is the negated quotient by its magnitude, and the
// remainder is the floored remainder by the magnitude.
func.func @composed_edivsi(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %zero = rcgmp.constant 0 : !rcgmp.mpz
  %positive = rcgmp.cmpi sgt, %y, %zero : !rcgmp.mpz
  %r = scf.if %positive -> !rcgmp.mpz {
    %q = rcgmp.floordivsi %x, %y : !rcgmp.mpz
    scf.yield %q : !rcgmp.mpz
  } else {
    %m = rcgmp.negi %y : !rcgmp.mpz
    %q = rcgmp.floordivsi %x, %m : !rcgmp.mpz
    %n = rcgmp.negi %q : !rcgmp.mpz
    scf.yield %n : !rcgmp.mpz
  }
  return %r : !rcgmp.mpz
}
func.func @composed_emodsi(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %m = rcgmp.absi %y : !rcgmp.mpz
  %r = rcgmp.floormodsi %x, %m : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
// Lean division, next to the same result from a comparison and a division
// whose zero divisor is undefined.
func.func @divsi_lean(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.divsi %x, %y divzero<lean> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @guarded_divsi(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %zero = rcgmp.constant 0 : !rcgmp.mpz
  %by_zero = rcgmp.cmpi eq, %y, %zero : !rcgmp.mpz
  %r = scf.if %by_zero -> !rcgmp.mpz {
    rcgmp.release %y : !rcgmp.mpz
    rcgmp.release %x : !rcgmp.mpz
    %q = rcgmp.constant 0 : !rcgmp.mpz
    scf.yield %q : !rcgmp.mpz
  } else {
    %d = rcgmp.divsi %x, %y divzero<undefined> : !rcgmp.mpz
    scf.yield %d : !rcgmp.mpz
  }
  return %r : !rcgmp.mpz
}
func.func @remsi_lean(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.remsi %x, %y divzero<lean> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @guarded_remsi(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %zero = rcgmp.constant 0 : !rcgmp.mpz
  %by_zero = rcgmp.cmpi eq, %y, %zero : !rcgmp.mpz
  %r = scf.if %by_zero -> !rcgmp.mpz {
    rcgmp.release %y : !rcgmp.mpz
    scf.yield %x : !rcgmp.mpz
  } else {
    %d = rcgmp.remsi %x, %y divzero<undefined> : !rcgmp.mpz
    scf.yield %d : !rcgmp.mpz
  }
  return %r : !rcgmp.mpz
}
func.func @edivsi_lean(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.edivsi %x, %y divzero<lean> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @guarded_edivsi(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %zero = rcgmp.constant 0 : !rcgmp.mpz
  %by_zero = rcgmp.cmpi eq, %y, %zero : !rcgmp.mpz
  %r = scf.if %by_zero -> !rcgmp.mpz {
    rcgmp.release %y : !rcgmp.mpz
    rcgmp.release %x : !rcgmp.mpz
    %q = rcgmp.constant 0 : !rcgmp.mpz
    scf.yield %q : !rcgmp.mpz
  } else {
    %d = rcgmp.edivsi %x, %y divzero<undefined> : !rcgmp.mpz
    scf.yield %d : !rcgmp.mpz
  }
  return %r : !rcgmp.mpz
}
func.func @emodsi_lean(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.emodsi %x, %y divzero<lean> : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @guarded_emodsi(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %zero = rcgmp.constant 0 : !rcgmp.mpz
  %by_zero = rcgmp.cmpi eq, %y, %zero : !rcgmp.mpz
  %r = scf.if %by_zero -> !rcgmp.mpz {
    rcgmp.release %y : !rcgmp.mpz
    scf.yield %x : !rcgmp.mpz
  } else {
    %d = rcgmp.emodsi %x, %y divzero<undefined> : !rcgmp.mpz
    scf.yield %d : !rcgmp.mpz
  }
  return %r : !rcgmp.mpz
}
func.func @divremsi(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> (!rcgmp.mpz, !rcgmp.mpz) {
  %q, %r = rcgmp.divremsi %x, %y : !rcgmp.mpz
  return %q, %r : !rcgmp.mpz, !rcgmp.mpz
}
func.func @trunci(%x: !rcgmp.mpz {rcgmp.borrowed}) -> i64 {
  %v = rcgmp.trunci %x : !rcgmp.mpz to i64
  return %v : i64
}
func.func @trunci_overflow(%x: !rcgmp.mpz {rcgmp.borrowed}) -> i64 {
  %v, %o = rcgmp.trunci_overflow %x : !rcgmp.mpz to i64
  %e = arith.extui %o : i1 to i64
  %r = arith.xori %v, %e : i64
  return %r : i64
}
func.func @log2i(%x: !rcgmp.mpz {rcgmp.borrowed}) -> i64 {
  %r = rcgmp.log2i %x : !rcgmp.mpz to i64
  return %r : i64
}
// The same logarithm of a nonnegative integer without rcgmp.log2i: shift a
// copy right by a word until it fits one, and count the leading zeros of that
// word.
func.func @shifted_trunci_log2(%x: !rcgmp.mpz {rcgmp.borrowed}) -> i64 {
  %zero = arith.constant 0 : i64
  %one = arith.constant 1 : i64
  %top = arith.constant 63 : i64
  %word = arith.constant 64 : i64
  %v, %o = rcgmp.trunci_overflow %x unsigned : !rcgmp.mpz to i64
  %r:2 = scf.if %o -> (i64, i64) {
    rcgmp.retain %x : !rcgmp.mpz
    %first = rcgmp.shrsi %x, %word : !rcgmp.mpz, i64
    %w:3 = scf.while (%y = %first, %k = %word) : (!rcgmp.mpz, i64) -> (!rcgmp.mpz, i64, i64) {
      %low, %wide = rcgmp.trunci_overflow %y unsigned : !rcgmp.mpz to i64
      scf.condition(%wide) %y, %k, %low : !rcgmp.mpz, i64, i64
    } do {
    ^bb0(%y: !rcgmp.mpz, %k: i64, %low: i64):
      %z = rcgmp.shrsi %y, %word : !rcgmp.mpz, i64
      %next = arith.addi %k, %word : i64
      scf.yield %z, %next : !rcgmp.mpz, i64
    }
    rcgmp.release %w#0 : !rcgmp.mpz
    scf.yield %w#2, %w#1 : i64, i64
  } else {
    scf.yield %v, %zero : i64, i64
  }
  %nonzero = arith.ori %r#0, %one : i64
  %zeros = math.ctlz %nonzero : i64
  %low = arith.subi %top, %zeros : i64
  %log = arith.addi %r#1, %low : i64
  return %log : i64
}
func.func @gcdi(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.gcdi %x, %y : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
// The same result by Euclid's algorithm, from remainders.
func.func @euclid(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %a = rcgmp.absi %x : !rcgmp.mpz
  %b = rcgmp.absi %y : !rcgmp.mpz
  %r:2 = scf.while (%u = %a, %v = %b) : (!rcgmp.mpz, !rcgmp.mpz) -> (!rcgmp.mpz, !rcgmp.mpz) {
    %zero = rcgmp.constant 0 : !rcgmp.mpz
    %more = rcgmp.cmpi ne, %v, %zero : !rcgmp.mpz
    scf.condition(%more) %u, %v : !rcgmp.mpz, !rcgmp.mpz
  } do {
  ^bb0(%u: !rcgmp.mpz, %v: !rcgmp.mpz):
    rcgmp.retain %v : !rcgmp.mpz
    %m = rcgmp.remsi %u, %v divzero<undefined> : !rcgmp.mpz
    scf.yield %v, %m : !rcgmp.mpz, !rcgmp.mpz
  }
  rcgmp.release %r#1 : !rcgmp.mpz
  return %r#0 : !rcgmp.mpz
}
// Exact division, next to plain division of the same multiples, by a variable
// divisor and by constants.
func.func @divsi_exact(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> !rcgmp.mpz {
  %r = rcgmp.divsi %x, %y exact : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @divsi_exact_by_12(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %c = rcgmp.constant 12 : !rcgmp.mpz
  %r = rcgmp.divsi %x, %c exact : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @divsi_by_12(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %c = rcgmp.constant 12 : !rcgmp.mpz
  %r = rcgmp.divsi %x, %c : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @divsi_exact_by_8(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %c = rcgmp.constant 8 : !rcgmp.mpz
  %r = rcgmp.divsi %x, %c exact : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @divsi_by_8(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %c = rcgmp.constant 8 : !rcgmp.mpz
  %r = rcgmp.divsi %x, %c : !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @testbiti(%x: !rcgmp.mpz {rcgmp.borrowed}, %n: i64) -> i32 {
  %r = rcgmp.testbiti %x, %n : !rcgmp.mpz, i64
  %e = arith.extui %r : i1 to i32
  return %e : i32
}
// The same bit without rcgmp.testbiti, as Lean's Nat.testBit computes it:
// shift a reference to the integer right, mask the lowest bit, and compare it
// with zero.
func.func @shifted_testbit(%x: !rcgmp.mpz {rcgmp.borrowed}, %n: i64) -> i32 {
  %one = rcgmp.constant 1 : !rcgmp.mpz
  %zero = rcgmp.constant 0 : !rcgmp.mpz
  rcgmp.retain %x : !rcgmp.mpz
  %s = rcgmp.shrsi %x, %n : !rcgmp.mpz, i64
  %b = rcgmp.andi %s, %one : !rcgmp.mpz
  %r = rcgmp.cmpi ne, %b, %zero : !rcgmp.mpz
  rcgmp.release %b : !rcgmp.mpz
  %e = arith.extui %r : i1 to i32
  return %e : i32
}
