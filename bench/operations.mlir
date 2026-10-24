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
func.func @cmpi(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> i32 {
  %r = rcgmp.cmpi slt, %x, %y : !rcgmp.mpz
  %e = arith.extui %r : i1 to i32
  return %e : i32
}
func.func @extsi(%v: i64) -> !rcgmp.mpz {
  %r = rcgmp.extsi %v : i64 to !rcgmp.mpz
  return %r : !rcgmp.mpz
}
func.func @retain(%x: !rcgmp.mpz) -> !rcgmp.mpz {
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
func.func @cmpi_atomic(%x: !rcgmp.mpz<atomic>, %y: !rcgmp.mpz<atomic>) -> i32 {
  %r = rcgmp.cmpi slt, %x, %y : !rcgmp.mpz<atomic>
  %e = arith.extui %r : i1 to i32
  return %e : i32
}
func.func @extsi_atomic(%v: i64) -> !rcgmp.mpz<atomic> {
  %r = rcgmp.extsi %v : i64 to !rcgmp.mpz<atomic>
  return %r : !rcgmp.mpz<atomic>
}
func.func @retain_atomic(%x: !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic> {
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
func.func @divremsi(%x: !rcgmp.mpz, %y: !rcgmp.mpz) -> (!rcgmp.mpz, !rcgmp.mpz) {
  %q, %r = rcgmp.divremsi %x, %y : !rcgmp.mpz
  return %q, %r : !rcgmp.mpz, !rcgmp.mpz
}
func.func @trunci(%x: !rcgmp.mpz) -> i64 {
  %v = rcgmp.trunci %x : !rcgmp.mpz to i64
  return %v : i64
}
func.func @trunci_overflow(%x: !rcgmp.mpz) -> i64 {
  %v, %o = rcgmp.trunci_overflow %x : !rcgmp.mpz to i64
  %e = arith.extui %o : i1 to i64
  %r = arith.xori %v, %e : i64
  return %r : i64
}
