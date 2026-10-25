// RUN: %rcgmp-opt %s | %rcgmp-opt | %FileCheck %s --check-prefix=ROUND
// RUN: %rcgmp-opt %s -canonicalize | %FileCheck %s

// ROUND-LABEL: func.func @round_trip
// ROUND-SAME: (%[[X:.*]]: !rcgmp.mpz, %[[N:.*]]: i8, %[[Y:.*]]: !rcgmp.mpz<atomic>, %[[M:.*]]: i128)
// ROUND: rcgmp.floormod2expi %[[X]], %[[N]] : !rcgmp.mpz, i8
// ROUND: rcgmp.floormod2expi %[[Y]], %[[M]] : <atomic>, i128
func.func @round_trip(%x: !rcgmp.mpz, %n: i8, %y: !rcgmp.mpz<atomic>, %m: i128) -> (!rcgmp.mpz, !rcgmp.mpz<atomic>) {
  %r = rcgmp.floormod2expi %x, %n : !rcgmp.mpz, i8
  %a = rcgmp.floormod2expi %y, %m : !rcgmp.mpz<atomic>, i128
  return %r, %a : !rcgmp.mpz, !rcgmp.mpz<atomic>
}

// Constants fold to their nonnegative remainder, beyond any fixed width, with
// unsigned amounts. A nonnegative integer below the power and zero for any
// amount are their own remainders. A remainder of more than 4096 bits stays
// computed at run time.
// CHECK-LABEL: func.func @fold
// CHECK-SAME: (%[[N:.*]]: i64)
// CHECK-DAG: %[[THREE:.*]] = rcgmp.constant 3 : !rcgmp.mpz
// CHECK-DAG: %[[WIDE:.*]] = rcgmp.constant 1180591620717411303419 : !rcgmp.mpz
// CHECK-DAG: %[[SIX:.*]] = rcgmp.constant 6 : !rcgmp.mpz
// CHECK-DAG: %[[ZERO:.*]] = rcgmp.constant 0 : !rcgmp.mpz
// CHECK-DAG: %[[M1:.*]] = rcgmp.constant -1 : !rcgmp.mpz
// CHECK-DAG: %[[FAR:.*]] = rcgmp.floormod2expi %[[M1]], %{{.*}} : !rcgmp.mpz, i64
// CHECK: return %[[THREE]], %[[WIDE]], %[[THREE]], %[[SIX]], %[[ZERO]], %[[ZERO]], %[[FAR]]
func.func @fold(%n: i64) -> (!rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz) {
  %m5 = rcgmp.constant -5 : !rcgmp.mpz
  %m8 = rcgmp.constant -8 : !rcgmp.mpz
  %m1 = rcgmp.constant -1 : !rcgmp.mpz
  %six = rcgmp.constant 6 : !rcgmp.mpz
  %zero = rcgmp.constant 0 : !rcgmp.mpz
  %big = rcgmp.constant 340282366920938463463374607431768211459 : !rcgmp.mpz
  %c2 = arith.constant 2 : i64
  %c3 = arith.constant 3 : i64
  %c70 = arith.constant 70 : i8
  %all = arith.constant -1 : i8
  %c5000 = arith.constant 5000 : i64
  %a = rcgmp.floormod2expi %m5, %c3 : !rcgmp.mpz, i64
  %b = rcgmp.floormod2expi %m5, %c70 : !rcgmp.mpz, i8
  %c = rcgmp.floormod2expi %big, %c2 : !rcgmp.mpz, i64
  %d = rcgmp.floormod2expi %six, %all : !rcgmp.mpz, i8
  %e = rcgmp.floormod2expi %m8, %c3 : !rcgmp.mpz, i64
  %f = rcgmp.floormod2expi %zero, %n : !rcgmp.mpz, i64
  %g = rcgmp.floormod2expi %m1, %c5000 : !rcgmp.mpz, i64
  return %a, %b, %c, %d, %e, %f, %g : !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz
}

// The remainder by two to the power zero is zero, and the integer is released.
// CHECK-LABEL: func.func @zero_amount
// CHECK-SAME: (%[[X:.*]]: !rcgmp.mpz)
// CHECK: %[[ZERO:.*]] = rcgmp.constant 0 : !rcgmp.mpz
// CHECK: rcgmp.release %[[X]] : !rcgmp.mpz
// CHECK: return %[[ZERO]]
func.func @zero_amount(%x: !rcgmp.mpz) -> !rcgmp.mpz {
  %c0 = arith.constant 0 : i8
  %r = rcgmp.floormod2expi %x, %c0 : !rcgmp.mpz, i8
  return %r : !rcgmp.mpz
}

// Euclidean remainders by plus or minus 2^k and floored ones by 2^k, a
// constant or a shift of one used only there, take the low k bits. A floored
// remainder by -2^k is not nonnegative, and other divisors divide.
// CHECK-LABEL: func.func @remainders
// CHECK-SAME: (%[[X:.*]]: !rcgmp.mpz, %[[Y:.*]]: !rcgmp.mpz, %[[Z:.*]]: !rcgmp.mpz, %[[W:.*]]: !rcgmp.mpz, %[[V:.*]]: !rcgmp.mpz, %[[N:.*]]: i64)
// CHECK-DAG: %[[M8:.*]] = rcgmp.constant -8 : !rcgmp.mpz
// CHECK-DAG: %[[C6:.*]] = rcgmp.constant 6 : !rcgmp.mpz
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : i64
// CHECK-DAG: %[[C128:.*]] = arith.constant 128 : i64
// CHECK-DAG: %[[A:.*]] = rcgmp.floormod2expi %[[X]], %[[C3]] : !rcgmp.mpz, i64
// CHECK-DAG: %[[B:.*]] = rcgmp.floormod2expi %[[Y]], %[[C128]] : !rcgmp.mpz, i64
// CHECK-DAG: %[[C:.*]] = rcgmp.floormodsi %[[Z]], %[[M8]] : !rcgmp.mpz
// CHECK-DAG: %[[D:.*]] = rcgmp.emodsi %[[W]], %[[C6]] : !rcgmp.mpz
// CHECK-DAG: %[[E:.*]] = rcgmp.floormod2expi %[[V]], %[[N]] : !rcgmp.mpz, i64
// CHECK-NOT: rcgmp.shli
// CHECK: return %[[A]], %[[B]], %[[C]], %[[D]], %[[E]]
func.func @remainders(%x: !rcgmp.mpz, %y: !rcgmp.mpz, %z: !rcgmp.mpz, %w: !rcgmp.mpz, %v: !rcgmp.mpz, %n: i64) -> (!rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz) {
  %one = rcgmp.constant 1 : !rcgmp.mpz
  %m8 = rcgmp.constant -8 : !rcgmp.mpz
  %six = rcgmp.constant 6 : !rcgmp.mpz
  %power = rcgmp.constant 340282366920938463463374607431768211456 : !rcgmp.mpz
  %a = rcgmp.emodsi %x, %m8 : !rcgmp.mpz
  %b = rcgmp.floormodsi %y, %power divzero<lean> : !rcgmp.mpz
  %c = rcgmp.floormodsi %z, %m8 : !rcgmp.mpz
  %d = rcgmp.emodsi %w, %six : !rcgmp.mpz
  %shifted = rcgmp.shli %one, %n : !rcgmp.mpz, i64
  %e = rcgmp.emodsi %v, %shifted : !rcgmp.mpz
  return %a, %b, %c, %d, %e : !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz
}

// Truncated and floored remainders by plus or minus 2^k that are only compared
// with zero, on either side, or released, take the low k bits. A remainder
// used otherwise, compared with nonzero or ordered, or by another divisor,
// divides.
// CHECK-LABEL: func.func @zero_tests
// CHECK-SAME: (%[[X:.*]]: !rcgmp.mpz, %[[Y:.*]]: !rcgmp.mpz, %[[Z:.*]]: !rcgmp.mpz, %[[W:.*]]: !rcgmp.mpz, %[[V:.*]]: !rcgmp.mpz)
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : i64
// CHECK-DAG: %[[A:.*]] = rcgmp.floormod2expi %[[X]], %[[C1]] : !rcgmp.mpz, i64
// CHECK-DAG: %[[B:.*]] = rcgmp.floormod2expi %[[Y]], %[[C3]] : !rcgmp.mpz, i64
// CHECK-DAG: %[[C:.*]] = rcgmp.remsi %[[Z]], %{{.*}} : !rcgmp.mpz
// CHECK-DAG: %[[D:.*]] = rcgmp.remsi %[[W]], %{{.*}} : !rcgmp.mpz
// CHECK-DAG: %[[E:.*]] = rcgmp.remsi %[[V]], %{{.*}} : !rcgmp.mpz
// CHECK-DAG: rcgmp.cmpi eq, %[[A]]
// CHECK-DAG: rcgmp.release %[[A]]
// CHECK-DAG: rcgmp.cmpi ne, {{.*}}%[[B]]
// CHECK-DAG: rcgmp.release %[[B]]
// CHECK: return {{.*}}, %[[C]]
func.func @zero_tests(%x: !rcgmp.mpz, %y: !rcgmp.mpz, %z: !rcgmp.mpz, %w: !rcgmp.mpz, %v: !rcgmp.mpz) -> (i1, i1, i1, i1, !rcgmp.mpz) {
  %zero = rcgmp.constant 0 : !rcgmp.mpz
  %one = rcgmp.constant 1 : !rcgmp.mpz
  %two = rcgmp.constant 2 : !rcgmp.mpz
  %m8 = rcgmp.constant -8 : !rcgmp.mpz
  %six = rcgmp.constant 6 : !rcgmp.mpz
  %a = rcgmp.remsi %x, %two : !rcgmp.mpz
  %ta = rcgmp.cmpi eq, %a, %zero : !rcgmp.mpz
  rcgmp.release %a : !rcgmp.mpz
  %b = rcgmp.floormodsi %y, %m8 : !rcgmp.mpz
  %tb = rcgmp.cmpi ne, %zero, %b : !rcgmp.mpz
  rcgmp.release %b : !rcgmp.mpz
  %c = rcgmp.remsi %z, %two : !rcgmp.mpz
  %d = rcgmp.remsi %w, %two : !rcgmp.mpz
  %td = rcgmp.cmpi slt, %d, %zero : !rcgmp.mpz
  rcgmp.release %d : !rcgmp.mpz
  %e = rcgmp.remsi %v, %six : !rcgmp.mpz
  %te = rcgmp.cmpi eq, %e, %one : !rcgmp.mpz
  rcgmp.release %e : !rcgmp.mpz
  return %ta, %tb, %td, %te, %c : i1, i1, i1, i1, !rcgmp.mpz
}

// A shift of one used again keeps its reference for that use, and a shift of
// another integer is not a power of two.
// CHECK-LABEL: func.func @shared_shift
// CHECK: %[[SHIFTED:.*]] = rcgmp.shli
// CHECK: rcgmp.floormodsi %{{.*}}, %[[SHIFTED]]
// CHECK: %[[THREE:.*]] = rcgmp.shli
// CHECK: rcgmp.emodsi %{{.*}}, %[[THREE]]
func.func @shared_shift(%x: !rcgmp.mpz, %y: !rcgmp.mpz, %n: i64) -> (!rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz) {
  %one = rcgmp.constant 1 : !rcgmp.mpz
  %three = rcgmp.constant 3 : !rcgmp.mpz
  %shifted = rcgmp.shli %one, %n : !rcgmp.mpz, i64
  rcgmp.retain %shifted : !rcgmp.mpz
  %a = rcgmp.floormodsi %x, %shifted : !rcgmp.mpz
  %t = rcgmp.shli %three, %n : !rcgmp.mpz, i64
  %b = rcgmp.emodsi %y, %t : !rcgmp.mpz
  return %a, %b, %shifted : !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz
}

// Masks of k low bits take the low k bits, whether immediate or not, and a
// constant on the left moves to the right first. The mask zero, of no bits,
// leaves zero and releases the integer. Other masks stay.
// CHECK-LABEL: func.func @masks
// CHECK-SAME: (%[[X:.*]]: !rcgmp.mpz, %[[Y:.*]]: !rcgmp.mpz, %[[Z:.*]]: !rcgmp.mpz, %[[W:.*]]: !rcgmp.mpz, %[[V:.*]]: !rcgmp.mpz)
// CHECK-DAG: %[[ZERO:.*]] = rcgmp.constant 0 : !rcgmp.mpz
// CHECK-DAG: %[[SIX:.*]] = rcgmp.constant 6 : !rcgmp.mpz
// CHECK-DAG: %[[M1:.*]] = rcgmp.constant -1 : !rcgmp.mpz
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : i64
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : i64
// CHECK-DAG: %[[A:.*]] = rcgmp.floormod2expi %[[X]], %[[C8]] : !rcgmp.mpz, i64
// CHECK-DAG: %[[B:.*]] = rcgmp.floormod2expi %[[Y]], %[[C64]] : !rcgmp.mpz, i64
// CHECK-DAG: rcgmp.release %[[Z]] : !rcgmp.mpz
// CHECK-DAG: %[[D:.*]] = rcgmp.andi %[[W]], %[[SIX]] : !rcgmp.mpz
// CHECK-DAG: %[[E:.*]] = rcgmp.andi %[[V]], %[[M1]] : !rcgmp.mpz
// CHECK: return %[[A]], %[[B]], %[[ZERO]], %[[D]], %[[E]]
func.func @masks(%x: !rcgmp.mpz, %y: !rcgmp.mpz, %z: !rcgmp.mpz, %w: !rcgmp.mpz, %v: !rcgmp.mpz) -> (!rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz) {
  %byte = rcgmp.constant 255 : !rcgmp.mpz
  %limb = rcgmp.constant 18446744073709551615 : !rcgmp.mpz
  %zero = rcgmp.constant 0 : !rcgmp.mpz
  %six = rcgmp.constant 6 : !rcgmp.mpz
  %m1 = rcgmp.constant -1 : !rcgmp.mpz
  %a = rcgmp.andi %x, %byte : !rcgmp.mpz
  %b = rcgmp.andi %limb, %y : !rcgmp.mpz
  %c = rcgmp.andi %z, %zero : !rcgmp.mpz
  %d = rcgmp.andi %w, %six : !rcgmp.mpz
  %e = rcgmp.andi %v, %m1 : !rcgmp.mpz
  return %a, %b, %c, %d, %e : !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz, !rcgmp.mpz
}
