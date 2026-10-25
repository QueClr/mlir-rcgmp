// RUN: %rcgmp-opt %s | %rcgmp-opt | %FileCheck %s

// The default trap mode is elided.
// CHECK-LABEL: func.func @divisions
// CHECK: rcgmp.divsi %{{[^ ]*}}, %{{[^ ]*}} : !rcgmp.mpz
// CHECK: rcgmp.remsi %{{.*}}, %{{.*}} divzero<undefined> : !rcgmp.mpz
// CHECK: rcgmp.floordivsi %{{.*}}, %{{.*}} divzero<lean> : <atomic>
// CHECK: rcgmp.floormodsi %{{.*}}, %{{.*}} divzero<undefined> : <atomic>
// CHECK: %{{.*}}, %{{.*}} = rcgmp.divremsi %{{[^ ]*}}, %{{[^ ]*}} : !rcgmp.mpz
// CHECK: %{{.*}}, %{{.*}} = rcgmp.floordivmodsi %{{.*}}, %{{.*}} divzero<undefined> : <atomic>
// CHECK: rcgmp.edivsi %{{.*}}, %{{.*}} divzero<lean> : !rcgmp.mpz
// CHECK: rcgmp.emodsi %{{.*}}, %{{.*}} divzero<undefined> : !rcgmp.mpz
// CHECK: %{{.*}}, %{{.*}} = rcgmp.edivmodsi %{{.*}}, %{{.*}} divzero<lean> : <atomic>
func.func @divisions(%a: !rcgmp.mpz, %b: !rcgmp.mpz, %c: !rcgmp.mpz,
                     %d: !rcgmp.mpz, %e: !rcgmp.mpz<atomic>,
                     %f: !rcgmp.mpz<atomic>, %g: !rcgmp.mpz<atomic>,
                     %h: !rcgmp.mpz<atomic>, %i: !rcgmp.mpz) {
  %q = rcgmp.divsi %a, %b divzero<trap> : !rcgmp.mpz
  %r = rcgmp.remsi %c, %d divzero<undefined> : !rcgmp.mpz
  %fq = "rcgmp.floordivsi"(%e, %f) {divZero = #rcgmp.divzero<lean>}
      : (!rcgmp.mpz<atomic>, !rcgmp.mpz<atomic>) -> !rcgmp.mpz<atomic>
  %fr = rcgmp.floormodsi %g, %h divzero<undefined> : !rcgmp.mpz<atomic>
  %q2, %r2 = rcgmp.divremsi %q, %r : !rcgmp.mpz
  %fq2, %fr2 = rcgmp.floordivmodsi %fq, %fr divzero<undefined> : !rcgmp.mpz<atomic>
  %eq = rcgmp.edivsi %q2, %r2 divzero<lean> : !rcgmp.mpz
  %er = rcgmp.emodsi %eq, %i divzero<undefined> : !rcgmp.mpz
  %eq2, %er2 = rcgmp.edivmodsi %fq2, %fr2 divzero<lean> : !rcgmp.mpz<atomic>
  rcgmp.release %er : !rcgmp.mpz
  rcgmp.release %eq2 : !rcgmp.mpz<atomic>
  rcgmp.release %er2 : !rcgmp.mpz<atomic>
  return
}

// nonnegative follows exact and precedes the zero-divisor mode.
// CHECK-LABEL: func.func @nonnegative
// CHECK: rcgmp.divsi %{{[^ ]*}}, %{{[^ ]*}} exact nonnegative divzero<lean> : !rcgmp.mpz
// CHECK: rcgmp.remsi %{{[^ ]*}}, %{{[^ ]*}} nonnegative : !rcgmp.mpz
// CHECK: rcgmp.edivmodsi %{{[^ ]*}}, %{{[^ ]*}} nonnegative divzero<undefined> : !rcgmp.mpz
func.func @nonnegative(%a: !rcgmp.mpz, %b: !rcgmp.mpz, %c: !rcgmp.mpz,
                       %d: !rcgmp.mpz) {
  %q = rcgmp.divsi %a, %b exact nonnegative divzero<lean> : !rcgmp.mpz
  %r = rcgmp.remsi %q, %c nonnegative : !rcgmp.mpz
  %eq, %er = rcgmp.edivmodsi %r, %d nonnegative divzero<undefined> : !rcgmp.mpz
  rcgmp.release %eq : !rcgmp.mpz
  rcgmp.release %er : !rcgmp.mpz
  return
}
