// RUN: %rcgmp-opt %s | %rcgmp-opt | %FileCheck %s

// CHECK-LABEL: func.func @divisions
// CHECK: rcgmp.divsi %{{.*}}, %{{.*}} : !rcgmp.mpz
// CHECK: rcgmp.remsi %{{.*}}, %{{.*}} assume_nonzero : !rcgmp.mpz
// CHECK: rcgmp.floordivsi %{{.*}}, %{{.*}} : <atomic>
// CHECK: rcgmp.floormodsi %{{.*}}, %{{.*}} assume_nonzero : <atomic>
// CHECK: %{{.*}}, %{{.*}} = rcgmp.divremsi %{{.*}}, %{{.*}} : !rcgmp.mpz
// CHECK: %{{.*}}, %{{.*}} = rcgmp.floordivmodsi %{{.*}}, %{{.*}} assume_nonzero : <atomic>
// CHECK: rcgmp.edivsi %{{.*}}, %{{.*}} : !rcgmp.mpz
// CHECK: rcgmp.emodsi %{{.*}}, %{{.*}} assume_nonzero : !rcgmp.mpz
// CHECK: %{{.*}}, %{{.*}} = rcgmp.edivmodsi %{{.*}}, %{{.*}} : <atomic>
func.func @divisions(%a: !rcgmp.mpz, %b: !rcgmp.mpz, %c: !rcgmp.mpz,
                     %d: !rcgmp.mpz, %e: !rcgmp.mpz<atomic>,
                     %f: !rcgmp.mpz<atomic>, %g: !rcgmp.mpz<atomic>,
                     %h: !rcgmp.mpz<atomic>, %i: !rcgmp.mpz) {
  %q = rcgmp.divsi %a, %b : !rcgmp.mpz
  %r = rcgmp.remsi %c, %d assume_nonzero : !rcgmp.mpz
  %fq = rcgmp.floordivsi %e, %f : !rcgmp.mpz<atomic>
  %fr = rcgmp.floormodsi %g, %h assume_nonzero : !rcgmp.mpz<atomic>
  %q2, %r2 = rcgmp.divremsi %q, %r : !rcgmp.mpz
  %fq2, %fr2 = rcgmp.floordivmodsi %fq, %fr assume_nonzero : !rcgmp.mpz<atomic>
  %eq = rcgmp.edivsi %q2, %r2 : !rcgmp.mpz
  %er = rcgmp.emodsi %eq, %i assume_nonzero : !rcgmp.mpz
  %eq2, %er2 = rcgmp.edivmodsi %fq2, %fr2 : !rcgmp.mpz<atomic>
  rcgmp.release %er : !rcgmp.mpz
  rcgmp.release %eq2 : !rcgmp.mpz<atomic>
  rcgmp.release %er2 : !rcgmp.mpz<atomic>
  return
}
