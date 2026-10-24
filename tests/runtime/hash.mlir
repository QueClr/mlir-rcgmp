// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf | %FileCheck %s --check-prefix=SCF
// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts | %FileCheck %s --check-prefix=LLVM
// Entry points compared with the runtime and a reference hash by hash.cpp.

// An immediate hashes to its value; the runtime hashes a heap integer.
// SCF-DAG: func.func private @rcgmp_mpz_hash(!rcgmp.mpz) -> i64
// SCF-DAG: func.func private @rcgmp_mpz_hash_atomic(!rcgmp.mpz<atomic>) -> i64
// SCF-LABEL: func.func @hashi_local
// SCF: %[[BITS:.*]] = rcgmp.bridge.mpz.to_bits %{{.*}}
// SCF: %[[C2:.*]] = arith.constant 2 : i64
// SCF: %[[VALUE:.*]] = arith.shrsi %[[BITS]], %[[C2]] : i64
// SCF: scf.if %{{.*}} -> (i64) {
// SCF-NEXT: scf.yield %[[VALUE]] : i64
// SCF-NEXT: } else {
// SCF-NEXT: %[[HASH:.*]] = func.call @rcgmp_mpz_hash(%{{.*}}) : (!rcgmp.mpz) -> i64
// SCF-NEXT: scf.yield %[[HASH]] : i64
// LLVM-LABEL: llvm.func @hashi_atomic
// LLVM: llvm.call @rcgmp_mpz_hash_atomic
func.func @hashi_local(%x: !rcgmp.mpz) -> i64 {
  %h = rcgmp.hashi %x : !rcgmp.mpz
  return %h : i64
}
func.func @hashi_atomic(%x: !rcgmp.mpz<atomic>) -> i64 {
  %h = rcgmp.hashi %x : !rcgmp.mpz<atomic>
  return %h : i64
}
// A compiled constant beyond the immediate range is an immortal heap integer,
// which hashes as the same value on the heap does.
func.func @hashi_constant() -> i64 {
  %c = rcgmp.constant -340282366920938463463374607431768211457 : !rcgmp.mpz
  %h = rcgmp.hashi %c : !rcgmp.mpz
  return %h : i64
}
