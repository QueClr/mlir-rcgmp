// RUN: %rcgmp-opt %s -split-input-file -verify-diagnostics
func.func @limb_wrong_width(%x: !rcgmp.mpz, %i: index) -> i32 {
  // expected-error@+1 {{requires an integer matching the handle bit width}}
  %r = rcgmp.limb %x[%i] : !rcgmp.mpz to i32
  return %r : i32
}
// -----
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi64>>>} {
  func.func @limb_wrong_target_width(%x: !rcgmp.mpz, %i: index) -> i64 {
    // expected-error@+1 {{requires an integer matching the handle bit width}}
    %r = rcgmp.limb %x[%i] : !rcgmp.mpz to i64
    return %r : i64
  }
}
