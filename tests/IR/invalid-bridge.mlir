// RUN: %rcgmp-opt %s -split-input-file -verify-diagnostics
func.func @to_wrong_width(%x: !rcgmp.mpz) -> i32 {
  // expected-error@+1 {{requires an integer matching the handle bit width}}
  %r = rcgmp.bridge.mpz.to_bits %x : !rcgmp.mpz to i32
  return %r : i32
}
// -----
func.func @from_wrong_width(%x: i32) -> !rcgmp.mpz {
  // expected-error@+1 {{requires an integer matching the handle bit width}}
  %r = rcgmp.bridge.mpz.from_bits %x : i32 to !rcgmp.mpz
  return %r : !rcgmp.mpz
}
// -----
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<32> : vector<4xi64>>>} {
  func.func @wrong_target_width(%x: !rcgmp.mpz<atomic>) -> i64 {
    // expected-error@+1 {{requires an integer matching the handle bit width}}
    %r = rcgmp.bridge.mpz.to_bits %x : !rcgmp.mpz<atomic> to i64
    return %r : i64
  }
}
