// RUN: %rcgmp-opt %s -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm -reconcile-unrealized-casts -o /dev/null
// Truncations compared against GMP by truncation.cpp. Each returns its value
// extended to i128, and the overflow variants store it and return the bit.
func.func @trunci_1(%x: !rcgmp.mpz) -> i128 {
  %v = rcgmp.trunci %x : !rcgmp.mpz to i1
  %e = arith.extsi %v : i1 to i128
  return %e : i128
}
func.func @overflow_signed_1(%x: !rcgmp.mpz, %out: !llvm.ptr) -> i32 {
  %v, %o = rcgmp.trunci_overflow %x : !rcgmp.mpz to i1
  %e = arith.extsi %v : i1 to i128
  llvm.store %e, %out : i128, !llvm.ptr
  %r = arith.extui %o : i1 to i32
  return %r : i32
}
func.func @overflow_unsigned_1(%x: !rcgmp.mpz, %out: !llvm.ptr) -> i32 {
  %v, %o = rcgmp.trunci_overflow %x unsigned : !rcgmp.mpz to i1
  %e = arith.extsi %v : i1 to i128
  llvm.store %e, %out : i128, !llvm.ptr
  %r = arith.extui %o : i1 to i32
  return %r : i32
}
func.func @trunci_8(%x: !rcgmp.mpz) -> i128 {
  %v = rcgmp.trunci %x : !rcgmp.mpz to i8
  %e = arith.extsi %v : i8 to i128
  return %e : i128
}
func.func @overflow_signed_8(%x: !rcgmp.mpz, %out: !llvm.ptr) -> i32 {
  %v, %o = rcgmp.trunci_overflow %x : !rcgmp.mpz to i8
  %e = arith.extsi %v : i8 to i128
  llvm.store %e, %out : i128, !llvm.ptr
  %r = arith.extui %o : i1 to i32
  return %r : i32
}
func.func @overflow_unsigned_8(%x: !rcgmp.mpz, %out: !llvm.ptr) -> i32 {
  %v, %o = rcgmp.trunci_overflow %x unsigned : !rcgmp.mpz to i8
  %e = arith.extsi %v : i8 to i128
  llvm.store %e, %out : i128, !llvm.ptr
  %r = arith.extui %o : i1 to i32
  return %r : i32
}
func.func @trunci_32(%x: !rcgmp.mpz) -> i128 {
  %v = rcgmp.trunci %x : !rcgmp.mpz to i32
  %e = arith.extsi %v : i32 to i128
  return %e : i128
}
func.func @overflow_signed_32(%x: !rcgmp.mpz, %out: !llvm.ptr) -> i32 {
  %v, %o = rcgmp.trunci_overflow %x : !rcgmp.mpz to i32
  %e = arith.extsi %v : i32 to i128
  llvm.store %e, %out : i128, !llvm.ptr
  %r = arith.extui %o : i1 to i32
  return %r : i32
}
func.func @overflow_unsigned_32(%x: !rcgmp.mpz, %out: !llvm.ptr) -> i32 {
  %v, %o = rcgmp.trunci_overflow %x unsigned : !rcgmp.mpz to i32
  %e = arith.extsi %v : i32 to i128
  llvm.store %e, %out : i128, !llvm.ptr
  %r = arith.extui %o : i1 to i32
  return %r : i32
}
func.func @trunci_61(%x: !rcgmp.mpz) -> i128 {
  %v = rcgmp.trunci %x : !rcgmp.mpz to i61
  %e = arith.extsi %v : i61 to i128
  return %e : i128
}
func.func @overflow_signed_61(%x: !rcgmp.mpz, %out: !llvm.ptr) -> i32 {
  %v, %o = rcgmp.trunci_overflow %x : !rcgmp.mpz to i61
  %e = arith.extsi %v : i61 to i128
  llvm.store %e, %out : i128, !llvm.ptr
  %r = arith.extui %o : i1 to i32
  return %r : i32
}
func.func @overflow_unsigned_61(%x: !rcgmp.mpz, %out: !llvm.ptr) -> i32 {
  %v, %o = rcgmp.trunci_overflow %x unsigned : !rcgmp.mpz to i61
  %e = arith.extsi %v : i61 to i128
  llvm.store %e, %out : i128, !llvm.ptr
  %r = arith.extui %o : i1 to i32
  return %r : i32
}
func.func @trunci_62(%x: !rcgmp.mpz) -> i128 {
  %v = rcgmp.trunci %x : !rcgmp.mpz to i62
  %e = arith.extsi %v : i62 to i128
  return %e : i128
}
func.func @overflow_signed_62(%x: !rcgmp.mpz, %out: !llvm.ptr) -> i32 {
  %v, %o = rcgmp.trunci_overflow %x : !rcgmp.mpz to i62
  %e = arith.extsi %v : i62 to i128
  llvm.store %e, %out : i128, !llvm.ptr
  %r = arith.extui %o : i1 to i32
  return %r : i32
}
func.func @overflow_unsigned_62(%x: !rcgmp.mpz, %out: !llvm.ptr) -> i32 {
  %v, %o = rcgmp.trunci_overflow %x unsigned : !rcgmp.mpz to i62
  %e = arith.extsi %v : i62 to i128
  llvm.store %e, %out : i128, !llvm.ptr
  %r = arith.extui %o : i1 to i32
  return %r : i32
}
func.func @trunci_63(%x: !rcgmp.mpz) -> i128 {
  %v = rcgmp.trunci %x : !rcgmp.mpz to i63
  %e = arith.extsi %v : i63 to i128
  return %e : i128
}
func.func @overflow_signed_63(%x: !rcgmp.mpz, %out: !llvm.ptr) -> i32 {
  %v, %o = rcgmp.trunci_overflow %x : !rcgmp.mpz to i63
  %e = arith.extsi %v : i63 to i128
  llvm.store %e, %out : i128, !llvm.ptr
  %r = arith.extui %o : i1 to i32
  return %r : i32
}
func.func @overflow_unsigned_63(%x: !rcgmp.mpz, %out: !llvm.ptr) -> i32 {
  %v, %o = rcgmp.trunci_overflow %x unsigned : !rcgmp.mpz to i63
  %e = arith.extsi %v : i63 to i128
  llvm.store %e, %out : i128, !llvm.ptr
  %r = arith.extui %o : i1 to i32
  return %r : i32
}
func.func @trunci_64(%x: !rcgmp.mpz) -> i128 {
  %v = rcgmp.trunci %x : !rcgmp.mpz to i64
  %e = arith.extsi %v : i64 to i128
  return %e : i128
}
func.func @overflow_signed_64(%x: !rcgmp.mpz, %out: !llvm.ptr) -> i32 {
  %v, %o = rcgmp.trunci_overflow %x : !rcgmp.mpz to i64
  %e = arith.extsi %v : i64 to i128
  llvm.store %e, %out : i128, !llvm.ptr
  %r = arith.extui %o : i1 to i32
  return %r : i32
}
func.func @overflow_unsigned_64(%x: !rcgmp.mpz, %out: !llvm.ptr) -> i32 {
  %v, %o = rcgmp.trunci_overflow %x unsigned : !rcgmp.mpz to i64
  %e = arith.extsi %v : i64 to i128
  llvm.store %e, %out : i128, !llvm.ptr
  %r = arith.extui %o : i1 to i32
  return %r : i32
}
func.func @trunci_65(%x: !rcgmp.mpz) -> i128 {
  %v = rcgmp.trunci %x : !rcgmp.mpz to i65
  %e = arith.extsi %v : i65 to i128
  return %e : i128
}
func.func @overflow_signed_65(%x: !rcgmp.mpz, %out: !llvm.ptr) -> i32 {
  %v, %o = rcgmp.trunci_overflow %x : !rcgmp.mpz to i65
  %e = arith.extsi %v : i65 to i128
  llvm.store %e, %out : i128, !llvm.ptr
  %r = arith.extui %o : i1 to i32
  return %r : i32
}
func.func @overflow_unsigned_65(%x: !rcgmp.mpz, %out: !llvm.ptr) -> i32 {
  %v, %o = rcgmp.trunci_overflow %x unsigned : !rcgmp.mpz to i65
  %e = arith.extsi %v : i65 to i128
  llvm.store %e, %out : i128, !llvm.ptr
  %r = arith.extui %o : i1 to i32
  return %r : i32
}
func.func @trunci_127(%x: !rcgmp.mpz) -> i128 {
  %v = rcgmp.trunci %x : !rcgmp.mpz to i127
  %e = arith.extsi %v : i127 to i128
  return %e : i128
}
func.func @overflow_signed_127(%x: !rcgmp.mpz, %out: !llvm.ptr) -> i32 {
  %v, %o = rcgmp.trunci_overflow %x : !rcgmp.mpz to i127
  %e = arith.extsi %v : i127 to i128
  llvm.store %e, %out : i128, !llvm.ptr
  %r = arith.extui %o : i1 to i32
  return %r : i32
}
func.func @overflow_unsigned_127(%x: !rcgmp.mpz, %out: !llvm.ptr) -> i32 {
  %v, %o = rcgmp.trunci_overflow %x unsigned : !rcgmp.mpz to i127
  %e = arith.extsi %v : i127 to i128
  llvm.store %e, %out : i128, !llvm.ptr
  %r = arith.extui %o : i1 to i32
  return %r : i32
}
func.func @trunci_128(%x: !rcgmp.mpz) -> i128 {
  %v = rcgmp.trunci %x : !rcgmp.mpz to i128
  return %v : i128
}
func.func @overflow_signed_128(%x: !rcgmp.mpz, %out: !llvm.ptr) -> i32 {
  %v, %o = rcgmp.trunci_overflow %x : !rcgmp.mpz to i128
  llvm.store %v, %out : i128, !llvm.ptr
  %r = arith.extui %o : i1 to i32
  return %r : i32
}
func.func @overflow_unsigned_128(%x: !rcgmp.mpz, %out: !llvm.ptr) -> i32 {
  %v, %o = rcgmp.trunci_overflow %x unsigned : !rcgmp.mpz to i128
  llvm.store %v, %out : i128, !llvm.ptr
  %r = arith.extui %o : i1 to i32
  return %r : i32
}
func.func @trunci_nsw_8(%x: !rcgmp.mpz) -> i128 {
  %v = rcgmp.trunci %x overflow<nsw> : !rcgmp.mpz to i8
  %e = arith.extsi %v : i8 to i128
  return %e : i128
}
func.func @trunci_nsw_32(%x: !rcgmp.mpz) -> i128 {
  %v = rcgmp.trunci %x overflow<nsw> : !rcgmp.mpz to i32
  %e = arith.extsi %v : i32 to i128
  return %e : i128
}
func.func @trunci_nsw_62(%x: !rcgmp.mpz) -> i128 {
  %v = rcgmp.trunci %x overflow<nsw> : !rcgmp.mpz to i62
  %e = arith.extsi %v : i62 to i128
  return %e : i128
}
func.func @trunci_nsw_64(%x: !rcgmp.mpz) -> i128 {
  %v = rcgmp.trunci %x overflow<nsw> : !rcgmp.mpz to i64
  %e = arith.extsi %v : i64 to i128
  return %e : i128
}
func.func @trunci_nuw_8(%x: !rcgmp.mpz) -> i128 {
  %v = rcgmp.trunci %x overflow<nuw> : !rcgmp.mpz to i8
  %e = arith.extui %v : i8 to i128
  return %e : i128
}
func.func @trunci_nuw_32(%x: !rcgmp.mpz) -> i128 {
  %v = rcgmp.trunci %x overflow<nuw> : !rcgmp.mpz to i32
  %e = arith.extui %v : i32 to i128
  return %e : i128
}
func.func @trunci_nuw_61(%x: !rcgmp.mpz) -> i128 {
  %v = rcgmp.trunci %x overflow<nuw> : !rcgmp.mpz to i61
  %e = arith.extui %v : i61 to i128
  return %e : i128
}
func.func @trunci_nuw_64(%x: !rcgmp.mpz) -> i128 {
  %v = rcgmp.trunci %x overflow<nuw> : !rcgmp.mpz to i64
  %e = arith.extui %v : i64 to i128
  return %e : i128
}
func.func @trunci_atomic_64(%x: !rcgmp.mpz<atomic>) -> i64 {
  %v = rcgmp.trunci %x : !rcgmp.mpz<atomic> to i64
  return %v : i64
}
