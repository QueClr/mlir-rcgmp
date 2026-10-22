module;
#include "RCGMP/Runtime/Mpz.h"
#include <gmp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
export module rcgmp.runtime:unary;
import :binary;
import :mpz;

namespace rcgmp {

// Returns storage for a result of up to required limbs: the operand if it may
// be reused and has enough capacity, or else a new integer.
inline Mpz &destination(Handle value, bool reuse, std::size_t required) {
  return destination(value, value, {.lhs = reuse}, required);
}

// Consumes the operand after all its limbs have been read, then canonicalizes
// the result.
template <Counting counting> Handle finish(Mpz &result, Handle value) {
  if (Handle(result) != value)
    release<counting>(value);
  return normalize<counting>(result);
}

// Consumes an operand and returns its magnitude with the given sign.
template <Counting counting>
Handle withSign(Handle value, bool reuse, bool negative) {
  Magnitude magnitude(value);
  if (value.isImmediate())
    return fromWords(magnitude.limbs, negative);
  if (magnitude.negative == negative)
    return value; // The consumed reference becomes the result's.
  std::size_t count = magnitude.count();
  Mpz &result = destination(value, reuse, count);
  if (Handle(result) != value)
    std::ranges::copy(magnitude.limbs, result.limbs().begin());
  result.setMagnitude(count, negative);
  // Negating 2^(W-3) yields the immediate minimum.
  return finish<counting>(result, value);
}

export template <Counting counting> Handle negate(Handle value, bool reuse) {
  return withSign<counting>(value, reuse, !Magnitude(value).negative);
}

export template <Counting counting> Handle absolute(Handle value, bool reuse) {
  return withSign<counting>(value, reuse, false);
}

// The complement of x is -(x + 1) for nonnegative x and |x| - 1 for negative x.
export template <Counting counting>
Handle complement(Handle value, bool reuse) {
  if (value.isImmediate())
    return Handle::immediate(~value.value());
  Mpz &heap = value.heap();
  bool negative = heap.negative();
  std::size_t count = heap.count();
  Mpz &result = destination(value, reuse, count + !negative);
  std::span<Limb> out = result.limbs();
  const Limb *x = heap.magnitude().data();
  if (negative) {
    // A heap magnitude is nonzero, so there is no final borrow.
    mpn_sub_1(out.data(), x, count, 1);
  } else {
    out[count] = mpn_add_1(out.data(), x, count, 1);
    count += out[count];
  }
  result.setMagnitude(count, !negative);
  return finish<counting>(result, value);
}

// Adapts a unary operation to the runtime ABI, where bit 0 of reuse permits
// reusing the operand.
template <auto operation>
void *unaryEntryPoint(void *value, std::uint32_t reuse) {
  return operation(Handle(value), (reuse & 1) != 0).pointer();
}

} // namespace rcgmp

extern "C" {

void *rcgmp_mpz_not(void *value, std::uint32_t reuse) {
  return rcgmp::unaryEntryPoint<rcgmp::complement<rcgmp::Counting::nonatomic>>(
      value, reuse);
}

void *rcgmp_mpz_not_atomic(void *value, std::uint32_t reuse) {
  return rcgmp::unaryEntryPoint<rcgmp::complement<rcgmp::Counting::atomic>>(
      value, reuse);
}

void *rcgmp_mpz_neg(void *value, std::uint32_t reuse) {
  return rcgmp::unaryEntryPoint<rcgmp::negate<rcgmp::Counting::nonatomic>>(
      value, reuse);
}

void *rcgmp_mpz_neg_atomic(void *value, std::uint32_t reuse) {
  return rcgmp::unaryEntryPoint<rcgmp::negate<rcgmp::Counting::atomic>>(value,
                                                                        reuse);
}

void *rcgmp_mpz_abs(void *value, std::uint32_t reuse) {
  return rcgmp::unaryEntryPoint<rcgmp::absolute<rcgmp::Counting::nonatomic>>(
      value, reuse);
}

void *rcgmp_mpz_abs_atomic(void *value, std::uint32_t reuse) {
  return rcgmp::unaryEntryPoint<rcgmp::absolute<rcgmp::Counting::atomic>>(
      value, reuse);
}
}
