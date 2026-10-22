module;
#include "RCGMP/Runtime/Mpz.h"
#include <gmp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
export module rcgmp.runtime:shift;
import :binary;
import :mpz;

namespace rcgmp {

constexpr unsigned limbBits = std::numeric_limits<Limb>::digits;

// Returns storage for a result of up to required limbs: the heap operand if
// it may be reused and has enough capacity, or else a new integer.
inline Mpz &shiftDestination(Handle value, bool reuse, std::size_t required) {
  if (reuse && value.heap().capacity >= required)
    return value.heap();
  return allocateResult(required);
}

// Multiplies by 2^amount, moving the magnitude up by whole limbs and then by
// the remaining bits. The sign is unchanged.
export template <Counting counting>
Handle shiftLeft(Handle value, std::uintptr_t amount, bool reuse) {
  Magnitude source(value);
  if (!source.count() || !amount)
    return value; // The consumed reference becomes the result's.
  std::size_t limbs = amount / limbBits;
  unsigned bits = amount % limbBits;
  std::size_t count = source.count();
  // Allocation rejects the saturated count.
  std::size_t required =
      std::add_sat(std::add_sat(count, limbs), std::size_t{bits != 0});
  Mpz &result = shiftDestination(value, reuse, required);
  Limb *out = result.limbs().data();
  const Limb *in = source.limbs.data();
  // Both move limbs to higher addresses, which the shared storage of a
  // reused operand permits.
  std::size_t total = limbs + count;
  if (bits) {
    out[total] = mpn_lshift(out + limbs, in, count, bits);
    total += out[total] != 0;
  } else {
    std::copy_backward(in, in + count, out + total);
  }
  std::fill(out, out + limbs, Limb{0});
  result.setMagnitude(total, source.negative);
  if (Handle(result) != value)
    release<counting>(value);
  return normalize<counting>(result);
}

// Divides by 2^amount, rounding toward negative infinity: a negative integer
// whose shifted-out bits are not all zero gains one in magnitude.
export template <Counting counting>
Handle shiftRight(Handle value, std::uintptr_t amount, bool reuse) {
  if (value.isImmediate()) {
    std::intptr_t integer = value.value();
    unsigned shift = static_cast<unsigned>(std::min<std::uintptr_t>(
        amount, std::numeric_limits<std::intptr_t>::digits));
    return Handle::immediate(integer >> shift);
  }
  if (!amount)
    return value; // The consumed reference becomes the result's.
  Mpz &heap = value.heap();
  bool negative = heap.negative();
  std::size_t count = heap.count();
  std::size_t limbs = amount / limbBits;
  unsigned bits = amount % limbBits;
  if (limbs >= count) {
    release<counting>(value);
    return Handle::immediate(negative ? -1 : 0);
  }
  const Limb *in = heap.limbs().data();
  bool inexact =
      negative &&
      (std::any_of(in, in + limbs, [](Limb limb) { return limb != 0; }) ||
       (bits && (in[limbs] & ((Limb{1} << bits) - 1))));
  std::size_t total = count - limbs;
  // Rounding a negative quotient up carries into one more limb only when the
  // amount is a whole number of limbs, and then a reused operand has that
  // limb. Otherwise the shift clears the top bit, so the carry stays inside.
  Mpz &result = shiftDestination(value, reuse, total + (negative && !bits));
  Limb *out = result.limbs().data();
  // Both move limbs to lower addresses, which the shared storage of a reused
  // operand permits.
  if (bits)
    mpn_rshift(out, in + limbs, total, bits);
  else
    std::copy(in + limbs, in + count, out);
  if (inexact)
    if (Limb carry = mpn_add_1(out, out, total, 1))
      out[total++] = carry;
  result.setMagnitude(total, negative);
  if (Handle(result) != value)
    release<counting>(value);
  return normalize<counting>(result);
}

// Adapts a shift to the runtime ABI, where bit 0 of reuse permits reusing the
// operand.
template <auto shift>
void *shiftEntryPoint(void *value, std::uintptr_t amount, std::uint32_t reuse) {
  return shift(Handle(value), amount, (reuse & 1) != 0).pointer();
}

} // namespace rcgmp

extern "C" {

void *rcgmp_mpz_shl(void *value, std::uintptr_t amount, std::uint32_t reuse) {
  return rcgmp::shiftEntryPoint<rcgmp::shiftLeft<rcgmp::Counting::nonatomic>>(
      value, amount, reuse);
}

void *rcgmp_mpz_shl_atomic(void *value, std::uintptr_t amount,
                           std::uint32_t reuse) {
  return rcgmp::shiftEntryPoint<rcgmp::shiftLeft<rcgmp::Counting::atomic>>(
      value, amount, reuse);
}

void *rcgmp_mpz_shr(void *value, std::uintptr_t amount, std::uint32_t reuse) {
  return rcgmp::shiftEntryPoint<rcgmp::shiftRight<rcgmp::Counting::nonatomic>>(
      value, amount, reuse);
}

void *rcgmp_mpz_shr_atomic(void *value, std::uintptr_t amount,
                           std::uint32_t reuse) {
  return rcgmp::shiftEntryPoint<rcgmp::shiftRight<rcgmp::Counting::atomic>>(
      value, amount, reuse);
}
}
