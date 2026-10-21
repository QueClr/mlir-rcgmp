module;
#include "RCGMP/Runtime/Mpz.h"
#include <gmp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
export module rcgmp.runtime:bitwise;
import :binary;
import :mpz;

namespace rcgmp {

// Negates a number a limb at a time, least significant first, as ~x + 1. The
// same step converts between a negative magnitude and its two's complement.
class Negation {
public:
  constexpr Limb operator()(Limb limb) {
    limb = ~limb + carry;
    carry = carry && limb == 0;
    return limb;
  }

private:
  Limb carry = 1;
};

template <Counting counting, bool exclusive>
Handle bitwise(Handle lhs, Handle rhs, Reuse reuse) {
  Magnitude left(lhs), right(rhs);
  bool negative = exclusive ? left.negative != right.negative
                            : left.negative && right.negative;
  std::size_t count;
  if (exclusive || negative)
    // Converting a negative result back to magnitude may carry into a new limb.
    count = std::max(left.count(), right.count()) + negative;
  else if (left.negative)
    count = right.count();
  else if (right.negative)
    count = left.count();
  else
    count = std::min(left.count(), right.count());
  if (!count)
    return consume<counting>(lhs, rhs, Handle::immediate(0));

  Mpz &result = destination(lhs, rhs, reuse, count);
  std::span<Limb> out = result.limbs();
  if (!left.negative && !right.negative) {
    std::size_t common = std::min(left.count(), right.count());
    if (common) {
      if (exclusive)
        mpn_xor_n(out.data(), left.limbs.data(), right.limbs.data(), common);
      else
        mpn_and_n(out.data(), left.limbs.data(), right.limbs.data(), common);
    }
    if (count > common) {
      const Magnitude &longer = left.count() > right.count() ? left : right;
      if (out.data() != longer.limbs.data())
        std::ranges::copy(longer.limbs.subspan(common), &out[common]);
    }
  } else {
    // Read both source limbs before writing, so either unique input can be
    // reused without a temporary two's-complement allocation.
    Negation leftNegation, rightNegation, resultNegation;
    for (std::size_t i = 0; i < count; ++i) {
      Limb a = left.limb(i), b = right.limb(i);
      if (left.negative)
        a = leftNegation(a);
      if (right.negative)
        b = rightNegation(b);
      Limb limb = exclusive ? a ^ b : a & b;
      out[i] = negative ? resultNegation(limb) : limb;
    }
  }
  result.setMagnitude(count, negative);
  return finish<counting>(result, lhs, rhs);
}

export template <Counting counting>
Handle bitwiseAnd(Handle lhs, Handle rhs, Reuse reuse) {
  return bitwise<counting, false>(lhs, rhs, reuse);
}

export template <Counting counting>
Handle bitwiseXor(Handle lhs, Handle rhs, Reuse reuse) {
  return bitwise<counting, true>(lhs, rhs, reuse);
}

} // namespace rcgmp

extern "C" {

void *rcgmp_mpz_and(void *lhs, void *rhs, std::uint32_t reuse) {
  return rcgmp::binaryEntryPoint<rcgmp::bitwiseAnd<rcgmp::Counting::nonatomic>>(
      lhs, rhs, reuse);
}

void *rcgmp_mpz_and_atomic(void *lhs, void *rhs, std::uint32_t reuse) {
  return rcgmp::binaryEntryPoint<rcgmp::bitwiseAnd<rcgmp::Counting::atomic>>(
      lhs, rhs, reuse);
}

void *rcgmp_mpz_xor(void *lhs, void *rhs, std::uint32_t reuse) {
  return rcgmp::binaryEntryPoint<rcgmp::bitwiseXor<rcgmp::Counting::nonatomic>>(
      lhs, rhs, reuse);
}

void *rcgmp_mpz_xor_atomic(void *lhs, void *rhs, std::uint32_t reuse) {
  return rcgmp::binaryEntryPoint<rcgmp::bitwiseXor<rcgmp::Counting::atomic>>(
      lhs, rhs, reuse);
}
}
