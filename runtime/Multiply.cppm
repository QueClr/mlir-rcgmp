module;
#include "RCGMP/Runtime/Mpz.h"
#include <gmp.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
export module rcgmp.runtime:multiply;
import :binary;
import :mpz;

namespace rcgmp {

export template <Counting counting>
Handle multiply(Handle lhs, Handle rhs, Reuse reuse) {
  Magnitude left(lhs), right(rhs);
  if (!left.count() || !right.count())
    return consume<counting>(lhs, rhs, Handle::immediate(0));
  bool swapped = left.count() < right.count();
  Magnitude &a = swapped ? right : left;
  Magnitude &b = swapped ? left : right;
  std::size_t count = a.count() + b.count();
  // mpn_mul_1 permits exact overlap. General mpn_mul and mpn_sqr do not.
  bool scalar = b.count() == 1;
  Mpz &result = destination(lhs, rhs, scalar ? reuse : Reuse{}, count);
  std::span<Limb> out = result.limbs();
  const Limb *x = a.limbs.data();
  if (scalar) {
    Limb factor = b.limbs[0], carry = 0;
    if (factor == 1) {
      if (out.data() != x)
        std::ranges::copy(a.limbs, out.begin());
    } else if (std::has_single_bit(factor)) {
      carry = mpn_lshift(out.data(), x, a.count(), std::countr_zero(factor));
    } else {
      carry = mpn_mul_1(out.data(), x, a.count(), factor);
    }
    out[a.count()] = carry;
  } else if (x == b.limbs.data()) {
    mpn_sqr(out.data(), x, a.count());
  } else {
    mpn_mul(out.data(), x, a.count(), b.limbs.data(), b.count());
  }
  result.setMagnitude(count, a.negative != b.negative);
  return finish<counting>(result, lhs, rhs);
}

} // namespace rcgmp

extern "C" {

void *rcgmp_mpz_mul(void *lhs, void *rhs, std::uint32_t reuse) {
  return rcgmp::binaryEntryPoint<rcgmp::multiply<rcgmp::Counting::nonatomic>>(
      lhs, rhs, reuse);
}

void *rcgmp_mpz_mul_atomic(void *lhs, void *rhs, std::uint32_t reuse) {
  return rcgmp::binaryEntryPoint<rcgmp::multiply<rcgmp::Counting::atomic>>(
      lhs, rhs, reuse);
}
}
