module;
#include "RCGMP/Runtime/Mpz.h"
#include <gmp.h>

#include <compare>
#include <cstddef>
#include <cstdint>
#include <span>
export module rcgmp.runtime:add;
import :binary;
import :mpz;

namespace rcgmp {

template <Counting counting, bool subtracting>
Handle addOrSubtract(Handle lhs, Handle rhs, Reuse reuse) {
  Magnitude left(lhs), right(rhs);
  if (subtracting && right.count())
    right.negative = !right.negative;
  if (!left.count() && !subtracting)
    return rhs; // The consumed zero has no reference count.
  if (!right.count())
    return lhs;

  // The larger magnitude leads both addition and subtraction. Alias the views,
  // not their storage: immediate limbs refer to the views themselves.
  bool sameSign = left.negative == right.negative;
  std::strong_ordering order = left.count() <=> right.count();
  if (!sameSign && order == 0)
    order = mpn_cmp(left.limbs.data(), right.limbs.data(), left.count()) <=> 0;
  if (!sameSign && order == 0)
    return consume<counting>(lhs, rhs, Handle::immediate(0));
  Magnitude &a = order < 0 ? right : left;
  Magnitude &b = order < 0 ? left : right;

  if (a.count() == 1 && lhs.isImmediate() && rhs.isImmediate()) {
    // The sum or difference of two signed W-2-bit payloads fits intptr_t.
    std::intptr_t sum =
        subtracting ? lhs.value() - rhs.value() : lhs.value() + rhs.value();
    std::uintptr_t word = sum < 0 ? -static_cast<std::uintptr_t>(sum) : sum;
    return fromWords({&word, 1}, sum < 0);
  }

  std::size_t count = a.count();
  Mpz &result = destination(lhs, rhs, reuse, count + sameSign);
  std::span<Limb> out = result.limbs();
  const Limb *x = a.limbs.data(), *y = b.limbs.data();
  if (sameSign) {
    Limb carry;
    if (b.count() <= 1)
      carry = mpn_add_1(out.data(), x, count, b.limb(0));
    else if (x == y)
      carry = mpn_lshift(out.data(), x, count, 1);
    else {
      carry = mpn_add_n(out.data(), x, y, b.count());
      if (count > b.count())
        carry =
            mpn_add_1(&out[b.count()], x + b.count(), count - b.count(), carry);
    }
    out[count] = carry;
    count += carry;
  } else {
    // Magnitude ordering guarantees no final borrow.
    if (b.count() <= 1)
      mpn_sub_1(out.data(), x, count, b.limb(0));
    else {
      Limb borrow = mpn_sub_n(out.data(), x, y, b.count());
      if (count > b.count())
        mpn_sub_1(&out[b.count()], x + b.count(), count - b.count(), borrow);
    }
  }
  result.setMagnitude(count, a.negative);
  return finish<counting>(result, lhs, rhs);
}

export template <Counting counting>
Handle add(Handle lhs, Handle rhs, Reuse reuse) {
  return addOrSubtract<counting, false>(lhs, rhs, reuse);
}

export template <Counting counting>
Handle subtract(Handle lhs, Handle rhs, Reuse reuse) {
  return addOrSubtract<counting, true>(lhs, rhs, reuse);
}

} // namespace rcgmp

extern "C" {

void *rcgmp_mpz_add(void *lhs, void *rhs, std::uint32_t reuse) {
  return rcgmp::binaryEntryPoint<rcgmp::add<rcgmp::Counting::nonatomic>>(
      lhs, rhs, reuse);
}

void *rcgmp_mpz_add_atomic(void *lhs, void *rhs, std::uint32_t reuse) {
  return rcgmp::binaryEntryPoint<rcgmp::add<rcgmp::Counting::atomic>>(lhs, rhs,
                                                                      reuse);
}

void *rcgmp_mpz_sub(void *lhs, void *rhs, std::uint32_t reuse) {
  return rcgmp::binaryEntryPoint<rcgmp::subtract<rcgmp::Counting::nonatomic>>(
      lhs, rhs, reuse);
}

void *rcgmp_mpz_sub_atomic(void *lhs, void *rhs, std::uint32_t reuse) {
  return rcgmp::binaryEntryPoint<rcgmp::subtract<rcgmp::Counting::atomic>>(
      lhs, rhs, reuse);
}
}
