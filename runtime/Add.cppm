module;
#include "RCGMP/Runtime/Mpz.h"
#include <gmp.h>

#include <compare>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
export module rcgmp.runtime:add;
import :binary;
import :mpz;

namespace rcgmp {

// Adds a single limb with the given sign to a heap integer, both signs taking
// any negation the operation applies. The heap magnitude is at least every
// immediate one, so a nonzero result has the heap integer's sign, and
// subtracting shortens it by at most one limb.
template <Counting counting>
Handle addSmall(Handle heap, bool reuse, bool negative, Limb small,
                bool smallNegative) {
  Mpz &value = heap.heap();
  std::size_t count = value.count();
  bool sameSign = negative == smallNegative;
  std::size_t required = count + sameSign;
  // A reusable integer grows to hold the result, possibly moving.
  Mpz &result = !reuse ? allocateResult(required)
                : value.capacity >= required
                    ? value
                    : grow(value, required + required / 2 + 1);
  const Limb *in = (reuse ? result : value).limbs().data();
  Limb *out = result.limbs().data();
  if (sameSign) {
    out[count] = mpn_add_1(out, in, count, small);
    count += out[count];
  } else {
    mpn_sub_1(out, in, count, small);
    count -= out[count - 1] == 0;
  }
  if (!reuse)
    release<counting>(heap);
  if (count <= 1)
    if (std::optional<Handle> immediate =
            Handle::immediate(count ? out[0] : 0, negative)) {
      deallocate<counting>(result);
      return *immediate;
    }
  result.setMagnitude(count, negative);
  return Handle(result);
}

template <Counting counting, bool subtracting>
Handle addOrSubtract(Handle lhs, Handle rhs, Reuse reuse) {
  // A heap integer and a nonzero immediate, as in arithmetic with small
  // constants.
  if (lhs.isImmediate() != rhs.isImmediate()) {
    bool heapIsLhs = rhs.isImmediate();
    std::intptr_t small = (heapIsLhs ? rhs : lhs).value();
    if (small) {
      Handle heap = heapIsLhs ? lhs : rhs;
      return addSmall<counting>(
          heap, heapIsLhs ? reuse.lhs : reuse.rhs,
          heap.heap().negative() != (subtracting && !heapIsLhs),
          small < 0 ? -static_cast<Limb>(small) : static_cast<Limb>(small),
          (small < 0) != (subtracting && heapIsLhs));
    }
  }
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
  Mpz &result =
      growingDestination(lhs, rhs, reuse, count + sameSign, left, right);
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
