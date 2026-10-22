module;
#include "RCGMP/Runtime/Mpz.h"
#include <gmp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
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

enum class Logic { conjunction, disjunction, exclusiveDisjunction };

constexpr bool apply(Logic logic, bool a, bool b) {
  switch (logic) {
  case Logic::conjunction:
    return a && b;
  case Logic::disjunction:
    return a || b;
  case Logic::exclusiveDisjunction:
    return a != b;
  }
  std::unreachable();
}

constexpr Limb apply(Logic logic, Limb a, Limb b) {
  switch (logic) {
  case Logic::conjunction:
    return a & b;
  case Logic::disjunction:
    return a | b;
  case Logic::exclusiveDisjunction:
    return a ^ b;
  }
  std::unreachable();
}

// The limbs that hold a result. Above its limbs, the two's complement of an
// operand is all zeros if it is nonnegative and all ones if it is negative,
// which fixes the result of AND and of OR there, respectively. Converting a
// negative result back to magnitude may otherwise carry into a new limb.
std::size_t resultCount(Logic logic, const Magnitude &left,
                        const Magnitude &right, bool negative) {
  if (logic != Logic::exclusiveDisjunction) {
    // The sign of an operand whose high limbs fix the result.
    bool fixing = logic == Logic::disjunction;
    if (left.negative == fixing && right.negative == fixing)
      return std::min(left.count(), right.count());
    if (left.negative == fixing)
      return left.count();
    if (right.negative == fixing)
      return right.count();
  }
  return std::max(left.count(), right.count()) + negative;
}

template <Counting counting, Logic logic>
Handle bitwise(Handle lhs, Handle rhs, Reuse reuse) {
  Magnitude left(lhs), right(rhs);
  // An operand's sign is its infinite sign bit.
  bool negative = apply(logic, left.negative, right.negative);
  std::size_t count = resultCount(logic, left, right, negative);
  if (!count)
    return consume<counting>(lhs, rhs, Handle::immediate(0));

  Mpz &result = destination(lhs, rhs, reuse, count);
  std::span<Limb> out = result.limbs();
  if (!left.negative && !right.negative) {
    std::size_t common = std::min(left.count(), right.count());
    if (common) {
      const Limb *x = left.limbs.data(), *y = right.limbs.data();
      switch (logic) {
      case Logic::conjunction:
        mpn_and_n(out.data(), x, y, common);
        break;
      case Logic::disjunction:
        mpn_ior_n(out.data(), x, y, common);
        break;
      case Logic::exclusiveDisjunction:
        mpn_xor_n(out.data(), x, y, common);
        break;
      }
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
      Limb limb = apply(logic, a, b);
      out[i] = negative ? resultNegation(limb) : limb;
    }
  }
  result.setMagnitude(count, negative);
  return finish<counting>(result, lhs, rhs);
}

export template <Counting counting>
Handle bitwiseAnd(Handle lhs, Handle rhs, Reuse reuse) {
  return bitwise<counting, Logic::conjunction>(lhs, rhs, reuse);
}

export template <Counting counting>
Handle bitwiseOr(Handle lhs, Handle rhs, Reuse reuse) {
  return bitwise<counting, Logic::disjunction>(lhs, rhs, reuse);
}

export template <Counting counting>
Handle bitwiseXor(Handle lhs, Handle rhs, Reuse reuse) {
  return bitwise<counting, Logic::exclusiveDisjunction>(lhs, rhs, reuse);
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

void *rcgmp_mpz_or(void *lhs, void *rhs, std::uint32_t reuse) {
  return rcgmp::binaryEntryPoint<rcgmp::bitwiseOr<rcgmp::Counting::nonatomic>>(
      lhs, rhs, reuse);
}

void *rcgmp_mpz_or_atomic(void *lhs, void *rhs, std::uint32_t reuse) {
  return rcgmp::binaryEntryPoint<rcgmp::bitwiseOr<rcgmp::Counting::atomic>>(
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
