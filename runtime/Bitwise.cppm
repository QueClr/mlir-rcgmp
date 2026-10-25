module;
#include "RCGMP/Runtime/Mpz.h"
#include <gmp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
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

// Applies step to each index below count, in blocks of eight and then in at
// most one block each of four, two and one, so that compilers vectorize the
// steps on adjacent limbs. It is inlined into its callers, whose restrict
// qualifiers vectorization needs.
template <typename Step>
[[gnu::always_inline]] inline void forEachLimb(std::size_t count, Step step) {
  std::size_t i = 0;
  for (; i + 8 <= count; i += 8)
    for (std::size_t j = 0; j < 8; ++j)
      step(i + j);
  // count - i is count % 8 here, so its bits select the shorter blocks.
  for (std::size_t width : {4, 2, 1})
    if (count & width) {
      for (std::size_t j = 0; j < width; ++j)
        step(i + j);
      i += width;
    }
}

// Combines count limbs of x and y into out, which overlaps neither. x and y
// may be the same (x op x), as only out is written. The vectorized loop
// combines two or more limbs per instruction, where mpn_and_n, mpn_ior_n and
// mpn_xor_n combine one.
template <Logic logic>
void combine(Limb *__restrict out, const Limb *__restrict x,
             const Limb *__restrict y, std::size_t count) {
  forEachLimb(count, [&](std::size_t i) { out[i] = apply(logic, x[i], y[i]); });
}

// The same with x as out, for a result that replaces an operand. combine
// cannot compute it, as its restrict qualifiers forbid out to be x, and a loop
// with unrestricted sources would be vectorized only behind a runtime overlap
// test, which fails when out is a source.
template <Logic logic>
void combineInPlace(Limb *__restrict out, const Limb *__restrict y,
                    std::size_t count) {
  forEachLimb(count,
              [&](std::size_t i) { out[i] = apply(logic, out[i], y[i]); });
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

// Combines a nonnegative heap integer with an immediate that changes only its
// least significant limb: a nonnegative one, or a negative one under AND,
// whose higher bits are ones. AND with a nonnegative immediate leaves at most
// its bits. Otherwise the heap integer keeps its higher bits, so its length
// and its place outside the immediate range.
template <Counting counting, Logic logic>
Handle bitwiseSmall(Handle heap, bool reuse, std::intptr_t small) {
  Mpz &value = heap.heap();
  Limb low = apply(logic, value.limbs()[0], static_cast<Limb>(small));
  if (logic == Logic::conjunction && small >= 0) {
    release<counting>(heap);
    return Handle::immediate(static_cast<std::intptr_t>(low));
  }
  std::size_t count = value.count();
  Mpz &result = reuse ? value : allocateResult(count);
  if (&result != &value) {
    std::ranges::copy(value.magnitude(), result.limbs().begin());
    result.setMagnitude(count, false);
    release<counting>(heap);
  }
  result.limbs()[0] = low;
  return Handle(result);
}

template <Counting counting, Logic logic>
Handle bitwise(Handle lhs, Handle rhs, Reuse reuse) {
  if (lhs.isImmediate() != rhs.isImmediate()) {
    bool heapIsLhs = rhs.isImmediate();
    Handle heap = heapIsLhs ? lhs : rhs;
    std::intptr_t small = (heapIsLhs ? rhs : lhs).value();
    if (!heap.heap().negative() && (small >= 0 || logic == Logic::conjunction))
      return bitwiseSmall<counting, logic>(
          heap, heapIsLhs ? reuse.lhs : reuse.rhs, small);
  }
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
    const Limb *x = left.limbs.data(), *y = right.limbs.data();
    // A reused operand is uniquely owned, so it is not also the other one.
    // Each operation is commutative.
    if (out.data() == x)
      combineInPlace<logic>(out.data(), y, common);
    else if (out.data() == y)
      combineInPlace<logic>(out.data(), x, common);
    else
      combine<logic>(out.data(), x, y, common);
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

// Tests a bit of a borrowed integer's two's complement. That of a negative
// integer -m is ~(m - 1), whose limb at an index is the complement of the limb
// of m there, less one exactly when every limb below it is zero.
export bool testBit(Handle value, std::uintptr_t index) {
  Magnitude magnitude(value);
  std::size_t position = index / GMP_NUMB_BITS;
  if (position >= magnitude.count())
    return magnitude.negative;
  Limb limb = magnitude.limbs[position];
  if (magnitude.negative)
    limb = mpn_zero_p(magnitude.limbs.data(), position) ? -limb : ~limb;
  return limb >> index % GMP_NUMB_BITS & 1;
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

std::int32_t rcgmp_mpz_tstbit(void *value, std::uintptr_t index) {
  return rcgmp::testBit(rcgmp::Handle(value), index);
}

// Testing a bit reads no reference count, so atomicity makes no difference.
std::int32_t rcgmp_mpz_tstbit_atomic(void *value, std::uintptr_t index) {
  return rcgmp_mpz_tstbit(value, index);
}
}
