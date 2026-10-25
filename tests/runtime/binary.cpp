// Compiled subtraction, monus, multiplication, AND, XOR, and OR.
#include "RCGMP/Runtime/Mpz.h"
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>
import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.allocator;

extern "C" {
void *sub_local(void *, void *);
void *sub_atomic(void *, void *);
void *monus_local(void *, void *);
void *monus_atomic(void *, void *);
void *mul_local(void *, void *);
void *mul_atomic(void *, void *);
void *and_local(void *, void *);
void *and_atomic(void *, void *);
void *xor_local(void *, void *);
void *xor_atomic(void *, void *);
void *or_local(void *, void *);
void *or_atomic(void *, void *);
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

enum class Operator {
  subtract,
  monus,
  multiply,
  bitwiseAnd,
  bitwiseXor,
  bitwiseOr
};

struct Operation {
  std::string_view name;
  // Indexed by counting.
  std::array<void *(*)(void *, void *), 2> compiled;
  Oracle oracle;
};

constexpr std::array<Operation, 6> operations{{
    {"sub", {sub_local, sub_atomic}, mpz_sub},
    {"monus", {monus_local, monus_atomic}, gmpMonus},
    {"mul", {mul_local, mul_atomic}, mpz_mul},
    {"and", {and_local, and_atomic}, mpz_and},
    {"xor", {xor_local, xor_atomic}, mpz_xor},
    {"or", {or_local, or_atomic}, mpz_ior},
}};

constexpr std::array operators{Operator::subtract,   Operator::monus,
                               Operator::multiply,   Operator::bitwiseAnd,
                               Operator::bitwiseXor, Operator::bitwiseOr};

const Operation &operation(Operator op) {
  return operations[std::to_underlying(op)];
}

template <typename Sink> void AbslStringify(Sink &sink, Operator op) {
  sink.Append(operation(op).name);
}

void MatchesGmp(Operator op, const std::pair<Integer, Integer> &operands,
                Counting counting, Sharing sharing) {
  const auto &[lhs, rhs] = operands;
  checkBinary(operation(op).compiled[std::to_underlying(counting)],
              operation(op).oracle, lhs, rhs, counting, sharing);
}
FUZZ_TEST(Binary, MatchesGmp)
    .WithDomains(fuzztest::ElementOf(std::vector(operators.begin(),
                                                 operators.end())),
                 AnyOperands(), AnyCounting(), AnySharing());

class BinaryTest
    : public ::testing::TestWithParam<std::tuple<Operator, Counting>> {
protected:
  void TearDown() override { EXPECT_TRUE(allocations.balanced()); }

  void check(const Integer &lhs, const Integer &rhs, Sharing sharing = {}) {
    checkBinary(apply, oracle, lhs, rhs, counting, sharing);
  }
  Integer expected(const Integer &lhs, const Integer &rhs) const {
    Integer result;
    oracle(result, lhs, rhs);
    return result;
  }

  Operator op = std::get<Operator>(GetParam());
  Counting counting = std::get<Counting>(GetParam());
  bool atomic = counting == Counting::atomic;
  void *(*apply)(void *, void *) = operation(op).compiled[atomic];
  Oracle oracle = operation(op).oracle;
  AllocationCounter allocations;
};

std::string parameterName(
    const ::testing::TestParamInfo<std::tuple<Operator, Counting>> &info) {
  auto [op, counting] = info.param;
  return std::string(operation(op).name) + "_" +
         std::string(rcgmp::name(counting));
}

INSTANTIATE_TEST_SUITE_P(Operations, BinaryTest,
                         ::testing::Combine(::testing::ValuesIn(operators),
                                            ::testing::ValuesIn(countings)),
                         parameterName);

TEST_P(BinaryTest, MatchesGmpOnBoundaries) {
  // Products of operands near the square root of the immediate bound land on
  // either side of it.
  long root = 1L << (sizeof(std::uintptr_t) * CHAR_BIT / 2 - 1);
  std::vector<Integer> boundaries;
  for (long value : {0L, 1L, -1L, 2L, -2L, root - 1, root, -root,
                     Handle::immediateMin, Handle::immediateMax})
    boundaries.emplace_back(value);
  boundaries.push_back(Integer(Handle::immediateMin) - 1);
  boundaries.push_back(Integer(Handle::immediateMax) + 1);
  for (const auto &[lhs, rhs, sharing] :
       std::views::cartesian_product(boundaries, boundaries, sharings))
    check(lhs, rhs, sharing);
}

// Long runs of ones, negative sign extension, and an extra result limb when a
// negative bitwise result is converted back to magnitude.
TEST_P(BinaryTest, MatchesGmpOnLongRuns) {
  Integer minusTwo(-2);
  for (unsigned bits = GMP_NUMB_BITS; bits <= 4 * GMP_NUMB_BITS; ++bits) {
    Integer ones = -(Integer::powerOfTwo(bits) - 1);
    for (Sharing sharing : sharings) {
      check(ones, minusTwo, sharing);
      check(minusTwo, ones, sharing);
    }
  }
}

// Either operand position can reuse the heap allocation, with either sign,
// unless the result is immediate, as a monus that stops at zero is. AND uses
// -1, so that the result remains on the heap.
TEST_P(BinaryTest, ReusesHeapOperand) {
  long small = op == Operator::bitwiseAnd ? -1 : 3;
  Integer rhs(small);
  for (bool negative : {false, true}) {
    Integer lhs = Integer::powerOfTwo(130) + 17;
    if (negative)
      lhs = -lhs;
    for (bool reverse : {false, true}) {
      Integer value = reverse ? expected(rhs, lhs) : expected(lhs, rhs);
      if (value.fitsImmediate())
        continue;
      void *x = toRuntime(lhs), *y = Handle::immediate(small).pointer();
      AllocationCounter counter;
      void *result = reverse ? apply(y, x) : apply(x, y);
      EXPECT_EQ(result, x);
      EXPECT_EQ(counter.allocations(), 0u);
      EXPECT_THAT(result, Represents(value));
      rcgmp_mpz_release(result, atomic);
    }
  }
}

// Two references to the same heap operand, optionally with an external
// alias. Multiplication exercises the disjoint output that mpn_sqr requires.
TEST_P(BinaryTest, AppliesToAliasedOperand) {
  Integer value = -(Integer::powerOfTwo(130) + 17);
  for (bool shared : {false, true}) {
    void *x = toRuntime(value);
    rcgmp_mpz_retain(x, atomic);
    if (shared)
      rcgmp_mpz_retain(x, atomic);
    void *result = apply(x, x);
    EXPECT_THAT(result, Represents(expected(value, value)));
    if (shared) {
      EXPECT_THAT(x, Represents(value));
      rcgmp_mpz_release(x, atomic);
    }
    rcgmp_mpz_release(result, atomic);
  }
}

TEST_P(BinaryTest, AppliesToImmediatesWithoutAllocation) {
  AllocationCounter counter;
  void *result =
      apply(Handle::immediate(-7).pointer(), Handle::immediate(9).pointer());
  EXPECT_TRUE(Handle(result).isImmediate());
  EXPECT_THAT(result, Represents(expected(Integer(-7), Integer(9))));
  EXPECT_EQ(counter.allocations(), 0u);
}

// Bitwise operations on operands long enough for every part of their limb
// loops: blocks of eight limbs, and blocks of four, two and one after them.
class BitwiseTest : public BinaryTest {};

INSTANTIATE_TEST_SUITE_P(
    Operations, BitwiseTest,
    ::testing::Combine(::testing::Values(Operator::bitwiseAnd,
                                         Operator::bitwiseXor,
                                         Operator::bitwiseOr),
                       ::testing::ValuesIn(countings)),
    parameterName);

// Limb counts up to two blocks of eight and three more limbs, which take every
// combination of the shorter blocks.
constexpr auto longCounts = std::views::iota(std::size_t{1}, std::size_t{20});

// An integer of count words, which seed shifts among arbitrary words and runs
// of zeros and ones. The top bit is set, so even one word is not immediate.
Integer longInteger(std::size_t count, std::size_t seed,
                    bool negative = false) {
  std::vector<std::uintptr_t> words(count);
  for (auto [index, word] : std::views::enumerate(words)) {
    std::size_t position = index + seed;
    word = position % 3 == 0   ? 0
           : position % 3 == 1 ? ~std::uintptr_t{0}
                               : position * 0x9e3779b97f4a7c15;
  }
  words.back() |= std::uintptr_t{1} << (GMP_NUMB_BITS - 1);
  return Integer::fromWords(words, negative);
}

// A unique heap integer holding value, with room for capacity limbs. Like
// those of earlier results, its limbs above the magnitude are nonzero.
void *roomy(const Integer &value, std::size_t capacity) {
  std::vector<std::uintptr_t> words = value.words();
  rcgmp::Mpz &heap = rcgmp::allocate(capacity);
  std::ranges::fill(heap.limbs(), ~rcgmp::Limb{0});
  std::ranges::copy(words, heap.limbs().begin());
  heap.setMagnitude(words.size(), value.negative());
  return Handle(heap).pointer();
}

// Every pair of lengths, either operand longer, of either sign and sharing.
// Compiled code passes unique operands for reuse, which the result replaces
// when it has room, so the result is computed in place in either operand and
// in new storage. An operand is also combined with itself.
TEST_P(BitwiseTest, MatchesGmpOnLongOperands) {
  for (auto [lhsCount, rhsCount] :
       std::views::cartesian_product(longCounts, longCounts))
    for (auto [lhsNegative, rhsNegative] : std::views::cartesian_product(
             std::array{false, true}, std::array{false, true})) {
      Integer lhs = longInteger(lhsCount, 1, lhsNegative),
              rhs = longInteger(rhsCount, 2, rhsNegative);
      for (Sharing sharing : sharings)
        check(lhs, rhs, sharing);
    }
  for (std::size_t count : longCounts) {
    Integer value = longInteger(count, 1);
    void *x = toRuntime(value);
    rcgmp_mpz_retain(x, atomic);
    void *result = apply(x, x);
    EXPECT_THAT(result, Represents(expected(value, value)));
    rcgmp_mpz_release(result, atomic);
  }
}

// Operands that agree above their low limbs, for XOR and OR, or complement
// each other there, for AND, so that AND and XOR cancel the high limbs. The
// stripped result is zero, an immediate, or a shorter heap integer.
TEST_P(BitwiseTest, StripsCancelledLimbs) {
  for (std::size_t count : {9, 10, 17}) {
    std::vector<std::pair<std::size_t, Integer>> lows{
        {0, Integer(0)},
        {1, Integer(5)},
        {1, Integer::powerOfTwo(GMP_NUMB_BITS - 1) + 5},
        {2, longInteger(2, 3)},
        {8, longInteger(8, 3)},
        {count - 1, longInteger(count - 1, 3)}};
    for (const auto &[kept, low] : lows) {
      // high * 2^shift + low, for the high part of an operand.
      auto join = [shift = kept * GMP_NUMB_BITS](const Integer &high,
                                                 const Integer &low) {
        Integer value;
        mpz_mul_2exp(value, high, shift);
        mpz_add(value, value, low);
        return value;
      };
      Integer high = longInteger(count - kept, 1), lhs, rhs;
      if (op == Operator::bitwiseAnd) {
        // The complement of high within its limbs, and all ones below.
        Integer complement, ones;
        mpz_com(complement, high);
        mpz_fdiv_r_2exp(complement, complement, (count - kept) * GMP_NUMB_BITS);
        mpz_fdiv_r_2exp(ones, Integer(-1), kept * GMP_NUMB_BITS);
        lhs = join(high, low);
        rhs = join(complement, ones);
      } else {
        Integer other = kept ? longInteger(kept, 4) : Integer(0), mixed;
        mpz_xor(mixed, other, low);
        lhs = join(high, other);
        rhs = join(high, mixed);
      }
      if (op != Operator::bitwiseOr)
        EXPECT_EQ(expected(lhs, rhs), low);
      for (Sharing sharing : sharings)
        check(lhs, rhs, sharing);
    }
  }
}

// A unique operand with room holds the result in place, in either position
// and whichever operand is longer. The other operand is shared, so that only
// the roomy one is passed for reuse.
TEST_P(BitwiseTest, ReusesLongUniqueOperand) {
  constexpr std::array<std::size_t, 3> counts{8, 9, 17};
  for (auto [lhsCount, rhsCount, reuseLhs, lhsNegative, rhsNegative] :
       std::views::cartesian_product(counts, counts, std::array{true, false},
                                     std::array{false, true},
                                     std::array{false, true})) {
    Integer lhs = longInteger(lhsCount, 1, lhsNegative),
            rhs = longInteger(rhsCount, 2, rhsNegative);
    Integer value = expected(lhs, rhs);
    if (value.fitsImmediate())
      continue;
    // A negative result may need a limb more than both operands.
    std::size_t capacity = std::max(lhsCount, rhsCount) + 1;
    void *x = reuseLhs ? roomy(lhs, capacity) : toRuntime(lhs);
    void *y = reuseLhs ? toRuntime(rhs) : roomy(rhs, capacity);
    void *shared = reuseLhs ? y : x;
    rcgmp_mpz_retain(shared, atomic);
    AllocationCounter counter;
    void *result = apply(x, y);
    EXPECT_EQ(result, reuseLhs ? x : y);
    EXPECT_EQ(counter.allocations(), 0u);
    EXPECT_THAT(result, Represents(value))
        << "lhs = " << lhs << ", rhs = " << rhs;
    EXPECT_THAT(shared, Represents(reuseLhs ? rhs : lhs));
    rcgmp_mpz_release(shared, atomic);
    rcgmp_mpz_release(result, atomic);
  }
}

} // namespace
