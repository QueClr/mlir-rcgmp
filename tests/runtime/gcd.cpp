// Compiled greatest common divisors and their runtime entry points.
#include "RCGMP/Runtime/Mpz.h"
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <array>
#include <ranges>
#include <utility>
#include <vector>
import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.allocator;

extern "C" {
void *gcd_local(void *, void *);
void *gcd_atomic(void *, void *);
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

constexpr unsigned word = GMP_NUMB_BITS;

// Indexed by counting.
constexpr std::array compiled{gcd_local, gcd_atomic};

void MatchesGmp(const std::pair<Integer, Integer> &operands, Counting counting,
                Sharing sharing) {
  const auto &[lhs, rhs] = operands;
  checkBinary(compiled[std::to_underlying(counting)], mpz_gcd, lhs, rhs,
              counting, sharing);
}
FUZZ_TEST(Gcd, MatchesGmp)
    .WithDomains(AnyGcdOperands(), AnyCounting(), AnySharing());

class GcdTest : public ::testing::TestWithParam<Counting> {
protected:
  void TearDown() override { EXPECT_TRUE(allocations.balanced()); }

  static Integer expected(const Integer &lhs, const Integer &rhs) {
    Integer result;
    mpz_gcd(result, lhs, rhs);
    return result;
  }

  Counting counting = GetParam();
  bool atomic = counting == Counting::atomic;
  void *(*apply)(void *, void *) = compiled[atomic];
  AllocationCounter allocations;
};

INSTANTIATE_TEST_SUITE_P(Countings, GcdTest, ::testing::ValuesIn(countings),
                         countingName);

// Zero, one, and the immediate bounds, whose greatest common divisor with zero
// or itself leaves the immediate range, each times a power of two shared with
// the others or not, and a factor shared by every heap value.
TEST_P(GcdTest, MatchesGmpOnBoundaries) {
  std::vector<Integer> values;
  for (long value :
       {0L, 1L, 2L, 6L, Handle::immediateMax, Handle::immediateMin})
    for (bool negative : {false, true})
      values.push_back(negative ? -Integer(value) : Integer(value));
  Integer factor = Integer::powerOfTwo(word + 5) - 1;
  for (unsigned exponent : {word - 3, word, 2 * word + 7}) {
    Integer power = Integer::powerOfTwo(exponent), product;
    mpz_mul(product, power, factor);
    values.push_back(power);
    values.push_back(-product);
  }
  for (const auto &[lhs, rhs, sharing] :
       std::views::cartesian_product(values, values, sharings))
    checkBinary(apply, mpz_gcd, lhs, rhs, counting, sharing);
}

// A unique operand holds a heap result, as the greatest common divisor is no
// larger than either nonzero operand.
TEST_P(GcdTest, ReusesUniqueOperand) {
  Integer odd = Integer::powerOfTwo(2 * word + 3) + 9, x, y;
  mpz_mul_2exp(x, odd, word + 1);
  mpz_mul_ui(y, odd, 3);
  mpz_mul_2exp(y, y, word + 4);
  for (auto [lhs, rhs] :
       {std::pair{x, y}, {-y, x}, {x, Integer(0)}, {-x, Integer(0)}}) {
    for (bool reverse : {false, true}) {
      void *a = toRuntime(lhs), *b = toRuntime(rhs);
      if (!Handle(b).isImmediate())
        rcgmp_mpz_retain(b, atomic);
      AllocationCounter counter;
      void *result = reverse ? apply(b, a) : apply(a, b);
      EXPECT_EQ(result, a);
      EXPECT_EQ(counter.allocations(), 0u);
      EXPECT_THAT(result, Represents(expected(lhs, rhs)));
      rcgmp_mpz_release(result, atomic);
      if (!Handle(b).isImmediate()) {
        EXPECT_THAT(b, Represents(rhs));
        rcgmp_mpz_release(b, atomic);
      }
    }
  }
}

// Two references to one heap integer.
TEST_P(GcdTest, AppliesToAliasedOperand) {
  for (const Integer &value :
       {Integer::powerOfTwo(word + 3), -(Integer::powerOfTwo(2 * word) + 3)}) {
    void *x = toRuntime(value);
    rcgmp_mpz_retain(x, atomic);
    void *result = apply(x, x);
    EXPECT_THAT(result, Represents(expected(value, value)));
    rcgmp_mpz_release(result, atomic);
  }
}

// Operands longer than the scratch on the stack, and a much longer operand
// that is reduced by the shorter one first, with a factor in common.
TEST_P(GcdTest, MatchesGmpOnLongOperands) {
  Integer factor = Integer::powerOfTwo(3 * word) - 1, x, y;
  mpz_mul_2exp(x, factor, 40 * word + 3);
  mpz_add(x, x, factor);
  mpz_mul_2exp(y, factor, 2 * word + 5);
  Integer longer;
  mpz_mul_2exp(longer, x, 30 * word);
  for (const Integer &other : {y, factor, x}) {
    for (Sharing sharing : sharings) {
      checkBinary(apply, mpz_gcd, longer, other, counting, sharing);
      checkBinary(apply, mpz_gcd, other, longer, counting, sharing);
    }
  }
}

TEST_P(GcdTest, AppliesToImmediatesWithoutAllocation) {
  AllocationCounter counter;
  void *result =
      apply(Handle::immediate(-84).pointer(), Handle::immediate(126).pointer());
  EXPECT_THAT(result, Represents(Integer(42)));
  EXPECT_EQ(counter.allocations(), 0u);
}

// The runtime accepts immediates, which compiled code computes inline.
TEST_P(GcdTest, RuntimeAcceptsImmediates) {
  auto runtime = atomic ? rcgmp_mpz_gcd_atomic : rcgmp_mpz_gcd;
  for (auto [lhs, rhs] : {std::pair{0L, 0L},
                          {0L, -5L},
                          {12L, -18L},
                          {Handle::immediateMin, 0L},
                          {Handle::immediateMin, Handle::immediateMin}}) {
    void *result = runtime(Handle::immediate(lhs).pointer(),
                           Handle::immediate(rhs).pointer(), 0);
    EXPECT_THAT(result, Represents(expected(Integer(lhs), Integer(rhs))));
    rcgmp_mpz_release(result, atomic);
  }
}

} // namespace
