// Compiled division, its runtime entry points, and zero divisors.
#include "RCGMP/Runtime/Mpz.h"
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <array>
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
#define RCGMP_ENTRY(result, name)                                              \
  result name##_local(void *, void *);                                         \
  result name##_atomic(void *, void *);
RCGMP_ENTRY(void *, divsi)
RCGMP_ENTRY(void *, remsi)
RCGMP_ENTRY(void *, floordivsi)
RCGMP_ENTRY(void *, floormodsi)
RCGMP_ENTRY(rcgmp_mpz_qr, divremsi)
RCGMP_ENTRY(rcgmp_mpz_qr, floordivmodsi)
#undef RCGMP_ENTRY
void *divsi_assume(void *, void *);
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

constexpr unsigned word = GMP_NUMB_BITS;

using GmpDivision = void (*)(mpz_ptr, mpz_ptr, mpz_srcptr, mpz_srcptr);

struct Rounding {
  std::string_view name;
  // Indexed by counting.
  std::array<void *(*)(void *, void *), 2> quotient, remainder;
  std::array<rcgmp_mpz_qr (*)(void *, void *), 2> both;
  GmpDivision oracle;
};

constexpr std::array<Rounding, 2> roundings{{
    {"truncate",
     {divsi_local, divsi_atomic},
     {remsi_local, remsi_atomic},
     {divremsi_local, divremsi_atomic},
     mpz_tdiv_qr},
    {"floor",
     {floordivsi_local, floordivsi_atomic},
     {floormodsi_local, floormodsi_atomic},
     {floordivmodsi_local, floordivmodsi_atomic},
     mpz_fdiv_qr},
}};

// Applies each compiled division of a rounding to lhs and rhs, the lhs shared
// if the caller keeps an extra reference, and compares the results with GMP.
void checkDivision(const Rounding &rounding, const Integer &lhs,
                   const Integer &rhs, Counting counting, bool shared) {
  SCOPED_TRACE(std::string(rounding.name) + ": " + lhs.hex() + " / " +
               rhs.hex() + (shared ? " (shared)" : ""));
  bool atomic = counting == Counting::atomic;
  Integer quotient, remainder;
  rounding.oracle(quotient, remainder, lhs, rhs);
  auto apply = [&](auto divide) {
    void *x = toRuntime(lhs);
    if (shared)
      rcgmp_mpz_retain(x, atomic);
    auto result = divide(x, toRuntime(rhs));
    if (shared) {
      EXPECT_THAT(x, Represents(lhs));
      rcgmp_mpz_release(x, atomic);
    }
    return result;
  };
  void *q = apply(rounding.quotient[atomic]);
  EXPECT_THAT(q, Represents(quotient));
  rcgmp_mpz_release(q, atomic);
  void *r = apply(rounding.remainder[atomic]);
  EXPECT_THAT(r, Represents(remainder));
  rcgmp_mpz_release(r, atomic);
  auto [bothQ, bothR] = apply(rounding.both[atomic]);
  EXPECT_THAT(bothQ, Represents(quotient));
  EXPECT_THAT(bothR, Represents(remainder));
  rcgmp_mpz_release(bothQ, atomic);
  rcgmp_mpz_release(bothR, atomic);
}

class DivisionTest
    : public ::testing::TestWithParam<std::tuple<std::size_t, Counting>> {
protected:
  void TearDown() override { EXPECT_TRUE(allocations.balanced()); }

  const Rounding &rounding = roundings[std::get<std::size_t>(GetParam())];
  Counting counting = std::get<Counting>(GetParam());
  bool atomic = counting == Counting::atomic;
  AllocationCounter allocations;
};

std::string parameterName(
    const ::testing::TestParamInfo<std::tuple<std::size_t, Counting>> &info) {
  auto [rounding, counting] = info.param;
  return std::string(roundings[rounding].name) + "_" +
         std::string(rcgmp::name(counting));
}

INSTANTIATE_TEST_SUITE_P(Roundings, DivisionTest,
                         ::testing::Combine(::testing::Values(0, 1),
                                            ::testing::ValuesIn(countings)),
                         parameterName);

// Immediates around their bounds, and one to three limbs, of either sign:
// dividends shorter than, as long as, and longer than divisors of one or more
// limbs, with exact and inexact quotients.
TEST_P(DivisionTest, MatchesGmpOnBoundaries) {
  std::vector<Integer> values;
  for (long value : {0L, 1L, -1L, 2L, -2L, 7L, -7L, Handle::immediateMin,
                     Handle::immediateMax})
    values.emplace_back(value);
  for (unsigned exponent : {word - 3, word, 2 * word, 3 * word})
    for (long delta : {-1L, 0L, 1L}) {
      Integer value = Integer::powerOfTwo(exponent) + delta;
      values.push_back(value);
      values.push_back(-value);
    }
  for (const auto &[lhs, rhs, shared] :
       std::views::cartesian_product(values, values, std::array{false, true}))
    if (rhs.sign())
      checkDivision(rounding, lhs, rhs, counting, shared);
}

// Only the quotient of the immediate minimum by -1 leaves the immediate range.
TEST_P(DivisionTest, DividesImmediateMinimumByMinusOne) {
  checkDivision(rounding, Integer(Handle::immediateMin), Integer(-1), counting,
                false);
}

// Floor division of -(2^(3W) - 2^W + 1) by 2^W rounds the truncated quotient
// 2^(2W) - 1, whose limbs are all ones, to -2^(2W), which carries into a new
// limb.
TEST_P(DivisionTest, FloorQuotientCarriesIntoNewLimb) {
  Integer lhs = Integer::powerOfTwo(3 * word);
  mpz_sub(lhs, lhs, Integer::powerOfTwo(word));
  lhs = -(lhs + 1);
  checkDivision(rounding, lhs, Integer::powerOfTwo(word), counting, false);
}

// A unique dividend holds a heap remainder, and a heap quotient by a one-limb
// divisor.
TEST_P(DivisionTest, ReusesUniqueDividend) {
  Integer lhs = Integer::powerOfTwo(3 * word) + 12345;
  for (const auto &[divisor, quotient] :
       {std::pair{Integer::powerOfTwo(word + 1) + 3, false},
        {Integer(7), true}}) {
    void *x = toRuntime(lhs);
    void *result = (quotient ? rounding.quotient : rounding.remainder)[atomic](
        x, toRuntime(divisor));
    EXPECT_EQ(result, x);
    Integer q, r;
    rounding.oracle(q, r, lhs, divisor);
    EXPECT_THAT(result, Represents(quotient ? q : r));
    rcgmp_mpz_release(result, atomic);
  }
}

TEST(DivisionDeathTest, TerminatesOnZeroDivisor) {
  void *zero = Handle::immediate(0).pointer();
  EXPECT_DEATH(divsi_local(Handle::immediate(5).pointer(), zero), "");
  EXPECT_DEATH(
      floormodsi_atomic(toRuntime(Integer::powerOfTwo(2 * word)), zero), "");
  EXPECT_DEATH(divremsi_local(Handle::immediate(-5).pointer(), zero), "");
  EXPECT_DEATH(rcgmp_mpz_fdiv_r(Handle::immediate(5).pointer(), zero, 0), "");
  // The runtime still checks a divisor assumed nonzero.
  EXPECT_DEATH(divsi_assume(toRuntime(Integer::powerOfTwo(2 * word)), zero),
               "");
}

TEST(Division, AssumesNonzeroDivisor) {
  void *result = divsi_assume(Handle::immediate(-7).pointer(),
                              Handle::immediate(2).pointer());
  EXPECT_THAT(result, Represents(Integer(-3)));
}

} // namespace
