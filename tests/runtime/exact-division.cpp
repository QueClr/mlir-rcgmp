// Compiled exact quotients and their runtime entry points, compared with
// mpz_divexact on multiples of each divisor.
#include "RCGMP/Runtime/Mpz.h"
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <array>
#include <csignal>
#include <ranges>
#include <string>
#include <vector>
import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.allocator;

extern "C" {
void *divsi_exact_local(void *, void *);
void *divsi_exact_atomic(void *, void *);
void *floordivsi_exact_local(void *, void *);
void *edivsi_exact_local(void *, void *);
void *divsi_exact_undefined_local(void *, void *);
void *divsi_exact_undefined_atomic(void *, void *);
void *divsi_exact_lean_local(void *, void *);
void *divsi_exact_lean_atomic(void *, void *);
void *edivsi_exact_lean_atomic(void *, void *);
void *divsi_exact_by_12(void *);
void *divsi_exact_by_minus_8(void *);
void *divsi_exact_by_minus_7(void *);
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

constexpr unsigned word = GMP_NUMB_BITS;

using Quotient = void *(*)(void *, void *);

// The compiled exact quotients of each counting, in every zero-divisor mode.
const std::array<std::vector<Quotient>, 2> compiled{{
    {divsi_exact_local, floordivsi_exact_local, edivsi_exact_local,
     divsi_exact_undefined_local, divsi_exact_lean_local},
    {divsi_exact_atomic, divsi_exact_undefined_atomic, divsi_exact_lean_atomic,
     edivsi_exact_lean_atomic},
}};

Integer product(const Integer &x, const Integer &y) {
  Integer result;
  mpz_mul(result, x, y);
  return result;
}

// Divisors around the immediate bounds and of one to three limbs, ending in
// zero bits that fill no limb, one limb, or more, of either sign. Shifting
// out the zero bits moves set bits into the lowest limb.
std::vector<Integer> divisors() {
  std::vector<Integer> values;
  Integer odd = Integer::powerOfTwo(word + 3) - 5;
  for (const Integer &value :
       {Integer(1), Integer(2), Integer(3), Integer(7), Integer(96),
        Integer(Handle::immediateMax), Integer(Handle::immediateMin),
        Integer::powerOfTwo(word - 3), Integer::powerOfTwo(word) - 1,
        Integer::powerOfTwo(word) + 1, Integer::powerOfTwo(2 * word) - 1,
        Integer::powerOfTwo(3 * word),
        product(Integer(3), Integer::powerOfTwo(word)),
        product(odd, Integer::powerOfTwo(5)),
        product(odd, Integer::powerOfTwo(word + 7))}) {
    values.push_back(value);
    values.push_back(-value);
  }
  return values;
}

// Quotients that make immediate, one-limb, and longer dividends.
std::vector<Integer> quotients() {
  std::vector<Integer> values{Integer(0)};
  for (const Integer &value :
       {Integer(1), Integer(5), Integer(Handle::immediateMax),
        Integer(Handle::immediateMin), Integer::powerOfTwo(word - 3),
        Integer::powerOfTwo(word) + 3, Integer::powerOfTwo(2 * word) - 1,
        Integer::powerOfTwo(4 * word) + 1}) {
    values.push_back(value);
    values.push_back(-value);
  }
  return values;
}

class ExactDivisionTest : public ::testing::TestWithParam<Counting> {
protected:
  void TearDown() override { EXPECT_TRUE(allocations.balanced()); }

  Counting counting = GetParam();
  bool atomic = counting == Counting::atomic;
  AllocationCounter allocations;
};

INSTANTIATE_TEST_SUITE_P(Countings, ExactDivisionTest,
                         ::testing::ValuesIn(countings), countingName);

// Each compiled exact quotient of each multiple of each divisor, with either
// operand shared.
TEST_P(ExactDivisionTest, MatchesGmpOnMultiples) {
  for (const auto &[divisor, quotient, sharing] :
       std::views::cartesian_product(divisors(), quotients(), sharings))
    for (Quotient divide : compiled[atomic])
      checkBinary(divide, mpz_divexact, product(divisor, quotient), divisor,
                  counting, sharing);
}

// Zero is the only multiple of zero, whose lean quotient is zero.
TEST_P(ExactDivisionTest, DividesZeroByLeanZero) {
  void *zero = Handle::immediate(0).pointer();
  for (Quotient divide :
       atomic ? std::vector{divsi_exact_lean_atomic, edivsi_exact_lean_atomic}
              : std::vector{divsi_exact_lean_local})
    EXPECT_EQ(divide(zero, zero), zero);
}

// A unique dividend holds a heap quotient by divisors of one or more limbs,
// odd or even, including those ending in zero limbs, whose dividend moves
// toward its low end.
TEST_P(ExactDivisionTest, ReusesUniqueDividend) {
  Integer quotient = Integer::powerOfTwo(3 * word) + 12345;
  Integer odd = Integer::powerOfTwo(word + 1) + 3;
  auto runtime =
      atomic ? rcgmp_mpz_tdiv_q_exact_atomic : rcgmp_mpz_tdiv_q_exact;
  for (const Integer &divisor :
       {Integer(7), Integer(-12),
        product(Integer(3), Integer::powerOfTwo(word)), odd,
        product(odd, Integer::powerOfTwo(9)),
        product(odd, Integer::powerOfTwo(2 * word + 1))}) {
    SCOPED_TRACE("divisor " + divisor.hex());
    Integer lhs = product(quotient, divisor);
    void *x = toRuntime(lhs);
    void *result = compiled[atomic][0](x, toRuntime(divisor));
    EXPECT_EQ(result, x);
    EXPECT_THAT(result, Represents(quotient));
    rcgmp_mpz_release(result, atomic);
    x = toRuntime(lhs);
    result = runtime(x, toRuntime(divisor), 1);
    EXPECT_EQ(result, x);
    EXPECT_THAT(result, Represents(quotient));
    rcgmp_mpz_release(result, atomic);
  }
}

// Quotients and divisors of 180 limbs, which divide from the low end, and of
// 181, which mpn_tdiv_qr divides unless the divisor ends in a zero limb, by
// odd and even divisors.
TEST_P(ExactDivisionTest, DividesLongOperands) {
  gmp_randstate_t state;
  gmp_randinit_default(state);
  for (const auto &[limbs, shift] : std::views::cartesian_product(
           std::array{180u, 181u}, std::array{0u, 3u, word + 3})) {
    Integer quotient, divisor;
    mpz_urandomb(quotient, state, limbs * word);
    mpz_urandomb(divisor, state, limbs * word - shift);
    mpz_setbit(divisor, 0);
    mpz_mul_2exp(divisor, divisor, shift);
    mpz_setbit(quotient, limbs * word - 1);
    mpz_setbit(divisor, limbs * word - 1);
    SCOPED_TRACE(std::to_string(limbs) + " limbs, shift " +
                 std::to_string(shift));
    checkBinary(compiled[atomic][0], mpz_divexact, product(quotient, divisor),
                -divisor, counting, {});
  }
  gmp_randclear(state);
}

TEST(ExactDivisionDeathTest, TerminatesOnZeroDivisor) {
  void *zero = Handle::immediate(0).pointer();
  // The runtime aborts rather than divide by zero.
  EXPECT_EXIT(divsi_exact_local(zero, zero), ::testing::KilledBySignal(SIGABRT),
              "");
  EXPECT_DEATH(floordivsi_exact_local(zero, zero), "");
  EXPECT_DEATH(rcgmp_mpz_tdiv_q_exact_atomic(zero, zero, 0), "");
  // The runtime also terminates where the quotient is undefined.
  EXPECT_DEATH(divsi_exact_undefined_local(
                   toRuntime(Integer::powerOfTwo(2 * word)), zero),
               "");
  EXPECT_DEATH(
      divsi_exact_lean_local(toRuntime(Integer::powerOfTwo(2 * word)), zero),
      "");
}

// Constant divisors, which LLVM divides by shifts and multiplications.
TEST(ExactDivision, DividesByConstants) {
  for (const Integer &quotient : quotients())
    for (auto [divide, divisor] : {std::pair{divsi_exact_by_12, 12L},
                                   {divsi_exact_by_minus_8, -8L},
                                   {divsi_exact_by_minus_7, -7L}}) {
      Integer lhs = product(quotient, Integer(divisor));
      SCOPED_TRACE(lhs.hex() + " / " + std::to_string(divisor));
      void *result = divide(toRuntime(lhs));
      EXPECT_THAT(result, Represents(quotient));
      rcgmp_mpz_release(result, 0);
    }
}

} // namespace
