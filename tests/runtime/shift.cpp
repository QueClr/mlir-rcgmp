// Compiled left and arithmetic right shifts.
#include "RCGMP/Runtime/Mpz.h"
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
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
void *shl_local(void *, std::uint64_t);
void *shl_atomic(void *, std::uint64_t);
void *shr_local(void *, std::uint64_t);
void *shr_atomic(void *, std::uint64_t);
void *shl_narrow(void *, std::uint8_t);
void *shr_wide(void *, unsigned __int128);
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

constexpr unsigned word = GMP_NUMB_BITS;

enum class Direction { left, right };

struct Shift {
  std::string_view name;
  // Indexed by counting.
  std::array<void *(*)(void *, std::uint64_t), 2> compiled;
  void (*oracle)(mpz_ptr, mpz_srcptr, mp_bitcnt_t);
};

constexpr std::array<Shift, 2> shifts{{
    {"shl", {shl_local, shl_atomic}, mpz_mul_2exp},
    {"shr", {shr_local, shr_atomic}, mpz_fdiv_q_2exp},
}};

constexpr std::array directions{Direction::left, Direction::right};

const Shift &shift(Direction direction) {
  return shifts[std::to_underlying(direction)];
}

// Applies a compiled shift to an operand, shared if the caller keeps an extra
// reference, and compares the result with GMP.
void checkShift(Direction direction, const Integer &value, std::uint64_t amount,
                Counting counting, bool shared) {
  bool atomic = counting == Counting::atomic;
  void *x = toRuntime(value);
  if (shared)
    rcgmp_mpz_retain(x, atomic);
  Integer expected;
  shift(direction).oracle(expected, value, amount);
  void *result = shift(direction).compiled[atomic](x, amount);
  EXPECT_THAT(result, Represents(expected))
      << shift(direction).name << " " << value << " by " << amount;
  if (shared) {
    EXPECT_THAT(x, Represents(value));
    rcgmp_mpz_release(x, atomic);
  }
  rcgmp_mpz_release(result, atomic);
}

// Amounts around whole limbs, up to five of them.
fuzztest::Domain<std::uint64_t> AnyAmount() {
  return fuzztest::InRange<std::uint64_t>(0, 5 * word);
}

void ShiftLeftMatchesGmp(const Integer &value, std::uint64_t amount,
                         Counting counting, bool shared) {
  checkShift(Direction::left, value, amount, counting, shared);
}
FUZZ_TEST(Shift, ShiftLeftMatchesGmp)
    .WithDomains(AnyInteger(), AnyAmount(), AnyCounting(),
                 fuzztest::Arbitrary<bool>());

// Shifting right accepts any amount.
void ShiftRightMatchesGmp(const Integer &value, std::uint64_t amount,
                          Counting counting, bool shared) {
  checkShift(Direction::right, value, amount, counting, shared);
}
FUZZ_TEST(Shift, ShiftRightMatchesGmp)
    .WithDomains(AnyInteger(),
                 fuzztest::OneOf(AnyAmount(),
                                 fuzztest::Arbitrary<std::uint64_t>()),
                 AnyCounting(), fuzztest::Arbitrary<bool>());

class ShiftTest
    : public ::testing::TestWithParam<std::tuple<Direction, Counting>> {
protected:
  void TearDown() override { EXPECT_TRUE(allocations.balanced()); }

  Integer expected(const Integer &value, std::uint64_t amount) const {
    Integer result;
    shift(direction).oracle(result, value, amount);
    return result;
  }

  Direction direction = std::get<Direction>(GetParam());
  Counting counting = std::get<Counting>(GetParam());
  bool atomic = counting == Counting::atomic;
  void *(*apply)(void *, std::uint64_t) = shift(direction).compiled[atomic];
  AllocationCounter allocations;
};

std::string parameterName(
    const ::testing::TestParamInfo<std::tuple<Direction, Counting>> &info) {
  auto [direction, counting] = info.param;
  return std::string(shift(direction).name) + "_" +
         std::string(rcgmp::name(counting));
}

INSTANTIATE_TEST_SUITE_P(Shifts, ShiftTest,
                         ::testing::Combine(::testing::ValuesIn(directions),
                                            ::testing::ValuesIn(countings)),
                         parameterName);

// Immediates near the bounds overflow when shifted left by one, and negative
// integers round toward negative infinity: -1 stays -1 for any amount. Amounts
// around whole limbs move limbs with and without a bit shift.
TEST_P(ShiftTest, MatchesGmpOnBoundaries) {
  std::vector<Integer> values;
  for (long value : {0L, 1L, -1L, 3L, -3L, Handle::immediateMin,
                     Handle::immediateMax, Handle::immediateMin / 2,
                     Handle::immediateMax / 2, Handle::immediateMax / 2 + 1})
    values.emplace_back(value);
  for (unsigned exponent : {word - 3, word, 2 * word})
    for (long delta : {-1L, 0L, 1L}) {
      Integer value = Integer::powerOfTwo(exponent) + delta;
      values.push_back(value);
      values.push_back(-value);
    }
  std::vector<std::uint64_t> amounts{
      0,        1,    2,        word - 4,     word - 3, word - 2,
      word - 1, word, word + 1, 2 * word - 1, 2 * word, 3 * word + 1};
  for (const auto &[value, amount, shared] :
       std::views::cartesian_product(values, amounts, std::array{false, true}))
    checkShift(direction, value, amount, counting, shared);
}

// A unique heap operand of either sign with room for the result holds it.
TEST_P(ShiftTest, ReusesUniqueHeapOperand) {
  for (bool negative : {false, true}) {
    Integer value = Integer::powerOfTwo(130) + 17;
    if (negative)
      value = -value;
    void *x = toRuntime(value);
    AllocationCounter counter;
    void *result = apply(x, 1);
    EXPECT_EQ(result, x);
    EXPECT_EQ(counter.allocations(), 0u);
    EXPECT_THAT(result, Represents(expected(value, 1)));
    rcgmp_mpz_release(result, atomic);
  }
}

TEST_P(ShiftTest, AppliesToImmediatesWithoutAllocation) {
  for (long value : {-7L, 0L, 9L}) {
    AllocationCounter counter;
    void *result = apply(Handle::immediate(value).pointer(), 3);
    EXPECT_TRUE(Handle(result).isImmediate());
    EXPECT_THAT(result, Represents(expected(Integer(value), 3)));
    EXPECT_EQ(counter.allocations(), 0u);
  }
}

// Amounts are unsigned, so a narrow one of 200 is not negative.
TEST(Shift, ExtendsNarrowAmount) {
  Integer expected;
  mpz_mul_2exp(expected, Integer(-5), 200);
  void *result = shl_narrow(Handle::immediate(-5).pointer(), 200);
  EXPECT_THAT(result, Represents(expected));
  rcgmp_mpz_release(result, false);
}

// An amount beyond the word saturates instead of wrapping to a small one.
TEST(Shift, SaturatesWideAmount) {
  unsigned __int128 amount = (static_cast<unsigned __int128>(1) << 64) + 1;
  for (long sign : {1L, -1L}) {
    Integer value = Integer::powerOfTwo(130) + 17;
    if (sign < 0)
      value = -value;
    void *result = shr_wide(toRuntime(value), amount);
    EXPECT_THAT(result, Represents(Integer(sign < 0 ? -1 : 0)));
    void *immediate = shr_wide(Handle::immediate(sign * 5).pointer(), amount);
    EXPECT_THAT(immediate, Represents(Integer(sign < 0 ? -1 : 0)));
  }
}

// A left shift whose result cannot be allocated terminates.
TEST(ShiftDeathTest, TerminatesOnUnallocatableResult) {
  EXPECT_DEATH(shl_local(Handle::immediate(1).pointer(),
                         std::numeric_limits<std::uint64_t>::max()),
               "");
}

} // namespace
