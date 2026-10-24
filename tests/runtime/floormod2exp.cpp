// Compiled remainders by powers of two.
#include "RCGMP/Runtime/Mpz.h"
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <vector>
import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.allocator;

extern "C" {
void *mod_local(void *, std::uint64_t);
void *mod_atomic(void *, std::uint64_t);
void *mod_by_0(void *);
void *mod_by_32(void *);
void *mod_by_61(void *);
void *mod_by_62(void *);
void *mod_by_64(void *);
void *mod_by_130_atomic(void *);
void *mod_narrow(void *, std::uint8_t);
void *mod_wide(void *, unsigned __int128);
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

constexpr unsigned word = GMP_NUMB_BITS;
constexpr std::array remainders{mod_local, mod_atomic};

Integer remainder(const Integer &value, std::uint64_t amount) {
  Integer result;
  mpz_fdiv_r_2exp(result, value, amount);
  return result;
}

// Applies a compiled remainder by 2^amount to an operand, shared if the caller
// keeps an extra reference, and compares the result with GMP.
template <typename Apply>
void checkRemainder(Apply apply, std::uint64_t amount, const Integer &value,
                    Counting counting, bool shared) {
  bool atomic = counting == Counting::atomic;
  void *x = toRuntime(value);
  if (shared)
    rcgmp_mpz_retain(x, atomic);
  void *result = apply(x);
  EXPECT_THAT(result, Represents(remainder(value, amount)))
      << value << " modulo 2^" << amount;
  if (shared) {
    EXPECT_THAT(x, Represents(value));
    rcgmp_mpz_release(x, atomic);
  }
  rcgmp_mpz_release(result, atomic);
}

void checkRemainder(const Integer &value, std::uint64_t amount,
                    Counting counting, bool shared) {
  auto apply = remainders[counting == Counting::atomic];
  checkRemainder([&](void *x) { return apply(x, amount); }, amount, value,
                 counting, shared);
}

// Integers at the edges of the representation, and multiples of powers of two
// up to three limbs, whose low bits are zero, so that a negative one has a
// zero remainder by smaller powers.
fuzztest::Domain<Integer> AnyValue() {
  return fuzztest::OneOf(AnyInteger(),
                         fuzztest::Map(
                             [](const Integer &factor, unsigned zeros) {
                               Integer value;
                               mpz_mul_2exp(value, factor, zeros);
                               return value;
                             },
                             AnyInteger(), fuzztest::InRange(0u, 3 * word)));
}

// Amounts at the edge of the immediate range and around whole limbs, and any
// up to eight limbs, beyond the magnitude of most values.
fuzztest::Domain<std::uint64_t> AnyAmount() {
  return fuzztest::OneOf(
      fuzztest::ElementOf<std::uint64_t>(
          {0, 1, word - 4, word - 3, word - 2, word - 1}),
      fuzztest::Map(
          [](std::uint64_t limbs, int delta) { return limbs * word + delta; },
          fuzztest::InRange<std::uint64_t>(1, 6), fuzztest::InRange(-2, 2)),
      fuzztest::InRange<std::uint64_t>(0, 8 * word));
}

void MatchesGmp(const Integer &value, std::uint64_t amount, Counting counting,
                bool shared) {
  checkRemainder(value, amount, counting, shared);
}
FUZZ_TEST(FloorMod2Exp, MatchesGmp)
    .WithDomains(AnyValue(), AnyAmount(), AnyCounting(),
                 fuzztest::Arbitrary<bool>());

// A compiled remainder by a constant amount.
struct ConstantRemainder {
  std::uint64_t amount;
  void *(*apply)(void *);
  Counting counting;
};

constexpr std::array<ConstantRemainder, 6> constantRemainders{{
    {0, mod_by_0, Counting::nonatomic},
    {32, mod_by_32, Counting::nonatomic},
    {61, mod_by_61, Counting::nonatomic},
    {62, mod_by_62, Counting::nonatomic},
    {64, mod_by_64, Counting::nonatomic},
    {130, mod_by_130_atomic, Counting::atomic},
}};

void checkConstantRemainder(std::size_t index, const Integer &value,
                            bool shared) {
  auto [amount, apply, counting] = constantRemainders[index];
  checkRemainder(apply, amount, value, counting, shared);
}

void ConstantRemaindersMatchGmp(std::size_t index, const Integer &value,
                                bool shared) {
  checkConstantRemainder(index, value, shared);
}
FUZZ_TEST(FloorMod2Exp, ConstantRemaindersMatchGmp)
    .WithDomains(fuzztest::InRange<std::size_t>(0,
                                                constantRemainders.size() - 1),
                 AnyValue(), fuzztest::Arbitrary<bool>());

// Zero, one, the immediate bounds, and one to three limbs, of either sign and
// with low limbs of all zeros or all ones.
std::vector<Integer> boundaries() {
  std::vector<Integer> values;
  for (long value : {0L, 1L, -1L, 5L, -5L, Handle::immediateMax,
                     Handle::immediateMin, Handle::immediateMin + 1})
    values.emplace_back(value);
  for (unsigned exponent : {word - 3, word - 2, word, 2 * word, 3 * word})
    for (long delta : {-1L, 0L, 1L}) {
      Integer value = Integer::powerOfTwo(exponent) + delta;
      values.push_back(value);
      values.push_back(-value);
    }
  return values;
}

TEST(FloorMod2Exp, MatchesGmpOnConstantAmounts) {
  AllocationCounter allocations;
  for (const auto &[index, value, shared] : std::views::cartesian_product(
           std::views::iota(std::size_t{0}, constantRemainders.size()),
           boundaries(), std::array{false, true}))
    checkConstantRemainder(index, value, shared);
  EXPECT_TRUE(allocations.balanced());
}

class FloorMod2ExpTest : public ::testing::TestWithParam<Counting> {
protected:
  void TearDown() override { EXPECT_TRUE(allocations.balanced()); }

  Counting counting = GetParam();
  bool atomic = counting == Counting::atomic;
  void *(*apply)(void *, std::uint64_t) = remainders[atomic];
  AllocationCounter allocations;
};

INSTANTIATE_TEST_SUITE_P(Countings, FloorMod2ExpTest,
                         ::testing::ValuesIn(countings), countingName);

TEST_P(FloorMod2ExpTest, MatchesGmpOnBoundaries) {
  std::vector<std::uint64_t> amounts{
      0,        1,    2,        word - 4, word - 3, word - 2,
      word - 1, word, word + 1, 2 * word, 3 * word, 4 * word + 1};
  for (const auto &[value, amount, shared] : std::views::cartesian_product(
           boundaries(), amounts, std::array{false, true}))
    checkRemainder(value, amount, counting, shared);
}

// A nonnegative integer below the power is its own remainder, so even a
// shared one passes its reference on, for any amount.
TEST_P(FloorMod2ExpTest, KeepsIntegerBelowPower) {
  Integer value = Integer::powerOfTwo(2 * word + 5) + 3;
  for (std::uint64_t amount :
       {std::uint64_t{2 * word + 6}, std::uint64_t{3 * word},
        std::numeric_limits<std::uint64_t>::max()}) {
    void *x = toRuntime(value);
    rcgmp_mpz_retain(x, atomic);
    AllocationCounter counter;
    void *result = apply(x, amount);
    EXPECT_EQ(result, x);
    EXPECT_EQ(counter.allocations(), 0u);
    EXPECT_EQ(references(x, counting), 2u);
    rcgmp_mpz_release(result, atomic);
    rcgmp_mpz_release(x, atomic);
  }
}

// A unique heap operand holds a remainder that is not immediate in place:
// fewer limbs for a nonnegative one, and the two's complement of the low limbs
// for a negative one, up to its capacity, which is one limb more than its
// magnitude.
TEST_P(FloorMod2ExpTest, ReusesUniqueHeapOperand) {
  std::array<std::uintptr_t, 4> words{0x8000000000000005, 7, 0, 1};
  Integer value = Integer::fromWords(words, false);
  for (const auto &[operand, amount] : std::views::cartesian_product(
           std::array{value, -value},
           std::array{word, 2 * word - 3, 2 * word + 1, 5 * word})) {
    void *x = toRuntime(operand);
    AllocationCounter counter;
    void *result = apply(x, amount);
    EXPECT_EQ(result, x) << operand << " modulo 2^" << amount;
    EXPECT_EQ(counter.allocations(), 0u);
    EXPECT_THAT(result, Represents(remainder(operand, amount)));
    rcgmp_mpz_release(result, atomic);
  }
}

// An immediate remainder allocates nothing, whether compiled code finds it
// inline or the runtime does, and a unique heap operand is freed.
TEST_P(FloorMod2ExpTest, ReturnsImmediateRemainderWithoutAllocation) {
  Integer value = Integer::powerOfTwo(2 * word) + 0x1234;
  auto runtime = atomic ? rcgmp_mpz_fdiv_r_2exp_atomic : rcgmp_mpz_fdiv_r_2exp;
  for (const auto &[operand, shared, compiled] : std::views::cartesian_product(
           std::array{value, -value}, std::array{false, true},
           std::array{false, true})) {
    void *x = toRuntime(operand);
    if (shared)
      rcgmp_mpz_retain(x, atomic);
    AllocationCounter counter;
    void *result = compiled ? apply(x, 20) : runtime(x, 20, !shared);
    EXPECT_TRUE(Handle(result).isImmediate());
    EXPECT_THAT(result, Represents(remainder(operand, 20)));
    EXPECT_EQ(counter.allocations(), 0u);
    EXPECT_EQ(counter.deallocations(), shared ? 0u : 1u);
    if (shared)
      rcgmp_mpz_release(x, atomic);
  }
}

// Amounts are unsigned, so a narrow one of 200 is not negative.
TEST(FloorMod2Exp, ExtendsNarrowAmount) {
  void *result = mod_narrow(Handle::immediate(-5).pointer(), 200);
  EXPECT_THAT(result, Represents(remainder(Integer(-5), 200)));
  rcgmp_mpz_release(result, false);
}

// An amount beyond the word saturates instead of wrapping to a small one, and
// no nonnegative integer reaches the saturated power.
TEST(FloorMod2Exp, SaturatesWideAmount) {
  unsigned __int128 amount = (static_cast<unsigned __int128>(1) << 64) + 1;
  for (const Integer &value : {Integer::powerOfTwo(130) + 17, Integer(5)}) {
    void *result = mod_wide(toRuntime(value), amount);
    EXPECT_THAT(result, Represents(value));
    rcgmp_mpz_release(result, false);
  }
}

// The remainder of a negative integer by a power too large to allocate
// terminates.
TEST(FloorMod2ExpDeathTest, TerminatesOnUnallocatableResult) {
  constexpr std::uint64_t huge = std::numeric_limits<std::uint64_t>::max();
  EXPECT_DEATH(mod_local(Handle::immediate(-3).pointer(), huge), "");
  EXPECT_DEATH(mod_local(toRuntime(-Integer::powerOfTwo(word + 3)), huge), "");
}

} // namespace
