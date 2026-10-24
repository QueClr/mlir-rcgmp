// Compiled powers.
#include "RCGMP/Runtime/Mpz.h"
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <string>
#include <utility>
#include <vector>
import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.allocator;

extern "C" {
void *pow_local(void *, std::uint64_t);
void *pow_atomic(void *, std::uint64_t);
void *pow_narrow(void *, std::uint8_t);
void *pow_wide(void *, unsigned __int128);
void *pow_by_2(void *);
void *pow_by_3_atomic(void *);
void *pow_by_5(void *);
void *pow_by_61(void *);
void *pow_by_62(void *);
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

constexpr unsigned word = GMP_NUMB_BITS;
constexpr std::array powers{pow_local, pow_atomic};

Integer power(const Integer &base, std::uint64_t exponent) {
  Integer result;
  mpz_pow_ui(result, base, exponent);
  return result;
}

// Applies a compiled power to a base, shared if the caller keeps an extra
// reference, and compares the result with GMP.
void checkPower(const Integer &base, std::uint64_t exponent, Counting counting,
                bool shared) {
  bool atomic = counting == Counting::atomic;
  void *x = toRuntime(base);
  if (shared)
    rcgmp_mpz_retain(x, atomic);
  void *result = powers[atomic](x, exponent);
  EXPECT_THAT(result, Represents(power(base, exponent)))
      << base << " to the power " << exponent;
  if (shared) {
    EXPECT_THAT(x, Represents(base));
    rcgmp_mpz_release(x, atomic);
  }
  rcgmp_mpz_release(result, atomic);
}

// The k-th root of 2^bits, near which the k-th power leaves the immediate
// range, a limb, or two limbs.
Integer root(unsigned bits, unsigned k) {
  Integer result;
  mpz_root(result, Integer::powerOfTwo(bits), k);
  return result;
}

// Small exponents select the inline path, and larger ones grow results to a
// few hundred limbs.
fuzztest::Domain<std::uint64_t> AnyExponent() {
  return fuzztest::OneOf(fuzztest::InRange<std::uint64_t>(0, 4),
                         fuzztest::InRange<std::uint64_t>(0, 200));
}

// Small bases, bases with trailing zero bits of whole limbs or within one,
// times a factor of one limb or more, and integers at the edges of the
// representation, which AnyInteger provides.
fuzztest::Domain<Integer> AnyBase() {
  return fuzztest::OneOf(
      fuzztest::Map([](long value) { return Integer(value); },
                    fuzztest::InRange(-17L, 17L)),
      fuzztest::Map(
          [](const Integer &factor, unsigned zeros) {
            Integer base;
            mpz_mul_2exp(base, factor, zeros);
            return base;
          },
          fuzztest::OneOf(
              fuzztest::Map([](long value) { return Integer(value); },
                            fuzztest::InRange(-9L, 9L)),
              AnyInteger()),
          fuzztest::InRange(0u, 3 * word)),
      AnyInteger());
}

// A base near the k-th root of a bound with an exponent near k, or any base
// and exponent.
fuzztest::Domain<std::pair<Integer, std::uint64_t>> AnyPower() {
  return fuzztest::OneOf(
      fuzztest::PairOf(AnyBase(), AnyExponent()),
      fuzztest::Map(
          [](unsigned bits, unsigned k, long delta, int offset, bool negative) {
            Integer base = root(bits, k) + delta;
            return std::pair{negative ? -base : base,
                             static_cast<std::uint64_t>(k + offset)};
          },
          fuzztest::ElementOf<unsigned>({word - 3, word, 2 * word}),
          fuzztest::InRange(2u, 7u), fuzztest::InRange(-2L, 2L),
          fuzztest::InRange(-1, 1), fuzztest::Arbitrary<bool>()));
}

void MatchesGmp(const std::pair<Integer, std::uint64_t> &power,
                Counting counting, bool shared) {
  checkPower(power.first, power.second, counting, shared);
}
FUZZ_TEST(Power, MatchesGmp)
    .WithDomains(AnyPower(), AnyCounting(), fuzztest::Arbitrary<bool>());

// A compiled power by a constant exponent.
struct ConstantPower {
  std::uint64_t exponent;
  void *(*apply)(void *);
  Counting counting;
};

constexpr std::array<ConstantPower, 5> constantPowers{{
    {2, pow_by_2, Counting::nonatomic},
    {3, pow_by_3_atomic, Counting::atomic},
    {5, pow_by_5, Counting::nonatomic},
    {61, pow_by_61, Counting::nonatomic},
    {62, pow_by_62, Counting::nonatomic},
}};

void checkConstantPower(std::size_t index, const Integer &base, bool shared) {
  auto [exponent, apply, counting] = constantPowers[index];
  bool atomic = counting == Counting::atomic;
  void *x = toRuntime(base);
  if (shared)
    rcgmp_mpz_retain(x, atomic);
  void *result = apply(x);
  EXPECT_THAT(result, Represents(power(base, exponent)))
      << base << " to the constant power " << exponent;
  if (shared) {
    EXPECT_THAT(x, Represents(base));
    rcgmp_mpz_release(x, atomic);
  }
  rcgmp_mpz_release(result, atomic);
}

// Bases near the k-th root of the immediate bound select the constant power
// k, and 61 or 62 for magnitudes up to two.
fuzztest::Domain<Integer> AnyConstantBase() {
  return fuzztest::OneOf(
      fuzztest::Map([](long value) { return Integer(value); },
                    fuzztest::InRange(-3L, 3L)),
      fuzztest::Map(
          [](unsigned k, long delta, bool negative) {
            Integer base = root(word - 3, k) + delta;
            return negative ? -base : base;
          },
          fuzztest::ElementOf<unsigned>({2, 3, 5}), fuzztest::InRange(-2L, 2L),
          fuzztest::Arbitrary<bool>()),
      AnyInteger());
}

void ConstantPowersMatchGmp(std::size_t index, const Integer &base,
                            bool shared) {
  checkConstantPower(index, base, shared);
}
FUZZ_TEST(Power, ConstantPowersMatchGmp)
    .WithDomains(fuzztest::InRange<std::size_t>(0, constantPowers.size() - 1),
                 AnyConstantBase(), fuzztest::Arbitrary<bool>());

// Each constant power at the edges of the immediate range, where its products
// overflow, and of magnitude at most two.
TEST(Power, MatchesGmpOnConstantExponents) {
  AllocationCounter allocations;
  for (std::size_t index = 0; index < constantPowers.size(); ++index) {
    std::uint64_t exponent = constantPowers[index].exponent;
    std::vector<Integer> bases;
    for (long value : {0L, 1L, -1L, 2L, -2L, 3L})
      bases.emplace_back(value);
    if (exponent <= 5)
      for (long delta : {0L, 1L}) {
        Integer base = root(word - 3, exponent) + delta;
        bases.push_back(base);
        bases.push_back(-base);
      }
    for (const auto &[base, shared] :
         std::views::cartesian_product(bases, std::array{false, true}))
      checkConstantPower(index, base, shared);
  }
  EXPECT_TRUE(allocations.balanced());
}

class PowerTest : public ::testing::TestWithParam<Counting> {
protected:
  void TearDown() override { EXPECT_TRUE(allocations.balanced()); }

  // A unique heap base with spare limbs, which may hold its power.
  static void *roomy(const Integer &value, std::size_t spare) {
    void *exact = toRuntime(value);
    rcgmp::Mpz &heap = Handle(exact).heap();
    rcgmp::Mpz &padded = rcgmp::allocate(heap.count() + spare);
    std::ranges::copy(heap.magnitude(), padded.limbs().begin());
    padded.size = heap.size;
    rcgmp_mpz_release(exact, false);
    return Handle(padded).pointer();
  }

  Counting counting = GetParam();
  bool atomic = counting == Counting::atomic;
  AllocationCounter allocations;
};

INSTANTIATE_TEST_SUITE_P(Countings, PowerTest, ::testing::ValuesIn(countings),
                         countingName);

// Squares and cubes near the immediate bound and the limb bound, bases of
// magnitude at most one, which keep any exponent immediate, and bases with
// trailing zero bits, of either sign.
TEST_P(PowerTest, MatchesGmpOnBoundaries) {
  std::vector<Integer> bases;
  for (long value : {0L, 1L, -1L, 2L, -2L, 3L, -3L, 10L, Handle::immediateMax,
                     Handle::immediateMin})
    bases.emplace_back(value);
  for (auto [bits, k] : {std::pair{word - 3, 2u},
                         {word - 3, 3u},
                         {word, 2u},
                         {word, 3u},
                         {2 * word, 2u}})
    for (long delta : {0L, 1L}) {
      Integer base = root(bits, k) + delta;
      bases.push_back(base);
      bases.push_back(-base);
    }
  for (unsigned zeros : {1u, word - 1, word, word + 1, 2 * word + 5}) {
    for (long factor : {1L, 3L, -5L}) {
      Integer base;
      mpz_mul_2exp(base, Integer(factor), zeros);
      bases.push_back(base);
    }
    Integer base;
    mpz_mul_2exp(base, Integer::powerOfTwo(word + 3) + 1, zeros);
    bases.push_back(base);
    bases.push_back(-base);
  }
  std::vector<std::uint64_t> exponents{0,  1,  2,  3,  4,  5,
                                       31, 32, 33, 63, 64, 65};
  for (const auto &[base, exponent, shared] :
       std::views::cartesian_product(bases, exponents, std::array{false, true}))
    checkPower(base, exponent, counting, shared);
}

// Powers of immediates that fit a limb allocate only a result that is not
// immediate.
TEST_P(PowerTest, AllocatesOnlyResultOfSmallPower) {
  for (auto [base, exponent] : {std::pair{3L, 39ul},
                                {-3L, 39ul},
                                {7L, 22ul},
                                {-2L, 61ul},
                                {3L, 40ul}}) {
    Integer expected = power(Integer(base), exponent);
    AllocationCounter counter;
    void *result = powers[atomic](Handle::immediate(base).pointer(), exponent);
    EXPECT_THAT(result, Represents(expected));
    EXPECT_EQ(counter.allocations(), expected.fitsImmediate() ? 0u : 1u);
    rcgmp_mpz_release(result, atomic);
  }
}

// A unique heap base with room for its power holds it, whether its odd factor
// is one limb, or several limbs with or without trailing zero bits, which
// must then be copied before the base is overwritten. Squares, like other
// products of several limbs, and powers of two, which are shifts of one, take
// new storage.
TEST_P(PowerTest, ReusesUniqueHeapBase) {
  Integer several = Integer::powerOfTwo(word + 3) + 1, shifted;
  mpz_mul_2exp(shifted, several, 5);
  for (const Integer &value : {power(Integer(3), 40), -power(Integer(3), 40),
                               several, -several, shifted}) {
    for (std::uint64_t exponent : {3, 5, 6}) {
      void *x = roomy(value, 16);
      AllocationCounter counter;
      void *result = powers[atomic](x, exponent);
      EXPECT_EQ(result, x) << value << " to the power " << exponent;
      EXPECT_EQ(counter.allocations(), 0u);
      EXPECT_THAT(result, Represents(power(value, exponent)));
      rcgmp_mpz_release(result, atomic);
    }
  }
}

// A shared base keeps its value and has its reference count restored.
TEST_P(PowerTest, KeepsSharedBase) {
  Integer value = Integer::powerOfTwo(word + 3) + 1;
  void *x = roomy(value, 16);
  rcgmp_mpz_retain(x, atomic);
  void *result = powers[atomic](x, 3);
  EXPECT_NE(result, x);
  EXPECT_THAT(result, Represents(power(value, 3)));
  EXPECT_THAT(x, Represents(value));
  EXPECT_EQ(references(x, counting), 1u);
  rcgmp_mpz_release(result, atomic);
  rcgmp_mpz_release(x, atomic);
}

// Exponents are unsigned, so a narrow one of 200 is not negative.
TEST(Power, ExtendsNarrowExponent) {
  void *result = pow_narrow(Handle::immediate(-3).pointer(), 200);
  EXPECT_THAT(result, Represents(power(Integer(-3), 200)));
  rcgmp_mpz_release(result, false);
}

// An exponent beyond the word saturates with its parity, which decides the
// powers of -1, and leaves zero and one unchanged.
TEST(Power, SaturatesWideExponent) {
  unsigned __int128 even = static_cast<unsigned __int128>(1) << 64;
  for (auto [base, exponent, expected] : {std::tuple{-1L, even, 1L},
                                          {-1L, even + 1, -1L},
                                          {-1L, even - 1, -1L},
                                          {0L, even, 0L},
                                          {1L, even + 1, 1L}})
    EXPECT_THAT(pow_wide(Handle::immediate(base).pointer(), exponent),
                Represents(Integer(expected)));
  EXPECT_THAT(pow_wide(Handle::immediate(5).pointer(), 0),
              Represents(Integer(1)));
}

// A power whose result cannot be allocated terminates.
TEST(PowerDeathTest, TerminatesOnUnallocatableResult) {
  constexpr std::uint64_t huge = std::numeric_limits<std::uint64_t>::max();
  EXPECT_DEATH(pow_local(Handle::immediate(3).pointer(), huge), "");
  EXPECT_DEATH(pow_local(Handle::immediate(-2).pointer(), huge), "");
  EXPECT_DEATH(pow_local(toRuntime(Integer::powerOfTwo(word + 3) + 1), huge),
               "");
}

} // namespace
