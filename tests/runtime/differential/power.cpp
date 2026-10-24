// Compares compiled powers and their runtime entry points with GMP, over the
// ownership of their base.
#include "RCGMP/Runtime/Mpz.h"
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <ranges>
#include <string>
#include <utility>
#include <vector>
import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.allocator;
import rcgmp.testing.differential;

extern "C" {
void *ipowi_local(void *, std::uint64_t);
void *ipowi_atomic(void *, std::uint64_t);
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

constexpr std::array compiled{ipowi_local, ipowi_atomic};
constexpr std::array runtime{rcgmp_mpz_pow, rcgmp_mpz_pow_atomic};

// A shared base has an extra reference held by the caller, which must survive
// with its value. An immortal heap base has a zero reference count, like a
// compiled constant.
struct PowerCase {
  std::uint64_t exponent;
  Counting counting;
  bool shared;
  bool immortal;
  Operand base;
  // The compiled constant, by index, that holds the base.
  std::optional<std::size_t> constant = std::nullopt;
  // Calls the runtime entry point instead of compiled code, with this reuse
  // mask limited to a base that compiled code could pass.
  std::optional<std::uint32_t> reuse = std::nullopt;

  std::string describe() const {
    std::string entry =
        reuse ? std::format(", runtime with reuse mask {}", *reuse) : "";
    return std::format("ipowi to the power {} ({}{}{}{}): base = {}", exponent,
                       rcgmp::name(counting), shared ? ", shared" : "",
                       immortal ? ", immortal" : "", entry,
                       base.describe(constant.has_value()));
  }
};

void checkPower(const PowerCase &test) {
  SCOPED_TRACE(test.describe());
  std::size_t index = std::to_underlying(test.counting);
  const Integer &value = test.base.value;
  AllocationCounter balance;
  void *x = makeOperand(test.base, test.constant, test.counting, test.immortal);
  bool heap = !Handle(x).isImmediate();
  bool immortal = heap && (test.constant || test.immortal);
  // The power has at most exponent times as many bits as the base, and two
  // spare limbs suffice for the runtime to hold it in the base. Squares, like
  // other products of several limbs, and powers of two, which are shifts of
  // one, take new storage.
  std::size_t bits = mpz_sizeinbase(value, 2);
  bool roomy = heap && test.exponent > 2 && mpz_scan1(value, 0) + 1 != bits &&
               Handle(x).heap().capacity >= test.exponent * bits / word + 2;
  if (test.shared)
    EXPECT_EQ(retains[index](x), x);
  // Compiled code permits reusing a uniquely owned, counted heap base.
  std::uint32_t reuse =
      test.reuse.value_or(1) & (heap && !immortal && !test.shared);
  Integer expected;
  mpz_pow_ui(expected, value, test.exponent);
  void *result = test.reuse ? runtime[index](x, test.exponent, reuse)
                            : compiled[index](x, test.exponent);
  EXPECT_THAT(result, Represents(expected));
  if (roomy && reuse)
    EXPECT_EQ(result, x) << "result did not reuse a unique base";
  if (test.shared)
    releaseShared(test.counting, x, value, result, immortal);
  releases[index](result);
  if (immortal)
    dropImmortal(test.counting, x, value, test.constant.has_value());
  EXPECT_TRUE(balance.balanced())
      << "allocations and deallocations do not balance";
}

// Exponents that the inline path takes, and larger ones that grow results to
// a few hundred limbs.
fuzztest::Domain<std::uint64_t> AnyExponent() {
  return fuzztest::OneOf(fuzztest::InRange<std::uint64_t>(0, 4),
                         fuzztest::InRange<std::uint64_t>(0, 200));
}

// Small bases, bases with trailing zero bits, and integers at the edges of the
// representation.
fuzztest::Domain<Integer> AnyBase() {
  return fuzztest::OneOf(
      fuzztest::Map([](long value) { return Integer(value); },
                    fuzztest::InRange(-9L, 9L)),
      fuzztest::Map(
          [](const Integer &factor, unsigned zeros) {
            Integer base;
            mpz_mul_2exp(base, factor, zeros);
            return base;
          },
          AnyInteger(), fuzztest::InRange(0u, 3 * word)),
      AnyInteger());
}

// Spare limbs up to a few powers of a one-limb base, so that a unique heap
// base can hold its power in place.
fuzztest::Domain<std::size_t> AnyPowerSpare() {
  return fuzztest::InRange<std::size_t>(0, 16);
}

void PowersMatchGmp(std::uint64_t exponent, Counting counting,
                    const Integer &base, std::size_t spare) {
  for (auto [shared, immortal] : unaryOwnerships)
    checkPower({exponent, counting, shared, immortal, {base, spare}});
}
FUZZ_TEST(Differential, PowersMatchGmp)
    .WithDomains(AnyExponent(), AnyCounting(), AnyBase(), AnyPowerSpare());

// Compiled code calls the runtime with a unique heap base in the reuse mask,
// but the runtime also accepts an empty mask, and immediate bases.
void PowerRuntimeMatchesGmp(std::uint64_t exponent, Counting counting,
                            const Integer &base, std::size_t spare,
                            std::uint32_t reuse) {
  for (auto [shared, immortal] : unaryOwnerships)
    checkPower(
        {exponent, counting, shared, immortal, {base, spare}, {}, reuse});
}
FUZZ_TEST(Differential, PowerRuntimeMatchesGmp)
    .WithDomains(AnyExponent(), AnyCounting(), AnyBase(), AnyPowerSpare(),
                 fuzztest::InRange<std::uint32_t>(0, 1));

class PowerTest : public ::testing::TestWithParam<Counting> {
protected:
  Counting counting = GetParam();

  static constexpr std::array<std::uint64_t, 7> exponents{0, 1,  2, 3,
                                                          5, 31, 64};
};

INSTANTIATE_TEST_SUITE_P(Countings, PowerTest, ::testing::ValuesIn(countings),
                         countingName);

// No spare limbs, so that a power needs a new allocation, and enough for the
// cube of a one-limb base to fit in it, from compiled code and the runtime.
TEST_P(PowerTest, MatchesGmpOnBoundaries) {
  for (const auto &[value, exponent, spare, ownership, reuse] :
       std::views::cartesian_product(
           boundaries(), exponents, std::array<std::size_t, 2>{0, 5},
           unaryOwnerships,
           std::array<std::optional<std::uint32_t>, 3>{std::nullopt, 0, 1})) {
    auto [shared, immortal] = ownership;
    checkPower({exponent,
                counting,
                shared,
                immortal,
                {value, spare},
                std::nullopt,
                reuse});
  }
}

TEST_P(PowerTest, MatchesGmpOnConstants) {
  for (const auto &[index, exponent, shared] : std::views::cartesian_product(
           std::views::iota(std::size_t{0}, constantValues.size()), exponents,
           std::array{false, true}))
    checkPower(
        {exponent, counting, shared, false, {constantValue(index)}, index});
}

} // namespace
