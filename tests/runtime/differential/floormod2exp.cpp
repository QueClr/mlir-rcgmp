// Compares compiled remainders by powers of two and their runtime entry points
// with GMP, over the ownership of their operand.
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
import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.allocator;
import rcgmp.testing.differential;

extern "C" {
void *floormod2expi_local(void *, std::uint64_t);
void *floormod2expi_atomic(void *, std::uint64_t);
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

constexpr std::array compiled{floormod2expi_local, floormod2expi_atomic};
constexpr std::array runtime{rcgmp_mpz_fdiv_r_2exp,
                             rcgmp_mpz_fdiv_r_2exp_atomic};

// A shared operand has an extra reference held by the caller, which must
// survive with its value. An immortal heap operand has a zero reference count,
// like a compiled constant.
struct RemainderCase {
  std::uint64_t amount;
  Counting counting;
  bool shared;
  bool immortal;
  Operand operand;
  // The compiled constant, by index, that holds the operand.
  std::optional<std::size_t> constant = std::nullopt;
  // Calls the runtime entry point instead of compiled code, with this reuse
  // mask limited to an operand that compiled code could pass.
  std::optional<std::uint32_t> reuse = std::nullopt;

  std::string describe() const {
    std::string entry =
        reuse ? std::format(", runtime with reuse mask {}", *reuse) : "";
    return std::format("floormod2expi by {} ({}{}{}{}): operand = {}", amount,
                       rcgmp::name(counting), shared ? ", shared" : "",
                       immortal ? ", immortal" : "", entry,
                       operand.describe(constant.has_value()));
  }
};

void checkRemainder(const RemainderCase &test) {
  SCOPED_TRACE(test.describe());
  std::size_t index = std::to_underlying(test.counting);
  const Integer &value = test.operand.value;
  AllocationCounter balance;
  void *x =
      makeOperand(test.operand, test.constant, test.counting, test.immortal);
  bool heap = !Handle(x).isImmediate();
  bool immortal = heap && (test.constant || test.immortal);
  // A nonnegative integer below the power is its own remainder. Any other
  // nonnegative remainder has at most the limbs of the operand, and that of a
  // negative integer the limbs that the amount covers.
  bool below = value.sign() >= 0 && mpz_sizeinbase(value, 2) <= test.amount;
  bool roomy = heap && (value.sign() > 0 ||
                        Handle(x).heap().capacity * word >= test.amount);
  if (test.shared)
    EXPECT_EQ(retains[index](x), x);
  // Compiled code permits reusing a uniquely owned, counted heap operand.
  std::uint32_t reuse =
      test.reuse.value_or(1) & (heap && !immortal && !test.shared);
  Integer expected;
  mpz_fdiv_r_2exp(expected, value, test.amount);
  AllocationCounter counter;
  void *result = test.reuse ? runtime[index](x, test.amount, reuse)
                            : compiled[index](x, test.amount);
  EXPECT_THAT(result, Represents(expected));
  if (heap && below)
    EXPECT_EQ(result, x) << "an integer below the power was not passed on";
  if (roomy && reuse && !Handle(result).isImmediate())
    EXPECT_EQ(counter.allocations(), 0u)
        << "result did not reuse a unique operand";
  if (test.shared)
    releaseShared(test.counting, x, value, result, immortal);
  releases[index](result);
  if (immortal)
    dropImmortal(test.counting, x, value, test.constant.has_value());
  EXPECT_TRUE(balance.balanced())
      << "allocations and deallocations do not balance";
}

// Amounts at the edge of the immediate range, around whole limbs, and any up
// to eight limbs, beyond the magnitude of most operands.
fuzztest::Domain<std::uint64_t> AnyAmount() {
  return fuzztest::OneOf(
      fuzztest::ElementOf<std::uint64_t>(
          {0, 1, word - 4, word - 3, word - 2, word - 1}),
      fuzztest::Map(
          [](std::uint64_t limbs, int delta) { return limbs * word + delta; },
          fuzztest::InRange<std::uint64_t>(1, 6), fuzztest::InRange(-2, 2)),
      fuzztest::InRange<std::uint64_t>(0, 8 * word));
}

// Integers at the edges of the representation, and multiples of powers of two
// up to three limbs, whose low limbs are zero.
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

void RemaindersMatchGmp(std::uint64_t amount, Counting counting,
                        const Integer &value, std::size_t spare) {
  for (auto [shared, immortal] : unaryOwnerships)
    checkRemainder({amount, counting, shared, immortal, {value, spare}});
}
FUZZ_TEST(Differential, RemaindersMatchGmp)
    .WithDomains(AnyAmount(), AnyCounting(), AnyValue(), AnySpare());

// Compiled code calls the runtime with a unique heap operand in the reuse
// mask, but the runtime also accepts an empty mask, and immediate operands.
void RemainderRuntimeMatchesGmp(std::uint64_t amount, Counting counting,
                                const Integer &value, std::size_t spare,
                                std::uint32_t reuse) {
  for (auto [shared, immortal] : unaryOwnerships)
    checkRemainder(
        {amount, counting, shared, immortal, {value, spare}, {}, reuse});
}
FUZZ_TEST(Differential, RemainderRuntimeMatchesGmp)
    .WithDomains(AnyAmount(), AnyCounting(), AnyValue(), AnySpare(),
                 fuzztest::InRange<std::uint32_t>(0, 1));

class RemainderTest : public ::testing::TestWithParam<Counting> {
protected:
  Counting counting = GetParam();

  static constexpr std::array<std::uint64_t, 9> amounts{
      0,        1,           word - 3, word - 2, word, word + 1, 2 * word - 1,
      2 * word, 3 * word + 1};
};

INSTANTIATE_TEST_SUITE_P(Countings, RemainderTest,
                         ::testing::ValuesIn(countings), countingName);

// No spare limbs, so that a negative operand cannot hold a remainder longer
// than itself, and enough for any amount here, from compiled code and the
// runtime.
TEST_P(RemainderTest, MatchesGmpOnBoundaries) {
  for (const auto &[value, amount, spare, ownership, reuse] :
       std::views::cartesian_product(
           boundaries(), amounts, std::array<std::size_t, 2>{0, 4},
           unaryOwnerships,
           std::array<std::optional<std::uint32_t>, 3>{std::nullopt, 0, 1})) {
    auto [shared, immortal] = ownership;
    checkRemainder({amount,
                    counting,
                    shared,
                    immortal,
                    {value, spare},
                    std::nullopt,
                    reuse});
  }
}

TEST_P(RemainderTest, MatchesGmpOnConstants) {
  for (const auto &[index, amount, shared] : std::views::cartesian_product(
           std::views::iota(std::size_t{0}, constantValues.size()), amounts,
           std::array{false, true}))
    checkRemainder(
        {amount, counting, shared, false, {constantValue(index)}, index});
}

} // namespace
