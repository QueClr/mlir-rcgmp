// Compares compiled shifts and their runtime entry points with GMP, over the
// ownership of their operand and amounts around whole limbs.
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <optional>
#include <ranges>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.allocator;
import rcgmp.testing.differential;

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

// A shared operand has an extra reference held by the caller, which must
// survive with its value. An immortal heap operand has a zero reference count,
// like a compiled constant.
struct ShiftCase {
  Shift shift;
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
    return std::format("{} by {} ({}{}{}{}): operand = {}",
                       operation(shift.op).name, shift.amount,
                       rcgmp::name(counting), shared ? ", shared" : "",
                       immortal ? ", immortal" : "", entry,
                       operand.describe(constant.has_value()));
  }
};

void checkShift(const ShiftCase &test) {
  SCOPED_TRACE(test.describe());
  std::size_t index = std::to_underlying(test.counting);
  auto [op, amount] = test.shift;
  const Integer &value = test.operand.value;
  AllocationCounter balance;
  void *x =
      makeOperand(test.operand, test.constant, test.counting, test.immortal);
  bool heap = !Handle(x).isImmediate();
  bool immortal = heap && (test.constant || test.immortal);
  // A right shift never needs more limbs than its operand has, and a left
  // shift needs room for the bits shifted in.
  bool roomy = heap && (op == ShiftOperator::shrsi ||
                        Handle(x).heap().capacity * word >=
                            mpz_size(value) * word + amount);
  if (test.shared)
    EXPECT_EQ(retains[index](x), x);
  // Compiled code permits reusing a uniquely owned, counted heap operand.
  std::uint32_t reuse =
      test.reuse.value_or(1) & (heap && !immortal && !test.shared);
  Integer expected;
  operation(op).oracle(expected, value, amount);
  AllocationCounter counter;
  void *result = test.reuse ? operation(op).runtime[index](x, amount, reuse)
                            : operation(op).compiled[index](x, amount);
  EXPECT_THAT(result, Represents(expected));
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

// Amounts near zero and near whole limbs, up to five of them.
fuzztest::Domain<std::uint64_t> AnyAmount() {
  return fuzztest::OneOf(
      fuzztest::InRange<std::uint64_t>(0, 4),
      fuzztest::Map(
          [](std::uint64_t limbs, int delta) { return limbs * word + delta; },
          fuzztest::InRange<std::uint64_t>(1, 4), fuzztest::InRange(-2, 2)),
      fuzztest::InRange<std::uint64_t>(0, 5 * word));
}

// Only a right shift takes amounts beyond any integer's bits.
fuzztest::Domain<Shift> AnyShift() {
  return fuzztest::OneOf(
      fuzztest::StructOf<Shift>(AnyShiftOperator(), AnyAmount()),
      fuzztest::StructOf<Shift>(
          fuzztest::Just(ShiftOperator::shrsi),
          fuzztest::OneOf(fuzztest::ElementOf<std::uint64_t>(
                              {std::uint64_t{1} << 32, std::uint64_t{1} << 63,
                               std::numeric_limits<std::uint64_t>::max()}),
                          fuzztest::Arbitrary<std::uint64_t>())));
}

void ShiftsMatchGmp(Shift shift, Counting counting, const Integer &value,
                    std::size_t spare) {
  for (auto [shared, immortal] : unaryOwnerships)
    checkShift({shift, counting, shared, immortal, {value, spare}});
}
FUZZ_TEST(Differential, ShiftsMatchGmp)
    .WithDomains(AnyShift(), AnyCounting(), AnyInteger(), AnySpare());

// Compiled code calls the runtime with a unique heap operand in the reuse
// mask, but the runtime also accepts an empty mask, and immediate operands.
void ShiftRuntimeMatchesGmp(Shift shift, Counting counting,
                            const Integer &value, std::size_t spare,
                            std::uint32_t reuse) {
  for (auto [shared, immortal] : unaryOwnerships)
    checkShift({shift, counting, shared, immortal, {value, spare}, {}, reuse});
}
FUZZ_TEST(Differential, ShiftRuntimeMatchesGmp)
    .WithDomains(AnyShift(), AnyCounting(), AnyInteger(), AnySpare(),
                 fuzztest::InRange<std::uint32_t>(0, 1));

class ShiftTest
    : public ::testing::TestWithParam<std::tuple<ShiftOperator, Counting>> {
protected:
  ShiftOperator op = std::get<ShiftOperator>(GetParam());
  Counting counting = std::get<Counting>(GetParam());

  // Amounts around whole limbs, and for a right shift, beyond any integer.
  std::vector<std::uint64_t> amounts() const {
    std::vector<std::uint64_t> amounts{0,    1,        2,        word - 1,
                                       word, word + 1, 2 * word, 3 * word - 1};
    if (op == ShiftOperator::shrsi)
      amounts.push_back(std::numeric_limits<std::uint64_t>::max());
    return amounts;
  }
};

std::string parameterName(
    const ::testing::TestParamInfo<std::tuple<ShiftOperator, Counting>> &info) {
  auto [op, counting] = info.param;
  return std::string(operation(op).name) + "_" +
         std::string(rcgmp::name(counting));
}

INSTANTIATE_TEST_SUITE_P(Shifts, ShiftTest,
                         ::testing::Combine(::testing::ValuesIn(shiftOperators),
                                            ::testing::ValuesIn(countings)),
                         parameterName);

// No spare limbs, so that a growing result needs a new allocation, and enough
// for the result of any amount here to fit in the operand, from compiled code
// and the runtime.
TEST_P(ShiftTest, MatchesGmpOnBoundaries) {
  for (const auto &[value, amount, spare, ownership, reuse] :
       std::views::cartesian_product(
           boundaries(), amounts(), std::array<std::size_t, 2>{0, 4},
           unaryOwnerships,
           std::array<std::optional<std::uint32_t>, 3>{std::nullopt, 0, 1})) {
    auto [shared, immortal] = ownership;
    checkShift({{op, amount},
                counting,
                shared,
                immortal,
                {value, spare},
                std::nullopt,
                reuse});
  }
}

TEST_P(ShiftTest, MatchesGmpOnConstants) {
  for (const auto &[index, amount, shared] : std::views::cartesian_product(
           std::views::iota(std::size_t{0}, constantValues.size()), amounts(),
           std::array{false, true}))
    checkShift(
        {{op, amount}, counting, shared, false, {constantValue(index)}, index});
}

} // namespace
