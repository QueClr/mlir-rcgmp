// Compares compiled unary operations and their runtime entry points with GMP,
// over the ownership of their operand.
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
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
struct UnaryCase {
  UnaryOperator op;
  Counting counting;
  bool shared;
  bool immortal;
  Operand operand;
  // A heap operand has room for the result, so a unique one must hold it.
  bool roomy;
  // The compiled constant, by index, that holds the operand.
  std::optional<std::size_t> constant = std::nullopt;
  // Calls the runtime entry point instead of compiled code, with this reuse
  // mask limited to an operand that compiled code could pass.
  std::optional<std::uint32_t> reuse = std::nullopt;

  std::string describe() const {
    std::string entry =
        reuse ? std::format(", runtime with reuse mask {}", *reuse) : "";
    return std::format("{} ({}{}{}{}): operand = {}", operation(op).name,
                       rcgmp::name(counting), shared ? ", shared" : "",
                       immortal ? ", immortal" : "", entry,
                       operand.describe(constant.has_value()));
  }
};

void checkUnary(const UnaryCase &test) {
  SCOPED_TRACE(test.describe());
  std::size_t index = std::to_underlying(test.counting);
  const Integer &value = test.operand.value;
  AllocationCounter balance;
  void *x =
      makeOperand(test.operand, test.constant, test.counting, test.immortal);
  bool heap = !Handle(x).isImmediate();
  bool immortal = heap && (test.constant || test.immortal);
  if (test.shared)
    EXPECT_EQ(retains[index](x), x);
  // Compiled code permits reusing a uniquely owned, counted heap operand.
  std::uint32_t reuse =
      test.reuse.value_or(1) & (heap && !immortal && !test.shared);
  Integer expected;
  operation(test.op).oracle(expected, value);
  AllocationCounter counter;
  void *result = test.reuse ? operation(test.op).runtime[index](x, reuse)
                            : operation(test.op).compiled[index](x);
  EXPECT_THAT(result, Represents(expected));
  if (test.roomy && reuse && !Handle(result).isImmediate())
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

void UnaryMatchesGmp(UnaryOperator op, Counting counting, const Integer &value,
                     std::size_t spare) {
  for (auto [shared, immortal] : unaryOwnerships)
    checkUnary({op, counting, shared, immortal, {value, spare}, false});
}
FUZZ_TEST(Differential, UnaryMatchesGmp)
    .WithDomains(AnyUnaryOperator(), AnyCounting(), AnyInteger(), AnySpare());

// Compiled code calls the runtime with a unique heap operand in the reuse
// mask, but the runtime also accepts an empty mask, and immediate operands.
void UnaryRuntimeMatchesGmp(UnaryOperator op, Counting counting,
                            const Integer &value, std::size_t spare,
                            std::uint32_t reuse) {
  for (auto [shared, immortal] : unaryOwnerships)
    checkUnary(
        {op, counting, shared, immortal, {value, spare}, false, {}, reuse});
}
FUZZ_TEST(Differential, UnaryRuntimeMatchesGmp)
    .WithDomains(AnyUnaryOperator(), AnyCounting(), AnyInteger(), AnySpare(),
                 fuzztest::InRange<std::uint32_t>(0, 1));

class UnaryTest
    : public ::testing::TestWithParam<std::tuple<UnaryOperator, Counting>> {
protected:
  UnaryOperator op = std::get<UnaryOperator>(GetParam());
  Counting counting = std::get<Counting>(GetParam());
};

std::string parameterName(
    const ::testing::TestParamInfo<std::tuple<UnaryOperator, Counting>> &info) {
  auto [op, counting] = info.param;
  return std::string(operation(op).name) + "_" +
         std::string(rcgmp::name(counting));
}

INSTANTIATE_TEST_SUITE_P(Operations, UnaryTest,
                         ::testing::Combine(::testing::ValuesIn(unaryOperators),
                                            ::testing::ValuesIn(countings)),
                         parameterName);

// No spare limbs, so that a growing result needs a new allocation, and enough
// for any result to fit in the operand, from compiled code and the runtime.
TEST_P(UnaryTest, MatchesGmpOnBoundaries) {
  for (const auto &[value, spare, ownership, reuse] :
       std::views::cartesian_product(
           boundaries(), std::array<std::size_t, 2>{0, 4}, unaryOwnerships,
           std::array<std::optional<std::uint32_t>, 3>{std::nullopt, 0, 1})) {
    auto [shared, immortal] = ownership;
    checkUnary({op,
                counting,
                shared,
                immortal,
                {value, spare},
                spare > 0,
                std::nullopt,
                reuse});
  }
}

TEST_P(UnaryTest, MatchesGmpOnConstants) {
  for (std::size_t i = 0; i < constantValues.size(); ++i)
    for (bool shared : {false, true})
      checkUnary({op, counting, shared, false, {constantValue(i)}, false, i});
}

} // namespace
