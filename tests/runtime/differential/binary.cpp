// Compares compiled binary operations and their runtime entry points with GMP,
// over the ownership of their operands.
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
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
using namespace rcgmp::test;

void PairsMatchGmp(Operator op, Counting counting,
                   const std::pair<Integer, Integer> &operands,
                   std::size_t lhsSpare, std::size_t rhsSpare) {
  for (Ownership ownership : ownerships)
    checkBinary({op,
                 counting,
                 ownership,
                 false,
                 {operands.first, lhsSpare},
                 {operands.second, rhsSpare},
                 false});
}
FUZZ_TEST(Differential, PairsMatchGmp)
    .WithDomains(AnyOperator(), AnyCounting(), AnyOperands(), AnySpare(),
                 AnySpare());

void AliasesMatchGmp(Operator op, Counting counting, const Integer &value,
                     std::size_t spare) {
  for (Ownership ownership : aliasOwnerships)
    checkBinary(
        {op, counting, ownership, true, {value, spare}, {value, spare}, false});
}
FUZZ_TEST(Differential, AliasesMatchGmp)
    .WithDomains(AnyOperator(), AnyCounting(), AnyInteger(), AnySpare());

// A compiled constant replaces either operand.
void ConstantsMatchGmp(Operator op, Counting counting, std::size_t constant,
                       const Integer &other, std::size_t spare, bool left) {
  Operand c{constantValue(constant)}, y{other, spare};
  for (Ownership ownership : ownerships) {
    BinaryCase test{op, counting, ownership, false, c, y, false};
    if (left) {
      test.lhsConstant = constant;
    } else {
      std::swap(test.lhs, test.rhs);
      test.rhsConstant = constant;
    }
    checkBinary(test);
  }
}
FUZZ_TEST(Differential, ConstantsMatchGmp)
    .WithDomains(AnyOperator(), AnyCounting(), AnyConstant(), AnyInteger(),
                 AnySpare(), fuzztest::Arbitrary<bool>());

// Compiled code calls the runtime with every unique heap operand in the reuse
// mask, but the runtime accepts any subset of them, and immediate operands.
void RuntimeMatchesGmp(Operator op, Counting counting,
                       const std::pair<Integer, Integer> &operands,
                       std::size_t lhsSpare, std::size_t rhsSpare,
                       std::uint32_t reuse) {
  Operand lhs{operands.first, lhsSpare}, rhs{operands.second, rhsSpare};
  for (Ownership ownership : ownerships)
    checkBinary(
        {op, counting, ownership, false, lhs, rhs, false, {}, {}, reuse});
  for (Ownership ownership : aliasOwnerships)
    checkBinary(
        {op, counting, ownership, true, lhs, lhs, false, {}, {}, reuse});
}
FUZZ_TEST(Differential, RuntimeMatchesGmp)
    .WithDomains(AnyOperator(), AnyCounting(), AnyOperands(), AnySpare(),
                 AnySpare(), fuzztest::InRange<std::uint32_t>(0, 3));

class DifferentialTest
    : public ::testing::TestWithParam<std::tuple<Operator, Counting>> {
protected:
  Operator op = std::get<Operator>(GetParam());
  Counting counting = std::get<Counting>(GetParam());
};

std::string parameterName(
    const ::testing::TestParamInfo<std::tuple<Operator, Counting>> &info) {
  auto [op, counting] = info.param;
  return std::string(operation(op).name) + "_" +
         std::string(rcgmp::name(counting));
}

INSTANTIATE_TEST_SUITE_P(Operations, DifferentialTest,
                         ::testing::Combine(::testing::ValuesIn(operators),
                                            ::testing::ValuesIn(countings)),
                         parameterName);

// No spare limbs, so that every growing result needs a new allocation, and
// enough for any result of two boundary operands to fit in either one.
TEST_P(DifferentialTest, MatchesGmpOnBoundaries) {
  std::vector<Integer> values = boundaries();
  for (std::size_t spare : {0, 4}) {
    for (const Integer &lhs : values) {
      Operand x{lhs, spare};
      for (const auto &[rhs, ownership] :
           std::views::cartesian_product(values, ownerships))
        checkBinary(
            {op, counting, ownership, false, x, {rhs, spare}, spare > 0});
      for (Ownership ownership : aliasOwnerships)
        checkBinary({op, counting, ownership, true, x, x, false});
    }
  }
}

// Constants against boundary operands on either side and each other.
TEST_P(DifferentialTest, MatchesGmpOnConstants) {
  std::vector<Integer> values = boundaries();
  for (std::size_t i = 0; i < constantValues.size(); ++i) {
    Operand c{constantValue(i)};
    for (std::size_t spare : {0, 4}) {
      for (const auto &[other, ownership] :
           std::views::cartesian_product(values, ownerships)) {
        Operand y{other, spare};
        checkBinary({op, counting, ownership, false, c, y, spare > 0, i});
        checkBinary(
            {op, counting, ownership, false, y, c, spare > 0, std::nullopt, i});
      }
    }
    for (std::size_t j = 0; j < constantValues.size(); ++j)
      for (Ownership ownership : ownerships)
        checkBinary({op,
                     counting,
                     ownership,
                     false,
                     c,
                     {constantValue(j)},
                     false,
                     i,
                     j});
    for (Ownership ownership : aliasOwnerships)
      checkBinary({op, counting, ownership, true, c, c, false, i});
  }
}

// Compiled code calls the runtime only when an operand is on the heap or an
// immediate result overflows, but every runtime entry point accepts any
// canonical operands and must return a canonical result.
TEST_P(DifferentialTest, RuntimeAcceptsImmediates) {
  std::vector<Integer> values = boundaries();
  for (const auto &[lhs, rhs] : std::views::cartesian_product(values, values)) {
    if (!lhs.fitsImmediate() || !rhs.fitsImmediate())
      continue;
    AllocationCounter balance;
    Integer expected;
    operation(op).oracle(expected, lhs, rhs);
    std::size_t index = std::to_underlying(counting);
    void *result =
        operation(op).runtime[index](toRuntime(lhs), toRuntime(rhs), 0);
    EXPECT_THAT(result, Represents(expected))
        << "lhs = " << lhs << ", rhs = " << rhs;
    releases[index](result);
    EXPECT_TRUE(balance.balanced());
  }
}

} // namespace
