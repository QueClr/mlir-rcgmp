// Compares compiled divisions and their runtime entry points with GMP, over
// the ownership of their operands. Divisors are nonzero.
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

// As for binary operations, a shared operand has an extra reference held by
// the caller, an aliased rhs is a retained reference to the lhs, and an
// immortal heap operand has a zero reference count.
struct DivisionCase {
  DivisionOperator op;
  Counting counting;
  Ownership ownership;
  bool aliased;
  Operand lhs, rhs;
  // The compiled constants, by index, that hold the lhs or the rhs.
  std::optional<std::size_t> lhsConstant = std::nullopt,
                             rhsConstant = std::nullopt;
  // Calls the runtime entry point instead of compiled code, with this reuse
  // mask limited to the operands that compiled code could pass.
  std::optional<std::uint32_t> reuse = std::nullopt;

  std::string describe() const {
    auto [sharing, immortal] = ownership;
    std::string entry =
        reuse ? std::format(", runtime with reuse mask {}", *reuse) : "";
    return std::format(
        "{} ({}{}{}{}{}{}{}): lhs = {}, rhs = {}", operation(op).name,
        rcgmp::name(counting), aliased ? ", aliased" : "",
        sharing.lhs ? ", shared lhs" : "", sharing.rhs ? ", shared rhs" : "",
        immortal.lhs ? ", immortal lhs" : "",
        immortal.rhs ? ", immortal rhs" : "", entry,
        lhs.describe(lhsConstant.has_value()),
        rhs.describe(rhsConstant.has_value()));
  }
};

void checkDivision(const DivisionCase &test) {
  SCOPED_TRACE(test.describe());
  std::size_t index = std::to_underlying(test.counting);
  const DivisionOperation &op = operation(test.op);
  auto [sharing, immortal] = test.ownership;
  const Integer &lhs = test.lhs.value;
  const Integer &rhs = test.aliased ? lhs : test.rhs.value;
  AllocationCounter balance;
  void *x =
      makeOperand(test.lhs, test.lhsConstant, test.counting, immortal.lhs);
  void *y = x;
  if (test.aliased)
    EXPECT_EQ(retains[index](x), x);
  else
    y = makeOperand(test.rhs, test.rhsConstant, test.counting, immortal.rhs);
  bool immortalX =
      !Handle(x).isImmediate() && (test.lhsConstant || immortal.lhs);
  bool immortalY =
      y == x ? immortalX
             : !Handle(y).isImmediate() && (test.rhsConstant || immortal.rhs);
  if (sharing.lhs)
    EXPECT_EQ(retains[index](x), x);
  if (sharing.rhs)
    EXPECT_EQ(retains[index](y), y);
  // Compiled code permits reusing each operand that is a uniquely owned,
  // counted heap integer.
  bool unique = !sharing.lhs && !sharing.rhs && !test.aliased;
  std::uint32_t uniqueness =
      (unique && !Handle(x).isImmediate() && !immortalX) |
      (unique && !Handle(y).isImmediate() && !immortalY) << 1;
  std::uint32_t reuse = test.reuse.value_or(3) & uniqueness;
  Integer quotient, remainder;
  op.oracle(quotient, remainder, lhs, rhs);
  void *q = nullptr, *r = nullptr;
  if (op.quotient && op.remainder) {
    auto results = test.reuse ? op.runtimePair[index](x, y, reuse)
                              : op.compiledPair[index](x, y);
    q = results.quotient;
    r = results.remainder;
  } else {
    void *result =
        test.reuse ? op.runtime[index](x, y, reuse) : op.compiled[index](x, y);
    (op.quotient ? q : r) = result;
  }
  if (q)
    EXPECT_THAT(q, Represents(quotient));
  if (r)
    EXPECT_THAT(r, Represents(remainder));
  // A unique dividend at least as long as the divisor holds a heap remainder
  // by a divisor of several limbs, and a heap quotient by one of one limb.
  if ((reuse & 1) && mpz_size(lhs) >= mpz_size(rhs)) {
    if (mpz_size(rhs) == 1 && q && !Handle(q).isImmediate())
      EXPECT_EQ(q, x) << "quotient did not reuse the unique dividend";
    if (mpz_size(rhs) > 1 && r && !Handle(r).isImmediate())
      EXPECT_EQ(r, x) << "remainder did not reuse the unique dividend";
  }
  // An operand returned as is is the remainder.
  if (sharing.lhs)
    releaseShared(test.counting, x, lhs, r, immortalX);
  if (sharing.rhs)
    releaseShared(test.counting, y, rhs, r, immortalY);
  if (q)
    releases[index](q);
  if (r)
    releases[index](r);
  if (immortalX)
    dropImmortal(test.counting, x, lhs, test.lhsConstant.has_value());
  if (immortalY && y != x)
    dropImmortal(test.counting, y, rhs, test.rhsConstant.has_value());
  EXPECT_TRUE(balance.balanced())
      << "allocations and deallocations do not balance";
}

// Each ownership of distinct operands, and of aliased ones, whose division
// needs a nonzero lhs.
void checkOwnerships(DivisionOperator op, Counting counting, const Operand &lhs,
                     const Operand &rhs,
                     std::optional<std::uint32_t> reuse = std::nullopt) {
  for (const Ownership &ownership : ownerships)
    checkDivision({op, counting, ownership, false, lhs, rhs, {}, {}, reuse});
  if (lhs.value.sign())
    for (const Ownership &ownership : aliasOwnerships)
      checkDivision({op, counting, ownership, true, lhs, lhs, {}, {}, reuse});
}

// Operands with a zero divisor replaced by one.
fuzztest::Domain<std::pair<Integer, Integer>> AnyDivisionOperands() {
  return fuzztest::Map(
      [](std::pair<Integer, Integer> operands) {
        if (!operands.second.sign())
          operands.second = Integer(1);
        return operands;
      },
      AnyOperands());
}

void DivisionMatchesGmp(DivisionOperator op, Counting counting,
                        const std::pair<Integer, Integer> &operands,
                        std::size_t lhsSpare, std::size_t rhsSpare) {
  checkOwnerships(op, counting, {operands.first, lhsSpare},
                  {operands.second, rhsSpare});
}
FUZZ_TEST(Differential, DivisionMatchesGmp)
    .WithDomains(AnyDivisionOperator(), AnyCounting(), AnyDivisionOperands(),
                 AnySpare(), AnySpare());

// Compiled code calls the runtime with unique heap operands in the reuse mask,
// but the runtime also accepts any subset of them, and immediate operands.
void DivisionRuntimeMatchesGmp(DivisionOperator op, Counting counting,
                               const std::pair<Integer, Integer> &operands,
                               std::size_t lhsSpare, std::size_t rhsSpare,
                               std::uint32_t reuse) {
  checkOwnerships(op, counting, {operands.first, lhsSpare},
                  {operands.second, rhsSpare}, reuse);
}
FUZZ_TEST(Differential, DivisionRuntimeMatchesGmp)
    .WithDomains(AnyDivisionOperator(), AnyCounting(), AnyDivisionOperands(),
                 AnySpare(), AnySpare(),
                 fuzztest::InRange<std::uint32_t>(0, 3));

class DivisionTest
    : public ::testing::TestWithParam<std::tuple<DivisionOperator, Counting>> {
protected:
  DivisionOperator op = std::get<DivisionOperator>(GetParam());
  Counting counting = std::get<Counting>(GetParam());
};

std::string parameterName(
    const ::testing::TestParamInfo<std::tuple<DivisionOperator, Counting>>
        &info) {
  auto [op, counting] = info.param;
  return std::string(operation(op).name) + "_" +
         std::string(rcgmp::name(counting));
}

INSTANTIATE_TEST_SUITE_P(
    Divisions, DivisionTest,
    ::testing::Combine(::testing::ValuesIn(divisionOperators),
                       ::testing::ValuesIn(countings)),
    parameterName);

TEST_P(DivisionTest, MatchesGmpOnBoundaries) {
  for (const auto &[lhs, rhs, reuse] : std::views::cartesian_product(
           boundaries(), boundaries(),
           std::array<std::optional<std::uint32_t>, 2>{std::nullopt, 3}))
    if (rhs.sign())
      checkOwnerships(op, counting, {lhs, 2}, {rhs, 2}, reuse);
}

TEST_P(DivisionTest, MatchesGmpOnConstants) {
  for (std::size_t i = 0; i < constantValues.size(); ++i)
    for (std::size_t j = 0; j < constantValues.size(); ++j)
      if (constantValue(j).sign())
        checkDivision({op,
                       counting,
                       ownerships[0],
                       false,
                       {constantValue(i)},
                       {constantValue(j)},
                       i,
                       j});
}

} // namespace
