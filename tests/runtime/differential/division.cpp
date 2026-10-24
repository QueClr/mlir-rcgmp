// Compares compiled divisions and their runtime entry points with GMP, over
// the ownership of their operands and the modes for a zero divisor. Only lean
// divisions receive zero divisors, which give the quotient zero and the
// remainder lhs; zero divisors of the others are tested to terminate in
// division.cpp.
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
  DivZero mode;
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
        "{} {} ({}{}{}{}{}{}{}): lhs = {}, rhs = {}", operation(op).name,
        name(mode), rcgmp::name(counting), aliased ? ", aliased" : "",
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
  Integer quotient, remainder = lhs;
  if (rhs.sign())
    op.oracle(quotient, remainder, lhs, rhs);
  std::size_t mode = std::to_underlying(test.mode);
  bool lean = test.mode == DivZero::lean;
  void *q = nullptr, *r = nullptr;
  if (op.quotient && op.remainder) {
    auto results = test.reuse ? op.runtimePair[lean][index](x, y, reuse)
                              : op.compiledPair[mode][index](x, y);
    q = results.quotient;
    r = results.remainder;
  } else {
    void *result = test.reuse ? op.runtime[lean][index](x, y, reuse)
                              : op.compiled[mode][index](x, y);
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
void checkOwnerships(DivisionOperator op, DivZero mode, Counting counting,
                     const Operand &lhs, const Operand &rhs,
                     std::optional<std::uint32_t> reuse = std::nullopt) {
  for (const Ownership &ownership : ownerships)
    checkDivision(
        {op, mode, counting, ownership, false, lhs, rhs, {}, {}, reuse});
  if (lhs.value.sign())
    for (const Ownership &ownership : aliasOwnerships)
      checkDivision(
          {op, mode, counting, ownership, true, lhs, lhs, {}, {}, reuse});
}

// Operands, of which half have a zero divisor.
fuzztest::Domain<std::pair<Integer, Integer>> AnyDivisionOperands() {
  return fuzztest::OneOf(AnyOperands(), fuzztest::Map(
                                            [](const Integer &lhs) {
                                              return std::pair{lhs, Integer(0)};
                                            },
                                            AnyInteger()));
}

// The divisor of a division in mode, where only lean divisions may divide by
// zero: the others divide by one instead.
Operand divisor(DivZero mode, const Integer &rhs, std::size_t spare) {
  if (!rhs.sign() && mode != DivZero::lean)
    return {Integer(1), spare};
  return {rhs, spare};
}

void DivisionMatchesGmp(DivisionOperator op, DivZero mode, Counting counting,
                        const std::pair<Integer, Integer> &operands,
                        std::size_t lhsSpare, std::size_t rhsSpare) {
  checkOwnerships(op, mode, counting, {operands.first, lhsSpare},
                  divisor(mode, operands.second, rhsSpare));
}
FUZZ_TEST(Differential, DivisionMatchesGmp)
    .WithDomains(AnyDivisionOperator(), AnyDivZero(), AnyCounting(),
                 AnyDivisionOperands(), AnySpare(), AnySpare());

// Compiled code calls the runtime with unique heap operands in the reuse mask,
// but the runtime also accepts any subset of them, and immediate operands.
void DivisionRuntimeMatchesGmp(DivisionOperator op, DivZero mode,
                               Counting counting,
                               const std::pair<Integer, Integer> &operands,
                               std::size_t lhsSpare, std::size_t rhsSpare,
                               std::uint32_t reuse) {
  checkOwnerships(op, mode, counting, {operands.first, lhsSpare},
                  divisor(mode, operands.second, rhsSpare), reuse);
}
FUZZ_TEST(Differential, DivisionRuntimeMatchesGmp)
    .WithDomains(AnyDivisionOperator(), AnyDivZero(), AnyCounting(),
                 AnyDivisionOperands(), AnySpare(), AnySpare(),
                 fuzztest::InRange<std::uint32_t>(0, 3));

using DivisionParameter = std::tuple<DivisionOperator, DivZero, Counting>;

class DivisionTest : public ::testing::TestWithParam<DivisionParameter> {
protected:
  DivisionOperator op = std::get<DivisionOperator>(GetParam());
  DivZero mode = std::get<DivZero>(GetParam());
  Counting counting = std::get<Counting>(GetParam());
};

std::string
parameterName(const ::testing::TestParamInfo<DivisionParameter> &info) {
  auto [op, mode, counting] = info.param;
  return std::format("{}_{}_{}", operation(op).name, name(mode),
                     rcgmp::name(counting));
}

INSTANTIATE_TEST_SUITE_P(
    Divisions, DivisionTest,
    ::testing::Combine(::testing::ValuesIn(divisionOperators),
                       ::testing::ValuesIn(divZeros),
                       ::testing::ValuesIn(countings)),
    parameterName);

TEST_P(DivisionTest, MatchesGmpOnBoundaries) {
  for (const auto &[lhs, rhs, reuse] : std::views::cartesian_product(
           boundaries(), boundaries(),
           std::array<std::optional<std::uint32_t>, 2>{std::nullopt, 3}))
    if (rhs.sign() || mode == DivZero::lean)
      checkOwnerships(op, mode, counting, {lhs, 2}, {rhs, 2}, reuse);
}

TEST_P(DivisionTest, MatchesGmpOnConstants) {
  for (std::size_t i = 0; i < constantValues.size(); ++i)
    for (std::size_t j = 0; j < constantValues.size(); ++j)
      if (constantValue(j).sign() || mode == DivZero::lean)
        checkDivision({op,
                       mode,
                       counting,
                       ownerships[0],
                       false,
                       {constantValue(i)},
                       {constantValue(j)},
                       i,
                       j});
}

} // namespace
