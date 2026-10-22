// Compiled complement, negation, and absolute value.
#include "RCGMP/Runtime/Mpz.h"
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
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
void *not_local(void *);
void *not_atomic(void *);
void *neg_local(void *);
void *neg_atomic(void *);
void *abs_local(void *);
void *abs_atomic(void *);
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

enum class Operator { complement, negate, absolute };

struct Operation {
  std::string_view name;
  // Indexed by counting.
  std::array<void *(*)(void *), 2> compiled;
  UnaryOracle oracle;
};

constexpr std::array<Operation, 3> operations{{
    {"not", {not_local, not_atomic}, mpz_com},
    {"neg", {neg_local, neg_atomic}, mpz_neg},
    {"abs", {abs_local, abs_atomic}, mpz_abs},
}};

constexpr std::array operators{Operator::complement, Operator::negate,
                               Operator::absolute};

const Operation &operation(Operator op) {
  return operations[std::to_underlying(op)];
}

template <typename Sink> void AbslStringify(Sink &sink, Operator op) {
  sink.Append(operation(op).name);
}

void MatchesGmp(Operator op, const Integer &value, Counting counting,
                bool shared) {
  checkUnary(operation(op).compiled[std::to_underlying(counting)],
             operation(op).oracle, value, counting, shared);
}
FUZZ_TEST(Unary, MatchesGmp)
    .WithDomains(fuzztest::ElementOf(std::vector(operators.begin(),
                                                 operators.end())),
                 AnyInteger(), AnyCounting(), fuzztest::Arbitrary<bool>());

class UnaryTest
    : public ::testing::TestWithParam<std::tuple<Operator, Counting>> {
protected:
  void TearDown() override { EXPECT_TRUE(allocations.balanced()); }

  Integer expected(const Integer &value) const {
    Integer result;
    operation(op).oracle(result, value);
    return result;
  }

  Operator op = std::get<Operator>(GetParam());
  Counting counting = std::get<Counting>(GetParam());
  bool atomic = counting == Counting::atomic;
  void *(*apply)(void *) = operation(op).compiled[atomic];
  AllocationCounter allocations;
};

std::string parameterName(
    const ::testing::TestParamInfo<std::tuple<Operator, Counting>> &info) {
  auto [op, counting] = info.param;
  return std::string(operation(op).name) + "_" +
         std::string(rcgmp::name(counting));
}

INSTANTIATE_TEST_SUITE_P(Operations, UnaryTest,
                         ::testing::Combine(::testing::ValuesIn(operators),
                                            ::testing::ValuesIn(countings)),
                         parameterName);

// Negation and absolute value leave the immediate range only at its minimum,
// and 2^(W-3) on the heap negates to that minimum. Magnitudes of a full limb,
// such as 2^W - 1, complement to one more limb when nonnegative and to one
// fewer when negative.
TEST_P(UnaryTest, MatchesGmpOnBoundaries) {
  std::vector<Integer> values;
  for (long value : {0L, 1L, -1L, Handle::immediateMin, Handle::immediateMax})
    values.emplace_back(value);
  values.push_back(Integer(Handle::immediateMin) - 1);
  values.push_back(Integer(Handle::immediateMax) + 1);
  for (unsigned exponent : {GMP_NUMB_BITS, 2 * GMP_NUMB_BITS})
    for (long delta : {-1L, 0L, 1L}) {
      Integer value = Integer::powerOfTwo(exponent) + delta;
      values.push_back(value);
      values.push_back(-value);
    }
  for (const auto &[value, shared] :
       std::views::cartesian_product(values, std::array{false, true}))
    checkUnary(apply, operation(op).oracle, value, counting, shared);
}

// A unique heap operand of either sign holds its result.
TEST_P(UnaryTest, ReusesUniqueHeapOperand) {
  for (bool negative : {false, true}) {
    Integer value = Integer::powerOfTwo(130) + 17;
    if (negative)
      value = -value;
    void *x = toRuntime(value);
    AllocationCounter counter;
    void *result = apply(x);
    EXPECT_EQ(result, x);
    EXPECT_EQ(counter.allocations(), 0u);
    EXPECT_THAT(result, Represents(expected(value)));
    rcgmp_mpz_release(result, atomic);
  }
}

TEST_P(UnaryTest, AppliesToImmediatesWithoutAllocation) {
  for (long value : {-7L, 0L, 9L}) {
    AllocationCounter counter;
    void *result = apply(Handle::immediate(value).pointer());
    EXPECT_TRUE(Handle(result).isImmediate());
    EXPECT_THAT(result, Represents(expected(Integer(value))));
    EXPECT_EQ(counter.allocations(), 0u);
  }
}

// The absolute value of a nonnegative heap operand is the operand itself, even
// when it is shared.
TEST(Unary, AbsoluteKeepsNonnegativeOperand) {
  for (Counting counting : countings) {
    bool atomic = counting == Counting::atomic;
    Integer value = Integer::powerOfTwo(130) + 17;
    void *x = toRuntime(value);
    rcgmp_mpz_retain(x, atomic);
    AllocationCounter counter;
    void *result = (atomic ? abs_atomic : abs_local)(x);
    EXPECT_EQ(result, x);
    EXPECT_EQ(counter.allocations(), 0u);
    EXPECT_EQ(references(x, counting), 2u);
    EXPECT_THAT(result, Represents(value));
    rcgmp_mpz_release(x, atomic);
    rcgmp_mpz_release(result, atomic);
  }
}

} // namespace
