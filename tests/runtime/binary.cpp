// Compiled subtraction, monus, multiplication, AND, XOR, and OR.
#include "RCGMP/Runtime/Mpz.h"
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <array>
#include <climits>
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
void *sub_local(void *, void *);
void *sub_atomic(void *, void *);
void *monus_local(void *, void *);
void *monus_atomic(void *, void *);
void *mul_local(void *, void *);
void *mul_atomic(void *, void *);
void *and_local(void *, void *);
void *and_atomic(void *, void *);
void *xor_local(void *, void *);
void *xor_atomic(void *, void *);
void *or_local(void *, void *);
void *or_atomic(void *, void *);
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

enum class Operator {
  subtract,
  monus,
  multiply,
  bitwiseAnd,
  bitwiseXor,
  bitwiseOr
};

struct Operation {
  std::string_view name;
  // Indexed by counting.
  std::array<void *(*)(void *, void *), 2> compiled;
  Oracle oracle;
};

constexpr std::array<Operation, 6> operations{{
    {"sub", {sub_local, sub_atomic}, mpz_sub},
    {"monus", {monus_local, monus_atomic}, gmpMonus},
    {"mul", {mul_local, mul_atomic}, mpz_mul},
    {"and", {and_local, and_atomic}, mpz_and},
    {"xor", {xor_local, xor_atomic}, mpz_xor},
    {"or", {or_local, or_atomic}, mpz_ior},
}};

constexpr std::array operators{Operator::subtract,   Operator::monus,
                               Operator::multiply,   Operator::bitwiseAnd,
                               Operator::bitwiseXor, Operator::bitwiseOr};

const Operation &operation(Operator op) {
  return operations[std::to_underlying(op)];
}

template <typename Sink> void AbslStringify(Sink &sink, Operator op) {
  sink.Append(operation(op).name);
}

void MatchesGmp(Operator op, const std::pair<Integer, Integer> &operands,
                Counting counting, Sharing sharing) {
  const auto &[lhs, rhs] = operands;
  checkBinary(operation(op).compiled[std::to_underlying(counting)],
              operation(op).oracle, lhs, rhs, counting, sharing);
}
FUZZ_TEST(Binary, MatchesGmp)
    .WithDomains(fuzztest::ElementOf(std::vector(operators.begin(),
                                                 operators.end())),
                 AnyOperands(), AnyCounting(), AnySharing());

class BinaryTest
    : public ::testing::TestWithParam<std::tuple<Operator, Counting>> {
protected:
  void TearDown() override { EXPECT_TRUE(allocations.balanced()); }

  void check(const Integer &lhs, const Integer &rhs, Sharing sharing = {}) {
    checkBinary(apply, oracle, lhs, rhs, counting, sharing);
  }
  Integer expected(const Integer &lhs, const Integer &rhs) const {
    Integer result;
    oracle(result, lhs, rhs);
    return result;
  }

  Operator op = std::get<Operator>(GetParam());
  Counting counting = std::get<Counting>(GetParam());
  bool atomic = counting == Counting::atomic;
  void *(*apply)(void *, void *) = operation(op).compiled[atomic];
  Oracle oracle = operation(op).oracle;
  AllocationCounter allocations;
};

std::string parameterName(
    const ::testing::TestParamInfo<std::tuple<Operator, Counting>> &info) {
  auto [op, counting] = info.param;
  return std::string(operation(op).name) + "_" +
         std::string(rcgmp::name(counting));
}

INSTANTIATE_TEST_SUITE_P(Operations, BinaryTest,
                         ::testing::Combine(::testing::ValuesIn(operators),
                                            ::testing::ValuesIn(countings)),
                         parameterName);

TEST_P(BinaryTest, MatchesGmpOnBoundaries) {
  // Products of operands near the square root of the immediate bound land on
  // either side of it.
  long root = 1L << (sizeof(std::uintptr_t) * CHAR_BIT / 2 - 1);
  std::vector<Integer> boundaries;
  for (long value : {0L, 1L, -1L, 2L, -2L, root - 1, root, -root,
                     Handle::immediateMin, Handle::immediateMax})
    boundaries.emplace_back(value);
  boundaries.push_back(Integer(Handle::immediateMin) - 1);
  boundaries.push_back(Integer(Handle::immediateMax) + 1);
  for (const auto &[lhs, rhs, sharing] :
       std::views::cartesian_product(boundaries, boundaries, sharings))
    check(lhs, rhs, sharing);
}

// Long runs of ones, negative sign extension, and an extra result limb when a
// negative bitwise result is converted back to magnitude.
TEST_P(BinaryTest, MatchesGmpOnLongRuns) {
  Integer minusTwo(-2);
  for (unsigned bits = GMP_NUMB_BITS; bits <= 4 * GMP_NUMB_BITS; ++bits) {
    Integer ones = -(Integer::powerOfTwo(bits) - 1);
    for (Sharing sharing : sharings) {
      check(ones, minusTwo, sharing);
      check(minusTwo, ones, sharing);
    }
  }
}

// Either operand position can reuse the heap allocation, with either sign,
// unless the result is immediate, as a monus that stops at zero is. AND uses
// -1, so that the result remains on the heap.
TEST_P(BinaryTest, ReusesHeapOperand) {
  long small = op == Operator::bitwiseAnd ? -1 : 3;
  Integer rhs(small);
  for (bool negative : {false, true}) {
    Integer lhs = Integer::powerOfTwo(130) + 17;
    if (negative)
      lhs = -lhs;
    for (bool reverse : {false, true}) {
      Integer value = reverse ? expected(rhs, lhs) : expected(lhs, rhs);
      if (value.fitsImmediate())
        continue;
      void *x = toRuntime(lhs), *y = Handle::immediate(small).pointer();
      AllocationCounter counter;
      void *result = reverse ? apply(y, x) : apply(x, y);
      EXPECT_EQ(result, x);
      EXPECT_EQ(counter.allocations(), 0u);
      EXPECT_THAT(result, Represents(value));
      rcgmp_mpz_release(result, atomic);
    }
  }
}

// Two references to the same heap operand, optionally with an external
// alias. Multiplication exercises the disjoint output that mpn_sqr requires.
TEST_P(BinaryTest, AppliesToAliasedOperand) {
  Integer value = -(Integer::powerOfTwo(130) + 17);
  for (bool shared : {false, true}) {
    void *x = toRuntime(value);
    rcgmp_mpz_retain(x, atomic);
    if (shared)
      rcgmp_mpz_retain(x, atomic);
    void *result = apply(x, x);
    EXPECT_THAT(result, Represents(expected(value, value)));
    if (shared) {
      EXPECT_THAT(x, Represents(value));
      rcgmp_mpz_release(x, atomic);
    }
    rcgmp_mpz_release(result, atomic);
  }
}

TEST_P(BinaryTest, AppliesToImmediatesWithoutAllocation) {
  AllocationCounter counter;
  void *result =
      apply(Handle::immediate(-7).pointer(), Handle::immediate(9).pointer());
  EXPECT_TRUE(Handle(result).isImmediate());
  EXPECT_THAT(result, Represents(expected(Integer(-7), Integer(9))));
  EXPECT_EQ(counter.allocations(), 0u);
}

} // namespace
