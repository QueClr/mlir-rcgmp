// Compiled addition: values, operand reuse, aliasing, and concurrent sharing.
#include "RCGMP/Runtime/Mpz.h"
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <ranges>
#include <thread>
#include <utility>
#include <vector>
import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.allocator;

extern "C" {
void *add_local(void *, void *);
void *add_atomic(void *, void *);
void *double_local(void *);
void *double_atomic(void *);
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

constexpr std::array additions{add_local, add_atomic};

void *add(void *lhs, void *rhs, Counting counting) {
  return additions[std::to_underlying(counting)](lhs, rhs);
}

void AddsLikeGmp(const std::pair<Integer, Integer> &operands, Counting counting,
                 Sharing sharing) {
  const auto &[lhs, rhs] = operands;
  checkBinary(additions[std::to_underlying(counting)], mpz_add, lhs, rhs,
              counting, sharing);
}
FUZZ_TEST(Addition, AddsLikeGmp)
    .WithDomains(AnyOperands(), AnyCounting(), AnySharing());

class AdditionTest : public ::testing::TestWithParam<Counting> {
protected:
  void TearDown() override { EXPECT_TRUE(allocations.balanced()); }

  Counting counting = GetParam();
  bool atomic = counting == Counting::atomic;
  AllocationCounter allocations;
};

INSTANTIATE_TEST_SUITE_P(Counting, AdditionTest, ::testing::ValuesIn(countings),
                         countingName);

TEST_P(AdditionTest, AddsBoundaries) {
  std::vector<Integer> boundaries;
  for (long value : {0L, 1L, -1L, Handle::immediateMin, Handle::immediateMax})
    boundaries.emplace_back(value);
  boundaries.push_back(Integer(Handle::immediateMin) - 1);
  boundaries.push_back(Integer(Handle::immediateMax) + 1);
  for (const auto &[lhs, rhs, sharing] :
       std::views::cartesian_product(boundaries, boundaries, sharings))
    checkBinary(additions[atomic], mpz_add, lhs, rhs, counting, sharing);
}

// A heap operand holds the sum with a small one, which is never boxed.
TEST_P(AdditionTest, ReusesHeapOperandForImmediate) {
  Integer value = Integer::powerOfTwo(130);
  void *one = Handle::immediate(1).pointer();
  for (bool reverse : {false, true}) {
    void *x = toRuntime(value);
    AllocationCounter counter;
    void *sum = reverse ? add(one, x, counting) : add(x, one, counting);
    EXPECT_EQ(sum, x);
    EXPECT_EQ(counter.allocations(), 0u);
    EXPECT_THAT(sum, Represents(value + 1));
    rcgmp_mpz_release(sum, atomic);
  }
}

// A shorter, unique operand can supply storage while the larger is shared.
TEST_P(AdditionTest, ReusesShorterUniqueOperand) {
  Integer lhs = Integer::powerOfTwo(GMP_NUMB_BITS - 1);
  Integer rhs = -Integer::powerOfTwo(GMP_NUMB_BITS);
  void *x = toRuntime(lhs), *y = toRuntime(rhs);
  rcgmp_mpz_retain(y, atomic);
  AllocationCounter counter;
  void *sum = add(x, y, counting);
  EXPECT_EQ(sum, x);
  EXPECT_EQ(counter.allocations(), 0u);
  Integer expected;
  mpz_add(expected, lhs, rhs);
  EXPECT_THAT(sum, Represents(expected));
  EXPECT_THAT(y, Represents(rhs));
  rcgmp_mpz_release(sum, atomic);
  rcgmp_mpz_release(y, atomic);
}

// Aliased operands consume two references and preserve an external alias.
TEST_P(AdditionTest, DoublesAliasedOperand) {
  Integer value = -Integer::powerOfTwo(GMP_NUMB_BITS);
  void *x = toRuntime(value);
  rcgmp_mpz_retain(x, atomic);
  void *sum = atomic ? double_atomic(x) : double_local(x);
  Integer expected;
  mpz_mul_2exp(expected, value, 1);
  EXPECT_THAT(sum, Represents(expected));
  EXPECT_THAT(x, Represents(value));
  rcgmp_mpz_release(sum, atomic);
  rcgmp_mpz_release(x, atomic);
}

// A carry fills the spare limb; subsequent growth must allocate.
TEST_P(AdditionTest, GrowsPastSpareLimb) {
  std::array<std::uintptr_t, 2> words{~std::uintptr_t{0}, ~std::uintptr_t{0}};
  Integer value = Integer::fromWords(words, false);
  void *x = toRuntime(value), *y = toRuntime(value);
  Integer expected;
  mpz_mul_2exp(expected, value, 1);
  AllocationCounter counter;
  void *sum = add(x, y, counting);
  EXPECT_EQ(sum, x);
  EXPECT_EQ(counter.allocations(), 0u);
  EXPECT_THAT(sum, Represents(expected));
  sum = add(sum, Handle::immediate(1).pointer(), counting);
  EXPECT_EQ(counter.allocations(), 1u);
  EXPECT_THAT(sum, Represents(expected + 1));
  rcgmp_mpz_release(sum, atomic);
}

TEST_P(AdditionTest, AddsImmediatesWithoutAllocation) {
  AllocationCounter counter;
  void *sum = add(Handle::immediate(-7).pointer(),
                  Handle::immediate(9).pointer(), counting);
  EXPECT_EQ(sum, Handle::immediate(2).pointer());
  EXPECT_EQ(counter.allocations(), 0u);
}

// Concurrent readers hold independent references, so none may mutate the
// shared operand. Once they finish, the last reference may be reused.
TEST(Addition, SharesOperandAcrossThreads) {
  AllocationCounter allocations;
  Integer value = Integer::powerOfTwo(200);
  void *shared = toRuntime(value);
  {
    std::vector<std::jthread> readers;
    for (int i = 0; i < 4; ++i)
      readers.emplace_back([&] {
        for (int j = 0; j < 1000; ++j) {
          void *sum = add_atomic(rcgmp_mpz_retain(shared, true),
                                 Handle::immediate(1).pointer());
          EXPECT_THAT(sum, Represents(value + 1));
          rcgmp_mpz_release(sum, true);
        }
      });
  }
  EXPECT_THAT(shared, Represents(value));
  void *sum = add_atomic(shared, Handle::immediate(1).pointer());
  EXPECT_EQ(sum, shared);
  EXPECT_THAT(sum, Represents(value + 1));
  rcgmp_mpz_release(sum, true);
  EXPECT_TRUE(allocations.balanced());
}

} // namespace
