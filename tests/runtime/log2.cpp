// Compiled logarithms, which borrow their operands.
#include "RCGMP/Runtime/Mpz.h"
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <ranges>
#include <utility>
#include <vector>
import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.allocator;

extern "C" {
std::int64_t log2i_local(void *);
std::int64_t log2i_atomic(void *);
std::uint8_t log2i_8(void *);
unsigned __int128 log2i_128(void *);
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

constexpr std::array log2i{log2i_local, log2i_atomic};

// The logarithm of the magnitude, rounded down, and zero for zero.
std::uint64_t expectedLog(const Integer &value) {
  return value.sign() ? mpz_sizeinbase(value, 2) - 1 : 0;
}

// Logarithms allocate and free nothing, and leave the operand and its count
// unchanged.
void checkLog(const Integer &value, Counting counting) {
  bool atomic = counting == Counting::atomic;
  void *x = toRuntime(value);
  AllocationCounter counter;
  std::uint64_t expected = expectedLog(value);
  EXPECT_EQ(static_cast<std::uint64_t>(log2i[std::to_underlying(counting)](x)),
            expected)
      << value;
  if (!atomic) {
    EXPECT_EQ(log2i_8(x), static_cast<std::uint8_t>(expected)) << value;
    unsigned __int128 wide = log2i_128(x);
    EXPECT_EQ(static_cast<std::uint64_t>(wide), expected) << value;
    EXPECT_EQ(static_cast<std::uint64_t>(wide >> 64), 0u) << value;
  }
  EXPECT_EQ(counter.allocations(), 0u);
  EXPECT_EQ(counter.deallocations(), 0u);
  EXPECT_THAT(x, Represents(value));
  if (!Handle(x).isImmediate())
    EXPECT_EQ(references(x, counting), 1u);
  rcgmp_mpz_release(x, atomic);
}

// Integers whose logarithms change: powers of two and their neighbours at
// limb and immediate bounds, and magnitudes of up to five limbs, including
// most significant limbs of one and of all ones.
fuzztest::Domain<Integer> AnyLogInput() {
  constexpr unsigned word = GMP_NUMB_BITS;
  return fuzztest::OneOf(
      AnyInteger(), NearPowersOfTwo({0, 1, 2, word - 4, word - 3, word - 2,
                                     word - 1, word + 1, 2 * word - 1,
                                     3 * word - 1, 3 * word, 5 * word - 1}));
}

void MatchesGmp(const Integer &value, Counting counting) {
  checkLog(value, counting);
}
FUZZ_TEST(Log2, MatchesGmp).WithDomains(AnyLogInput(), AnyCounting());

// Zero, each sign of one, the immediate bounds, and powers of two around them
// and around one to three limbs.
TEST(Log2, MatchesGmpOnBoundaries) {
  std::vector<Integer> values{Integer(0), Integer(1), Integer(-1),
                              Integer(Handle::immediateMin),
                              Integer(Handle::immediateMax)};
  for (unsigned exponent :
       {1, 2, GMP_NUMB_BITS - 3, GMP_NUMB_BITS - 1, GMP_NUMB_BITS,
        2 * GMP_NUMB_BITS - 1, 2 * GMP_NUMB_BITS, 3 * GMP_NUMB_BITS})
    for (long delta : {-1L, 0L, 1L}) {
      Integer value = Integer::powerOfTwo(exponent) + delta;
      values.push_back(value);
      values.push_back(-value);
    }
  for (const auto &[value, counting] :
       std::views::cartesian_product(values, countings))
    checkLog(value, counting);
}

} // namespace
