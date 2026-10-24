// Compiled bit tests, which borrow their operands.
#include "RCGMP/Runtime/Mpz.h"
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <utility>
#include <vector>
import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.allocator;

extern "C" {
std::int32_t testbiti_local(void *, std::uint64_t);
std::int32_t testbiti_atomic(void *, std::uint64_t);
std::int32_t testbiti_8(void *, std::uint8_t);
std::int32_t testbiti_128(void *, unsigned __int128);
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

constexpr unsigned word = GMP_NUMB_BITS;
constexpr std::array testbiti{testbiti_local, testbiti_atomic};

// Bit tests at index, at the index truncated to 8 bits, and at the index plus
// 2^64, which is the sign, allocate and free nothing, and leave the operand
// and its count unchanged.
void checkBit(const Integer &value, std::uint64_t index, Counting counting) {
  bool atomic = counting == Counting::atomic;
  void *x = toRuntime(value);
  AllocationCounter counter;
  EXPECT_EQ(testbiti[std::to_underlying(counting)](x, index),
            mpz_tstbit(value, index))
      << value << " bit " << index;
  if (!atomic) {
    std::uint8_t narrow = index;
    EXPECT_EQ(testbiti_8(x, narrow), mpz_tstbit(value, narrow))
        << value << " bit " << unsigned{narrow};
    EXPECT_EQ(testbiti_128(x, index), mpz_tstbit(value, index))
        << value << " bit " << index;
    EXPECT_EQ(testbiti_128(x, static_cast<unsigned __int128>(1) << 64 | index),
              value.negative())
        << value << " bit 2^64 + " << index;
  }
  EXPECT_EQ(counter.allocations(), 0u);
  EXPECT_EQ(counter.deallocations(), 0u);
  EXPECT_THAT(x, Represents(value));
  if (!Handle(x).isImmediate())
    EXPECT_EQ(references(x, counting), 1u);
  rcgmp_mpz_release(x, atomic);
}

// Integers whose low limbs are zero, so that a negative one has zero low limbs
// in its two's complement too.
fuzztest::Domain<Integer> AnyZeroLowLimbs() {
  return fuzztest::Map(
      [](std::vector<std::uintptr_t> words, std::size_t zeros, bool negative) {
        words.insert(words.begin(), zeros, 0);
        return Integer::fromWords(words, negative);
      },
      fuzztest::VectorOf(fuzztest::Arbitrary<std::uintptr_t>()).WithMaxSize(3),
      fuzztest::InRange<std::size_t>(1, 3), fuzztest::Arbitrary<bool>());
}

// Indices up to six limbs, around the immediate and limb bounds, and the
// largest.
fuzztest::Domain<std::uint64_t> AnyIndex() {
  return fuzztest::OneOf(
      fuzztest::InRange<std::uint64_t>(0, 6 * word),
      fuzztest::ElementOf<std::uint64_t>(
          {word - 3, word - 2, word - 1, word, 2 * word - 1, 2 * word, 3 * word,
           std::numeric_limits<std::uint64_t>::max()}));
}

void MatchesGmp(const Integer &value, std::uint64_t index, Counting counting) {
  checkBit(value, index, counting);
}
FUZZ_TEST(TestBit, MatchesGmp)
    .WithDomains(fuzztest::OneOf(AnyInteger(), AnyZeroLowLimbs()), AnyIndex(),
                 AnyCounting());

// Each sign of zero, one, the immediate bounds, and powers of two around one
// to three limbs, whose negations have zero low limbs, at every index up to
// four limbs and at the largest index.
TEST(TestBit, MatchesGmpOnBoundaries) {
  std::vector<Integer> values{Integer(0), Integer(1), Integer(-1),
                              Integer(Handle::immediateMin),
                              Integer(Handle::immediateMax)};
  for (unsigned exponent : {word - 3, word - 1, word, 2 * word, 3 * word})
    for (long delta : {-1L, 0L, 1L}) {
      Integer value = Integer::powerOfTwo(exponent) + delta;
      values.push_back(value);
      values.push_back(-value);
    }
  for (const auto &[value, counting] :
       std::views::cartesian_product(values, countings)) {
    for (std::uint64_t index = 0; index <= 4 * word; ++index)
      checkBit(value, index, counting);
    checkBit(value, std::numeric_limits<std::uint64_t>::max(), counting);
  }
}

} // namespace
