// Compares compiled bit tests with GMP, on operands with stale spare limbs,
// immortal operands, and compiled constants.
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>
import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.allocator;
import rcgmp.testing.differential;

extern "C" {
std::int32_t testbiti_local(void *, std::uint64_t);
std::int32_t testbiti_atomic(void *, std::uint64_t);
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

constexpr std::array testbiti{testbiti_local, testbiti_atomic};

// Tests a bit of a borrowed operand, made by makeOperand or a compiled
// constant, and checks that it is neither modified, counted, nor freed.
void checkBit(void *x, const Integer &value, std::uint64_t index,
              Counting counting) {
  std::optional<std::uintptr_t> count;
  if (!Handle(x).isImmediate())
    count = references(x, counting);
  AllocationCounter counter;
  EXPECT_EQ(testbiti[std::to_underlying(counting)](x, index),
            mpz_tstbit(value, index))
      << value.hex() << " bit " << index;
  EXPECT_EQ(counter.allocations(), 0u) << "bit test allocated";
  EXPECT_EQ(counter.deallocations(), 0u) << "bit test freed";
  EXPECT_THAT(x, Represents(value)) << "borrowed operand was modified";
  if (count)
    EXPECT_EQ(references(x, counting), *count)
        << "borrowed operand was counted";
}

// Integers with zero low limbs, of either sign, and powers of two and their
// neighbours around the limb bounds.
fuzztest::Domain<Integer> AnyBitInput() {
  return fuzztest::OneOf(
      AnyInteger(),
      NearPowersOfTwo({word - 1, word, 2 * word - 1, 2 * word, 3 * word}),
      fuzztest::Map(
          [](std::vector<std::uintptr_t> words, std::size_t zeros,
             bool negative) {
            words.insert(words.begin(), zeros, 0);
            return Integer::fromWords(words, negative);
          },
          fuzztest::VectorOf(fuzztest::Arbitrary<std::uintptr_t>())
              .WithMaxSize(3),
          fuzztest::InRange<std::size_t>(1, 3), fuzztest::Arbitrary<bool>()));
}

// Indices within and just beyond six limbs, at the limb bounds, and the
// largest.
fuzztest::Domain<std::uint64_t> AnyBitIndex() {
  return fuzztest::OneOf(fuzztest::InRange<std::uint64_t>(0, 7 * word),
                         fuzztest::ElementOf<std::uint64_t>(
                             {word - 1, word, 2 * word - 1, 2 * word, 3 * word,
                              std::numeric_limits<std::uint64_t>::max()}));
}

// Spare limbs past the magnitude hold nonzero stale values, which the bit
// test must not read.
void BitMatchesGmp(Counting counting, const Integer &value, std::size_t spare,
                   std::uint64_t index, bool immortal) {
  AllocationCounter balance;
  void *x = makeOperand({value, spare}, counting, immortal);
  checkBit(x, value, index, counting);
  dropOperand(x, counting, false, immortal);
  EXPECT_TRUE(balance.balanced());
}
FUZZ_TEST(Differential, BitMatchesGmp)
    .WithDomains(AnyCounting(), AnyBitInput(), AnySpare(), AnyBitIndex(),
                 fuzztest::Arbitrary<bool>());

TEST(Differential, BitsOfConstants) {
  for (Counting counting : countings)
    for (std::size_t i = 0; i < constantValues.size(); ++i)
      for (std::uint64_t index = 0; index <= 3 * word; ++index)
        checkBit(constants[std::to_underlying(counting)](i), constantValue(i),
                 index, counting);
}

} // namespace
