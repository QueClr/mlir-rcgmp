// Compares compiled logarithms with GMP, on operands with stale spare limbs,
// immortal operands, and compiled constants.
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.allocator;
import rcgmp.testing.differential;

extern "C" {
std::int64_t log2i_local(void *);
std::int64_t log2i_atomic(void *);
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

constexpr std::array log2i{log2i_local, log2i_atomic};

// Takes the logarithm of a borrowed operand, made by makeOperand or a compiled
// constant, and checks that it is neither modified, counted, nor freed.
void checkLog(void *x, const Integer &value, Counting counting) {
  std::optional<std::uintptr_t> count;
  if (!Handle(x).isImmediate())
    count = references(x, counting);
  AllocationCounter counter;
  std::uint64_t expected = value.sign() ? mpz_sizeinbase(value, 2) - 1 : 0;
  EXPECT_EQ(static_cast<std::uint64_t>(log2i[std::to_underlying(counting)](x)),
            expected)
      << value.hex();
  EXPECT_EQ(counter.allocations(), 0u) << "logarithm allocated";
  EXPECT_EQ(counter.deallocations(), 0u) << "logarithm freed";
  EXPECT_THAT(x, Represents(value)) << "borrowed operand was modified";
  if (count)
    EXPECT_EQ(references(x, counting), *count)
        << "borrowed operand was counted";
}

// Powers of two and their neighbours around the limb bounds, where the most
// significant limb is one or all ones.
fuzztest::Domain<Integer> AnyLogInput() {
  return fuzztest::OneOf(
      AnyInteger(), NearPowersOfTwo({0, 1, word - 1, word + 1, 2 * word - 1,
                                     3 * word - 1, 5 * word - 1}));
}

// Spare limbs past the magnitude hold nonzero stale values, which the
// logarithm must not read.
void LogMatchesGmp(Counting counting, const Integer &value, std::size_t spare,
                   bool immortal) {
  AllocationCounter balance;
  void *x = makeOperand({value, spare}, counting, immortal);
  checkLog(x, value, counting);
  dropOperand(x, counting, false, immortal);
  EXPECT_TRUE(balance.balanced());
}
FUZZ_TEST(Differential, LogMatchesGmp)
    .WithDomains(AnyCounting(), AnyLogInput(), AnySpare(),
                 fuzztest::Arbitrary<bool>());

TEST(Differential, LogsOfConstants) {
  for (Counting counting : countings)
    for (std::size_t i = 0; i < constantValues.size(); ++i)
      checkLog(constants[std::to_underlying(counting)](i), constantValue(i),
               counting);
}

} // namespace
