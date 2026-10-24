// Compares compiled limb reads with GMP, on operands with stale spare limbs,
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
std::uint64_t limb_count_local(void *);
std::uint64_t limb_count_atomic(void *);
std::uint64_t limb_local(void *, std::uint64_t);
std::uint64_t limb_atomic(void *, std::uint64_t);
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

constexpr std::array limbCount{limb_count_local, limb_count_atomic};
constexpr std::array limb{limb_local, limb_atomic};

// Reads the count and the limbs up to two beyond it of a borrowed operand,
// made by makeOperand or a compiled constant, and checks that it is neither
// modified, counted, nor freed.
void checkLimbs(void *x, const Integer &value, Counting counting) {
  auto mode = std::to_underlying(counting);
  std::optional<std::uintptr_t> count;
  if (!Handle(x).isImmediate())
    count = references(x, counting);
  AllocationCounter counter;
  std::size_t size = mpz_size(value);
  EXPECT_EQ(limbCount[mode](x), size) << value.hex();
  for (std::size_t index = 0; index <= size + 1; ++index)
    EXPECT_EQ(limb[mode](x, index), mpz_getlimbn(value, index))
        << value.hex() << " limb " << index;
  EXPECT_EQ(counter.allocations(), 0u) << "limb read allocated";
  EXPECT_EQ(counter.deallocations(), 0u) << "limb read freed";
  EXPECT_THAT(x, Represents(value)) << "borrowed operand was modified";
  if (count)
    EXPECT_EQ(references(x, counting), *count)
        << "borrowed operand was counted";
}

// Spare limbs past the magnitude hold nonzero stale values, which reads past
// the count must not return.
void LimbsMatchGmp(Counting counting, const Integer &value, std::size_t spare,
                   bool immortal) {
  AllocationCounter balance;
  void *x = makeOperand({value, spare}, counting, immortal);
  checkLimbs(x, value, counting);
  dropOperand(x, counting, false, immortal);
  EXPECT_TRUE(balance.balanced());
}
FUZZ_TEST(Differential, LimbsMatchGmp)
    .WithDomains(AnyCounting(),
                 fuzztest::OneOf(AnyInteger(),
                                 NearPowersOfTwo({word - 1, word, 2 * word - 1,
                                                  2 * word, 3 * word})),
                 AnySpare(), fuzztest::Arbitrary<bool>());

TEST(Differential, LimbsOfConstants) {
  for (Counting counting : countings)
    for (std::size_t i = 0; i < constantValues.size(); ++i)
      checkLimbs(constants[std::to_underlying(counting)](i), constantValue(i),
                 counting);
}

} // namespace
