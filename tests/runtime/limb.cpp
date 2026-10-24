// Compiled limb reads, which borrow their operands.
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
std::uint64_t limb_count_local(void *);
std::uint64_t limb_count_atomic(void *);
std::uint64_t limb_local(void *, std::uint64_t);
std::uint64_t limb_atomic(void *, std::uint64_t);
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

constexpr unsigned word = GMP_NUMB_BITS;
constexpr std::array limbCount{limb_count_local, limb_count_atomic};
constexpr std::array limb{limb_local, limb_atomic};

// The count and every limb up to two beyond it, and the limb at the largest
// index, match GMP, allocate and free nothing, and leave the operand and its
// count unchanged.
void checkLimbs(const Integer &value, Counting counting) {
  bool atomic = counting == Counting::atomic;
  auto mode = std::to_underlying(counting);
  void *x = toRuntime(value);
  AllocationCounter counter;
  std::size_t size = mpz_size(value);
  EXPECT_EQ(limbCount[mode](x), size) << value;
  for (std::size_t index = 0; index <= size + 1; ++index)
    EXPECT_EQ(limb[mode](x, index), mpz_getlimbn(value, index))
        << value << " limb " << index;
  EXPECT_EQ(limb[mode](x, std::numeric_limits<std::uint64_t>::max()), 0u)
      << value << " limb at the largest index";
  EXPECT_EQ(counter.allocations(), 0u);
  EXPECT_EQ(counter.deallocations(), 0u);
  EXPECT_THAT(x, Represents(value));
  if (!Handle(x).isImmediate())
    EXPECT_EQ(references(x, counting), 1u);
  rcgmp_mpz_release(x, atomic);
}

void MatchesGmp(const Integer &value, Counting counting) {
  checkLimbs(value, counting);
}
FUZZ_TEST(Limb, MatchesGmp).WithDomains(AnyInteger(), AnyCounting());

// Each sign of zero, one, the immediate bounds, and powers of two and their
// neighbours around one to three limbs.
TEST(Limb, MatchesGmpOnBoundaries) {
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
       std::views::cartesian_product(values, countings))
    checkLimbs(value, counting);
}

} // namespace
