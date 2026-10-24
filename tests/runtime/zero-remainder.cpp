// Remainders by plus or minus 2^k that are only compared with zero, computed
// from the low k bits after canonicalization.
#include "RCGMP/Runtime/Mpz.h"

#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.allocator;

extern "C" {
std::int32_t even(void *);
std::int32_t not_multiple_atomic(void *);
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

constexpr unsigned word = GMP_NUMB_BITS;

// Both tests consume the integer: a unique one is freed, and a shared one
// keeps the other reference.
void checkTests(const Integer &value, bool shared) {
  AllocationCounter allocations;
  void *x = toRuntime(value);
  if (shared)
    rcgmp_mpz_retain(x, 0);
  EXPECT_EQ(even(x), mpz_divisible_2exp_p(value, 1) != 0) << value;
  if (shared) {
    EXPECT_THAT(x, Represents(value));
    rcgmp_mpz_release(x, 0);
  }
  void *y = toRuntime(value);
  if (shared)
    rcgmp_mpz_retain(y, 1);
  EXPECT_EQ(not_multiple_atomic(y), mpz_divisible_2exp_p(value, 70) == 0)
      << value;
  if (shared) {
    EXPECT_THAT(y, Represents(value));
    rcgmp_mpz_release(y, 1);
  }
  EXPECT_TRUE(allocations.balanced()) << value;
}

void MatchesGmp(const Integer &value, bool shared) {
  checkTests(value, shared);
}
FUZZ_TEST(ZeroRemainder, MatchesGmp)
    .WithDomains(AnyInteger(), fuzztest::Arbitrary<bool>());

// Each sign of zero, one, the immediate bounds, and multiples of 2^70 and of
// 2 around one to three limbs.
TEST(ZeroRemainder, MatchesGmpOnBoundaries) {
  std::vector<Integer> values{Integer(0), Integer(1), Integer(-1),
                              Integer(Handle::immediateMin),
                              Integer(Handle::immediateMax)};
  for (unsigned exponent : {word - 3, word - 1, 70u, word + 6, 3 * word})
    for (long delta : {-2L, -1L, 0L, 1L, 2L}) {
      Integer value = Integer::powerOfTwo(exponent) + delta;
      values.push_back(value);
      values.push_back(-value);
    }
  for (const Integer &value : values)
    for (bool shared : {false, true})
      checkTests(value, shared);
}

} // namespace
