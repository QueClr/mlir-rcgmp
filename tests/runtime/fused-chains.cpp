// Fused chains of operations, against GMP, including chains whose first or
// later operation leaves the immediate range.
#include "RCGMP/Runtime/Mpz.h"

#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.allocator;

extern "C" {
void *collatz_step(void *);
std::int32_t even(void *);
void *polynomial_atomic(void *, void *);
}

namespace {

using rcgmp::Handle;
using namespace rcgmp::test;

constexpr unsigned word = GMP_NUMB_BITS;

// Each chain consumes its operands and returns one owned result.
void checkChains(const Integer &x, const Integer &y) {
  AllocationCounter allocations;
  Integer step;
  mpz_mul_ui(step, x, 3);
  mpz_add_ui(step, step, 1);
  void *result = collatz_step(toRuntime(x));
  EXPECT_THAT(result, Represents(step)) << x;
  rcgmp_mpz_release(result, 0);

  EXPECT_EQ(even(toRuntime(x)), mpz_even_p(static_cast<mpz_srcptr>(x)) != 0)
      << x;

  Integer polynomial;
  mpz_mul(polynomial, x, x);
  mpz_sub(polynomial, polynomial, y);
  mpz_fdiv_q_2exp(polynomial, polynomial, 1);
  void *atomicX = toRuntime(x), *atomicY = toRuntime(y);
  result = polynomial_atomic(atomicX, atomicY);
  EXPECT_THAT(result, Represents(polynomial)) << x << ", " << y;
  rcgmp_mpz_release(result, 1);
  EXPECT_TRUE(allocations.balanced()) << x << ", " << y;
}

void MatchesGmp(const Integer &x, const Integer &y) { checkChains(x, y); }
FUZZ_TEST(FusedChains, MatchesGmp).WithDomains(AnyInteger(), AnyInteger());

// Values whose triple, or the triple plus one, or whose square leaves the
// immediate range, and heap values, of each sign.
TEST(FusedChains, MatchesGmpOnBoundaries) {
  Integer max(Handle::immediateMax), min(Handle::immediateMin);
  std::vector<Integer> values{Integer(0), Integer(1), Integer(-1), max, min};
  for (long delta : {-2L, -1L, 0L, 1L, 2L}) {
    Integer third;
    mpz_fdiv_q_ui(third, max, 3);
    values.push_back(third + delta);
    values.push_back(-(third + delta));
    Integer root;
    mpz_sqrt(root, max);
    values.push_back(root + delta);
    values.push_back(-(root + delta));
    values.push_back(Integer::powerOfTwo(word) + delta);
  }
  for (const Integer &x : values)
    for (const Integer &y : {Integer(0), Integer(5), max, -max})
      checkChains(x, y);
}

} // namespace
