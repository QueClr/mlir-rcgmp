// Compiled comparisons, which borrow their operands.
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
std::int32_t cmpi_eq_local(void *, void *);
std::int32_t cmpi_eq_atomic(void *, void *);
std::int32_t cmpi_ne_local(void *, void *);
std::int32_t cmpi_ne_atomic(void *, void *);
std::int32_t cmpi_slt_local(void *, void *);
std::int32_t cmpi_slt_atomic(void *, void *);
std::int32_t cmpi_sle_local(void *, void *);
std::int32_t cmpi_sle_atomic(void *, void *);
std::int32_t cmpi_sgt_local(void *, void *);
std::int32_t cmpi_sgt_atomic(void *, void *);
std::int32_t cmpi_sge_local(void *, void *);
std::int32_t cmpi_sge_atomic(void *, void *);
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

// Compiled comparisons, indexed by counting.
constexpr std::array<Comparisons, 2> comparisons{{
    {cmpi_eq_local, cmpi_ne_local, cmpi_slt_local, cmpi_sle_local,
     cmpi_sgt_local, cmpi_sge_local},
    {cmpi_eq_atomic, cmpi_ne_atomic, cmpi_slt_atomic, cmpi_sle_atomic,
     cmpi_sgt_atomic, cmpi_sge_atomic},
}};

// Distinct operands, and the lhs as both operands, which needs no retain.
// Comparisons allocate and free nothing.
void checkOperands(const Integer &lhs, const Integer &rhs, Counting counting) {
  const Comparisons &compiled = comparisons[std::to_underlying(counting)];
  bool atomic = counting == Counting::atomic;
  void *x = toRuntime(lhs), *y = toRuntime(rhs);
  AllocationCounter counter;
  checkComparison(compiled, x, y, lhs, rhs, counting);
  checkComparison(compiled, x, x, lhs, lhs, counting);
  EXPECT_EQ(counter.allocations(), 0u);
  EXPECT_EQ(counter.deallocations(), 0u);
  rcgmp_mpz_release(x, atomic);
  rcgmp_mpz_release(y, atomic);
}

void MatchesGmp(const std::pair<Integer, Integer> &operands,
                Counting counting) {
  checkOperands(operands.first, operands.second, counting);
}
FUZZ_TEST(Compare, MatchesGmp).WithDomains(AnyOperands(), AnyCounting());

// Each sign around zero and the immediate bounds, and heap magnitudes of one
// and two limbs that differ in the most or the least significant limb. Equal
// values are distinct objects.
TEST(Compare, MatchesGmpOnBoundaries) {
  std::vector<Integer> values{Integer(0),
                              Integer(1),
                              Integer(-1),
                              Integer(Handle::immediateMin),
                              Integer(Handle::immediateMax),
                              Integer(Handle::immediateMin) - 1,
                              Integer(Handle::immediateMax) + 1};
  for (unsigned exponent : {GMP_NUMB_BITS, 2 * GMP_NUMB_BITS})
    for (long delta : {-1L, 0L, 1L}) {
      Integer value = Integer::powerOfTwo(exponent) + delta;
      values.push_back(value);
      values.push_back(-value);
    }
  for (const auto &[lhs, rhs, counting] :
       std::views::cartesian_product(values, values, countings))
    checkOperands(lhs, rhs, counting);
}

} // namespace
