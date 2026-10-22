// Compares compiled comparisons and the runtime comparison with GMP, on
// operands with stale spare limbs, immortal operands, and compiled constants.
#include <fuzztest/fuzztest.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <utility>
import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.differential;

namespace {

using rcgmp::Counting;
using namespace rcgmp::test;

// Spare limbs hold stale values that differ between the operands, which a
// comparison must not read.
void ComparisonsMatchGmp(Counting counting,
                         const std::pair<Integer, Integer> &operands,
                         std::size_t lhsSpare, std::size_t rhsSpare,
                         Immortality immortal) {
  Operand lhs{operands.first, lhsSpare}, rhs{operands.second, rhsSpare};
  checkComparison({counting, immortal, false, lhs, rhs});
  checkComparison({counting, immortal, true, lhs, lhs});
}
FUZZ_TEST(Differential, ComparisonsMatchGmp)
    .WithDomains(AnyCounting(), AnyOperands(), AnySpare(), AnySpare(),
                 AnyImmortality());

// A compiled constant replaces either operand.
void ConstantComparisonsMatchGmp(Counting counting, std::size_t constant,
                                 const Integer &other, std::size_t spare,
                                 bool immortal, bool left) {
  Operand c{constantValue(constant)}, y{other, spare};
  ComparisonCase test{counting, {false, false}, false, c, y, constant};
  test.immortal.rhs = immortal;
  if (!left) {
    std::swap(test.lhs, test.rhs);
    std::swap(test.lhsConstant, test.rhsConstant);
    std::swap(test.immortal.lhs, test.immortal.rhs);
  }
  checkComparison(test);
}
FUZZ_TEST(Differential, ConstantComparisonsMatchGmp)
    .WithDomains(AnyCounting(), AnyConstant(), AnyInteger(), AnySpare(),
                 fuzztest::Arbitrary<bool>(), fuzztest::Arbitrary<bool>());

// Every pair of compiled constants. A constant compared with itself is the
// same object or immediate on both sides.
TEST(Differential, ComparesConstants) {
  for (Counting counting : countings)
    for (std::size_t i = 0; i < constantValues.size(); ++i)
      for (std::size_t j = 0; j < constantValues.size(); ++j)
        checkComparison({counting,
                         {false, false},
                         false,
                         {constantValue(i)},
                         {constantValue(j)},
                         i,
                         j});
}

} // namespace
