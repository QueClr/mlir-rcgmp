// Compares compiled greatest common divisors and their runtime entry point
// with GMP on operands that share factors, over the ownership of the operands.
#include <fuzztest/fuzztest.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.differential;

namespace {

using rcgmp::Counting;
using namespace rcgmp::test;

// Each ownership of distinct operands and of aliased ones, through compiled
// code or the runtime with a reuse mask.
void GcdMatchesGmp(Counting counting,
                   const std::pair<Integer, Integer> &operands,
                   std::size_t lhsSpare, std::size_t rhsSpare,
                   std::optional<std::uint32_t> reuse) {
  Operand lhs{operands.first, lhsSpare}, rhs{operands.second, rhsSpare};
  for (Ownership ownership : ownerships)
    checkBinary({Operator::gcdi,
                 counting,
                 ownership,
                 false,
                 lhs,
                 rhs,
                 false,
                 {},
                 {},
                 reuse});
  for (Ownership ownership : aliasOwnerships)
    checkBinary({Operator::gcdi,
                 counting,
                 ownership,
                 true,
                 lhs,
                 lhs,
                 false,
                 {},
                 {},
                 reuse});
}
FUZZ_TEST(Differential, GcdMatchesGmp)
    .WithDomains(AnyCounting(), AnyGcdOperands(), AnySpare(), AnySpare(),
                 fuzztest::OptionalOf(fuzztest::InRange<std::uint32_t>(0, 3)));

} // namespace
