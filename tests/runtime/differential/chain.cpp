// Compares chains of compiled operations with GMP, where each result is the
// next lhs, as it carries headroom and stale limbs from earlier operations.
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <ranges>
#include <utility>
#include <variant>
#include <vector>
import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.allocator;
import rcgmp.testing.differential;

namespace {

using rcgmp::Counting;
using namespace rcgmp::test;

// A step of a chain applies an operation to the result so far and either a
// fresh operand, one related to the result so far, or a compiled constant.
struct Step {
  Operator op;
  std::variant<Integer, Relation, Constant> operand;
  std::size_t spare;
};

// Each chain feeds every result into the next operation as its uniquely owned
// lhs, so operands also carry the growth headroom and the stale limbs above the
// magnitude that the runtime leaves behind, and come from earlier reuse. A
// constant rhs may also come back as the result and become the next lhs.
void ChainsMatchGmp(Counting counting, const Operand &start,
                    const std::vector<Step> &steps) {
  std::size_t index = std::to_underlying(counting);
  AllocationCounter balance;
  Integer accumulator = start.value;
  void *result = makeOperand(start, counting);
  for (const auto &[position, step] : std::views::enumerate(steps)) {
    // Keep repeated products from growing without bound.
    if (step.op == Operator::muli && mpz_sizeinbase(accumulator, 2) > 1000)
      continue;
    auto *relation = std::get_if<Relation>(&step.operand);
    auto *constant = std::get_if<Constant>(&step.operand);
    Integer operand = relation   ? (*relation)(accumulator)
                      : constant ? constantValue(constant->index)
                                 : std::get<Integer>(step.operand);
    void *rhs = constant ? constants[index](constant->index)
                         : makeOperand({operand, step.spare}, counting);
    // Compare the result so far, with the headroom and stale limbs that
    // earlier operations leave behind, before the operation consumes it.
    checkComparison(comparisons[index], result, rhs, accumulator, operand,
                    counting);
    Integer expected;
    operation(step.op).oracle(expected, accumulator, operand);
    result = operation(step.op).compiled[index](result, rhs);
    ASSERT_THAT(result, Represents(expected))
        << operation(step.op).name << " at step " << position + 1 << " of "
        << steps.size() << ": lhs = " << accumulator << ", rhs = " << operand;
    accumulator = std::move(expected);
  }
  releases[index](result);
  EXPECT_TRUE(balance.balanced())
      << "allocations and deallocations do not balance";
}
FUZZ_TEST(Differential, ChainsMatchGmp)
    .WithDomains(
        AnyCounting(), AnyOperand(),
        fuzztest::VectorOf(fuzztest::StructOf<Step>(
                               AnyOperator(),
                               fuzztest::VariantOf(
                                   AnyInteger(), AnyRelation(),
                                   fuzztest::StructOf<Constant>(AnyConstant())),
                               AnySpare()))
            .WithMinSize(8)
            .WithMaxSize(16));

} // namespace
