// Compares compiled exact quotients and their runtime entry points with
// mpz_divexact on multiples, over the ownership of their operands and the
// modes for a zero divisor. Only lean quotients receive zero divisors, with
// the only multiple of zero, zero; zero divisors of the others are tested to
// terminate in exact-division.cpp.
#include "RCGMP/Runtime/Mpz.h"
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>
import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.allocator;
import rcgmp.testing.differential;

extern "C" {
#define RCGMP_ENTRY(name)                                                      \
  void *name##_exact_local(void *, void *);                                    \
  void *name##_exact_atomic(void *, void *);                                   \
  void *name##_exact_undefined_local(void *, void *);                          \
  void *name##_exact_undefined_atomic(void *, void *);                         \
  void *name##_exact_lean_local(void *, void *);                               \
  void *name##_exact_lean_atomic(void *, void *);
RCGMP_ENTRY(divsi)
RCGMP_ENTRY(floordivsi)
RCGMP_ENTRY(edivsi)
#undef RCGMP_ENTRY
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

enum class ExactOperator { divsi, floordivsi, edivsi };

// The compiled entry points of an exact quotient, indexed by mode and
// counting. Every exact quotient calls the same runtime entry points.
struct ExactOperation {
  std::string_view name;
  std::array<std::array<void *(*)(void *, void *), 2>, 3> compiled;
};

#define RCGMP_ENTRY(name)                                                      \
  ExactOperation {                                                             \
    #name, {                                                                   \
      {                                                                        \
        {name##_exact_local, name##_exact_atomic},                             \
            {name##_exact_undefined_local, name##_exact_undefined_atomic}, {   \
          name##_exact_lean_local, name##_exact_lean_atomic                    \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }
constexpr std::array exactOperations{
    RCGMP_ENTRY(divsi), RCGMP_ENTRY(floordivsi), RCGMP_ENTRY(edivsi)};
#undef RCGMP_ENTRY

constexpr std::array runtime{rcgmp_mpz_tdiv_q_exact,
                             rcgmp_mpz_tdiv_q_exact_atomic};

const ExactOperation &operation(ExactOperator op) {
  return exactOperations[std::to_underlying(op)];
}

template <typename Sink> void AbslStringify(Sink &sink, ExactOperator op) {
  sink.Append(operation(op).name);
}

// As for the other divisions, a shared operand has an extra reference held by
// the caller, an aliased rhs is a retained reference to the lhs, and an
// immortal heap operand has a zero reference count.
struct ExactCase {
  ExactOperator op;
  DivZero mode;
  Counting counting;
  Ownership ownership;
  bool aliased;
  Operand lhs, rhs;
  // Calls the runtime entry point instead of compiled code, with this reuse
  // mask limited to the operands that compiled code could pass.
  std::optional<std::uint32_t> reuse;

  std::string describe() const {
    auto [sharing, immortal] = ownership;
    std::string entry =
        reuse ? std::format(", runtime with reuse mask {}", *reuse) : "";
    return std::format(
        "{} exact {} ({}{}{}{}{}{}{}): lhs = {}, rhs = {}", operation(op).name,
        name(mode), rcgmp::name(counting), aliased ? ", aliased" : "",
        sharing.lhs ? ", shared lhs" : "", sharing.rhs ? ", shared rhs" : "",
        immortal.lhs ? ", immortal lhs" : "",
        immortal.rhs ? ", immortal rhs" : "", entry, lhs.describe(),
        rhs.describe());
  }
};

void checkExact(const ExactCase &test) {
  SCOPED_TRACE(test.describe());
  std::size_t index = std::to_underlying(test.counting);
  auto [sharing, immortal] = test.ownership;
  const Integer &lhs = test.lhs.value;
  const Integer &rhs = test.aliased ? lhs : test.rhs.value;
  AllocationCounter balance;
  void *x = makeOperand(test.lhs, test.counting, immortal.lhs);
  void *y = x;
  if (test.aliased)
    EXPECT_EQ(retains[index](x), x);
  else
    y = makeOperand(test.rhs, test.counting, immortal.rhs);
  bool immortalX = !Handle(x).isImmediate() && immortal.lhs;
  bool immortalY =
      y == x ? immortalX : !Handle(y).isImmediate() && immortal.rhs;
  if (sharing.lhs)
    EXPECT_EQ(retains[index](x), x);
  if (sharing.rhs)
    EXPECT_EQ(retains[index](y), y);
  // Compiled code permits reusing each operand that is a uniquely owned,
  // counted heap integer.
  bool unique = !sharing.lhs && !sharing.rhs && !test.aliased;
  std::uint32_t uniqueness =
      (unique && !Handle(x).isImmediate() && !immortalX) |
      (unique && !Handle(y).isImmediate() && !immortalY) << 1;
  std::uint32_t reuse = test.reuse.value_or(3) & uniqueness;
  Integer expected;
  if (rhs.sign())
    mpz_divexact(expected, lhs, rhs);
  AllocationCounter counter;
  void *q =
      test.reuse
          ? runtime[index](x, y, reuse)
          : operation(test.op).compiled[std::to_underlying(test.mode)][index](
                x, y);
  EXPECT_THAT(q, Represents(expected));
  // A unique dividend holds the quotient, which is no longer.
  if (reuse & 1) {
    EXPECT_EQ(counter.allocations(), 0u)
        << "quotient did not reuse the unique dividend";
    if (!Handle(q).isImmediate())
      EXPECT_EQ(q, x) << "quotient did not reuse the unique dividend";
  }
  if (sharing.lhs)
    releaseShared(test.counting, x, lhs, q, immortalX);
  if (sharing.rhs)
    releaseShared(test.counting, y, rhs, q, immortalY);
  releases[index](q);
  if (immortalX)
    dropImmortal(test.counting, x, lhs, false);
  if (immortalY && y != x)
    dropImmortal(test.counting, y, rhs, false);
  EXPECT_TRUE(balance.balanced())
      << "allocations and deallocations do not balance";
}

// Each ownership of the multiple lhs of rhs, and of rhs aliased as both
// operands. Only compiled lean quotients divide by zero, as the runtime
// terminates on a zero divisor; the others divide zero by one instead.
void checkOwnerships(ExactOperator op, DivZero mode, Counting counting,
                     const Operand &lhs, const Operand &rhs,
                     std::optional<std::uint32_t> reuse = std::nullopt) {
  Operand divisor = rhs.value.sign() || (mode == DivZero::lean && !reuse)
                        ? rhs
                        : Operand{Integer(1), rhs.spare};
  for (const Ownership &ownership : ownerships)
    checkExact({op, mode, counting, ownership, false, lhs, divisor, reuse});
  if (divisor.value.sign())
    for (const Ownership &ownership : aliasOwnerships)
      checkExact(
          {op, mode, counting, ownership, true, divisor, divisor, reuse});
}

fuzztest::Domain<ExactOperator> AnyExactOperator() {
  return fuzztest::ElementOf(
      {ExactOperator::divsi, ExactOperator::floordivsi, ExactOperator::edivsi});
}

// Multiples q f 2^k of divisors f 2^k. Factors near the immediate bounds make
// quotients that leave the immediate range or land on it, and factors of up
// to eight words make operands of different lengths. Shifts of up to three
// words make divisors that end in zero bits and zero limbs. A zero factor
// makes zero, which is the only multiple of zero.
fuzztest::Domain<std::pair<Integer, Integer>> AnyMultiples() {
  auto factor = fuzztest::OneOf(AnyInteger(), AnyWords(8));
  return fuzztest::Map(
      [](const Integer &quotient, const Integer &factor, unsigned shift) {
        Integer divisor, multiple;
        mpz_mul_2exp(divisor, factor, shift);
        mpz_mul(multiple, quotient, divisor);
        return std::pair{multiple, divisor};
      },
      factor, factor, fuzztest::InRange(0u, 3 * word));
}

void ExactMatchesGmp(ExactOperator op, DivZero mode, Counting counting,
                     const std::pair<Integer, Integer> &operands,
                     std::size_t lhsSpare, std::size_t rhsSpare,
                     std::optional<std::uint32_t> reuse) {
  checkOwnerships(op, mode, counting, {operands.first, lhsSpare},
                  {operands.second, rhsSpare}, reuse);
}
FUZZ_TEST(Differential, ExactMatchesGmp)
    .WithDomains(AnyExactOperator(), AnyDivZero(), AnyCounting(),
                 AnyMultiples(), AnySpare(), AnySpare(),
                 fuzztest::OptionalOf(fuzztest::InRange<std::uint32_t>(0, 3)));

using ExactParameter = std::tuple<ExactOperator, DivZero, Counting>;

class ExactTest : public ::testing::TestWithParam<ExactParameter> {
protected:
  ExactOperator op = std::get<ExactOperator>(GetParam());
  DivZero mode = std::get<DivZero>(GetParam());
  Counting counting = std::get<Counting>(GetParam());
};

std::string
parameterName(const ::testing::TestParamInfo<ExactParameter> &info) {
  auto [op, mode, counting] = info.param;
  return std::format("{}_{}_{}", operation(op).name, name(mode),
                     rcgmp::name(counting));
}

INSTANTIATE_TEST_SUITE_P(
    Exact, ExactTest,
    ::testing::Combine(::testing::Values(ExactOperator::divsi,
                                         ExactOperator::floordivsi,
                                         ExactOperator::edivsi),
                       ::testing::ValuesIn(divZeros),
                       ::testing::ValuesIn(countings)),
    parameterName);

// Every product of boundary values by a boundary divisor.
TEST_P(ExactTest, MatchesGmpOnBoundaries) {
  for (const auto &[quotient, divisor, reuse] : std::views::cartesian_product(
           boundaries(), boundaries(),
           std::array<std::optional<std::uint32_t>, 2>{std::nullopt, 3})) {
    Integer multiple;
    mpz_mul(multiple, quotient, divisor);
    checkOwnerships(op, mode, counting, {multiple, 2}, {divisor, 2}, reuse);
  }
}

} // namespace
