// Compiled constants: immediates, and immortal heap integers that compiled and
// runtime reference counting leave unchanged and no operation writes into.
#include "RCGMP/Runtime/Mpz.h"
#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <ranges>
import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.allocator;

extern "C" {
void *imm_max();
void *imm_min();
void *above_max();
void *above_max_atomic();
void *below_min();
void *wide();
void *one_limb_max();
void *int64_min();
void *limb_order();
struct Pair {
  void *first, *second;
};
Pair negatives();
void *unbalanced();
void *square();
void *add_constant(void *);
void *add_constant_atomic(void *);
void *addi_local(void *, void *);
void *addi_atomic(void *, void *);
void *subi_local(void *, void *);
void *subi_atomic(void *, void *);
void *muli_local(void *, void *);
void *muli_atomic(void *, void *);
void *andi_local(void *, void *);
void *andi_atomic(void *, void *);
void *xori_local(void *, void *);
void *xori_atomic(void *, void *);
void release_many(std::size_t);
void *accumulate_local(std::size_t);
void *accumulate_atomic(std::size_t);
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

constexpr const char *aboveMax = "2305843009213693952";
constexpr const char *wideValue = "-340282366920938463463374607431768211456";

Integer decimal(const char *digits) {
  Integer value;
  mpz_set_str(value, digits, 10);
  return value;
}

// Values outside the immediate range are immortal integers whose header
// matches a runtime allocation of exactly their magnitude.
void expectConstant(void *value, const char *digits, Counting counting) {
  Integer expected = decimal(digits);
  EXPECT_THAT(value, Represents(expected));
  EXPECT_EQ(Handle(value).isImmediate(), expected.fitsImmediate());
  if (!Handle(value).isImmediate()) {
    EXPECT_EQ(std::bit_cast<std::uintptr_t>(value) % sizeof(void *), 0u);
    EXPECT_EQ(references(value, counting), 0u);
    rcgmp::Mpz &heap = Handle(value).heap();
    EXPECT_EQ(heap.capacity, heap.count());
  }
}

struct Binary {
  std::array<void *(*)(void *, void *), 2> apply;
  Oracle oracle;
};

constexpr std::array<Binary, 5> operations{{
    {{addi_local, addi_atomic}, mpz_add},
    {{subi_local, subi_atomic}, mpz_sub},
    {{muli_local, muli_atomic}, mpz_mul},
    {{andi_local, andi_atomic}, mpz_and},
    {{xori_local, xori_atomic}, mpz_xor},
}};

struct Constant {
  void *(*get)();
  const char *digits;
};

// A constant integer has a zero count in either mode, so each one serves as an
// operand of both modes.
constexpr std::array<Constant, 4> constants{{
    {above_max, "2305843009213693952"},
    {below_min, "-2305843009213693953"},
    {one_limb_max, "18446744073709551615"},
    {wide, "-340282366920938463463374607431768211456"},
}};

void expectConstants(Counting counting) {
  for (const Constant &constant : constants)
    expectConstant(constant.get(), constant.digits, counting);
}

// Applies op to constant c and operand x, with c on the left or the right.
void checkWithConstant(const Binary &op, const Constant &c, void *x,
                       const Integer &value, bool left, Counting counting) {
  bool atomic = counting == Counting::atomic;
  void *constant = c.get();
  Integer expected, other = decimal(c.digits);
  void *result;
  if (left) {
    op.oracle(expected, other, value);
    result = op.apply[atomic](constant, x);
  } else {
    op.oracle(expected, value, other);
    result = op.apply[atomic](x, constant);
  }
  EXPECT_THAT(result, Represents(expected));
  // Adding zero, or subtracting it from the constant, hands the constant back.
  // A heap result is either an operand that is a constant or a fresh integer
  // owned only by the caller.
  if (!value.sign() && (op.oracle == mpz_add || (op.oracle == mpz_sub && left)))
    EXPECT_EQ(result, constant);
  if (!Handle(result).isImmediate()) {
    if (references(result, counting) == 0)
      EXPECT_TRUE(result == constant || result == x);
    else
      EXPECT_EQ(references(result, counting), 1u);
  }
  rcgmp_mpz_release(result, atomic);
  expectConstants(counting);
}

class ConstantTest : public ::testing::TestWithParam<Counting> {
protected:
  Counting counting = GetParam();
  bool atomic = counting == Counting::atomic;
};

INSTANTIATE_TEST_SUITE_P(Countings, ConstantTest,
                         ::testing::ValuesIn(countings), countingName);

TEST(Constant, MaterializesValues) {
  expectConstant(imm_max(), "2305843009213693951", Counting::nonatomic);
  expectConstant(imm_min(), "-2305843009213693952", Counting::nonatomic);
  expectConstant(above_max(), aboveMax, Counting::nonatomic);
  expectConstant(above_max_atomic(), aboveMax, Counting::atomic);
  EXPECT_EQ(above_max(), above_max_atomic());
  expectConstant(below_min(), "-2305843009213693953", Counting::nonatomic);
  expectConstant(wide(), wideValue, Counting::atomic);
  expectConstant(one_limb_max(), "18446744073709551615", Counting::atomic);
  expectConstant(int64_min(), "-9223372036854775808", Counting::nonatomic);
  expectConstant(limb_order(), "55340232221128654850", Counting::nonatomic);
  Pair small = negatives();
  expectConstant(small.first, "-1", Counting::nonatomic);
  expectConstant(small.second, "-5", Counting::nonatomic);

  void *constant = unbalanced();
  EXPECT_EQ(constant, above_max());
  expectConstant(constant, aboveMax, Counting::nonatomic);
}

TEST(Constant, SquaresWideConstant) {
  AllocationCounter balance;
  Integer expected = decimal(wideValue);
  mpz_mul(expected, expected, expected);
  void *result = square();
  EXPECT_THAT(result, Represents(expected));
  rcgmp_mpz_release(result, 0);
  EXPECT_TRUE(balance.balanced());
}

// A constant is never unique, so the runtime does not write into it, even when
// the other operand is a small immediate or a unique heap integer.
TEST_P(ConstantTest, AddingKeepsConstant) {
  AllocationCounter balance;
  const char *digits = atomic ? wideValue : aboveMax;
  for (const char *operand : {"-3", "3", "-18446744073709551621"}) {
    Integer value = decimal(operand);
    Integer expected;
    mpz_add(expected, decimal(digits), value);
    void *x = toRuntime(value);
    void *result = atomic ? add_constant_atomic(x) : add_constant(x);
    EXPECT_THAT(result, Represents(expected)) << "operand = " << value;
    rcgmp_mpz_release(result, atomic);
    expectConstant(atomic ? wide() : above_max(), digits, counting);
  }
  EXPECT_TRUE(balance.balanced());
}

TEST_P(ConstantTest, OperandsMatchGmp) {
  enum class Kind { immediate, unique, shared };
  struct Operand {
    Kind kind;
    Integer value;
  };
  std::array<Operand, 8> operands{{
      {Kind::immediate, Integer(0)},
      {Kind::immediate, Integer(1)},
      {Kind::immediate, Integer(-1)},
      {Kind::immediate, Integer(Handle::immediateMax)},
      {Kind::immediate, Integer(Handle::immediateMin)},
      {Kind::unique, decimal("4611686018427387904")},
      {Kind::unique, decimal("-18446744073709551621")},
      {Kind::shared, decimal("4611686018427387904")},
  }};
  AllocationCounter balance;
  for (const auto &[index, op] : std::views::enumerate(operations)) {
    for (bool left : {false, true}) {
      for (const Constant &constant : constants) {
        SCOPED_TRACE(::testing::Message()
                     << "operation " << index << ", constant "
                     << constant.digits << (left ? " on the left" : ""));
        for (const auto &[kind, value] : operands) {
          void *x = toRuntime(value);
          EXPECT_EQ(Handle(x).isImmediate(), kind == Kind::immediate);
          if (kind == Kind::shared)
            rcgmp_mpz_retain(x, atomic);
          checkWithConstant(op, constant, x, value, left, counting);
          if (kind == Kind::shared) {
            EXPECT_THAT(x, Represents(value));
            rcgmp_mpz_release(x, atomic);
          }
        }
        // Two constants, including the same one twice.
        for (const Constant &other : constants)
          checkWithConstant(op, constant, other.get(), decimal(other.digits),
                            left, counting);
      }
    }
  }
  EXPECT_TRUE(balance.balanced());
}

TEST_P(ConstantTest, RepeatedCountingLeavesConstants) {
  AllocationCounter counter;
  if (!atomic) {
    release_many(0);
    release_many(1);
    release_many(1000);
  }
  for (const Constant &constant : constants) {
    void *c = constant.get();
    for (int n = 0; n < 1000; ++n)
      EXPECT_EQ(rcgmp_mpz_retain(c, atomic), c);
    for (int n = 0; n < 1000; ++n)
      rcgmp_mpz_release(c, atomic);
    EXPECT_EQ(constant.get(), c);
  }
  expectConstants(counting);
  EXPECT_EQ(counter.deallocations(), 0u);
}

TEST_P(ConstantTest, AccumulatesConstants) {
  AllocationCounter balance;
  for (std::size_t count : {0, 1, 1000}) {
    Integer expected = decimal(atomic ? wideValue : "18446744073709551615");
    mpz_mul(expected, expected, expected);
    mpz_mul_ui(expected, expected, count);
    void *result = atomic ? accumulate_atomic(count) : accumulate_local(count);
    EXPECT_THAT(result, Represents(expected)) << "count = " << count;
    rcgmp_mpz_release(result, atomic);
  }
  expectConstants(counting);
  EXPECT_TRUE(balance.balanced());
}

} // namespace
