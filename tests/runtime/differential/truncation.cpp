// Compares compiled truncations of borrowed integers with GMP, at widths below,
// at, and above the immediate bounds and the word.
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ranges>
#include <vector>
import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.allocator;
import rcgmp.testing.differential;

extern "C" {
#define RCGMP_WIDTH(n)                                                         \
  __int128 trunci_##n(void *);                                                 \
  std::int32_t overflow_signed_##n(void *, __int128 *);                        \
  std::int32_t overflow_unsigned_##n(void *, __int128 *);
RCGMP_WIDTH(1)
RCGMP_WIDTH(8)
RCGMP_WIDTH(32)
RCGMP_WIDTH(61)
RCGMP_WIDTH(62)
RCGMP_WIDTH(63)
RCGMP_WIDTH(64)
RCGMP_WIDTH(65)
RCGMP_WIDTH(127)
RCGMP_WIDTH(128)
#undef RCGMP_WIDTH
__int128 trunci_nsw_8(void *);
__int128 trunci_nsw_32(void *);
__int128 trunci_nsw_62(void *);
__int128 trunci_nsw_64(void *);
__int128 trunci_nuw_8(void *);
__int128 trunci_nuw_32(void *);
__int128 trunci_nuw_61(void *);
__int128 trunci_nuw_64(void *);
std::int64_t trunci_atomic_64(void *);
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

// The compiled truncations to a width. Values are returned sign-extended to
// i128.
struct Width {
  unsigned width;
  __int128 (*trunci)(void *);
  std::int32_t (*overflowSigned)(void *, __int128 *);
  std::int32_t (*overflowUnsigned)(void *, __int128 *);
};

#define RCGMP_WIDTH(n)                                                         \
  {n, trunci_##n, overflow_signed_##n, overflow_unsigned_##n}
constexpr std::array<Width, 10> widths{{
    RCGMP_WIDTH(1),
    RCGMP_WIDTH(8),
    RCGMP_WIDTH(32),
    RCGMP_WIDTH(61),
    RCGMP_WIDTH(62),
    RCGMP_WIDTH(63),
    RCGMP_WIDTH(64),
    RCGMP_WIDTH(65),
    RCGMP_WIDTH(127),
    RCGMP_WIDTH(128),
}};
#undef RCGMP_WIDTH

// A truncation with an overflow flag, which applies only to integers that
// fit. nuw results are zero-extended to i128.
struct Assumed {
  unsigned width;
  bool isUnsigned;
  __int128 (*trunci)(void *);
};

constexpr std::array<Assumed, 8> assumed{{
    {8, false, trunci_nsw_8},
    {32, false, trunci_nsw_32},
    {62, false, trunci_nsw_62},
    {64, false, trunci_nsw_64},
    {8, true, trunci_nuw_8},
    {32, true, trunci_nuw_32},
    {61, true, trunci_nuw_61},
    {64, true, trunci_nuw_64},
}};

Integer fromInt128(__int128 value) {
  bool negative = value < 0;
  auto magnitude = negative ? -static_cast<unsigned __int128>(value)
                            : static_cast<unsigned __int128>(value);
  std::array<std::uintptr_t, 2> words{
      static_cast<std::uintptr_t>(magnitude),
      static_cast<std::uintptr_t>(magnitude >> 64)};
  return Integer::fromWords(words, negative);
}

// The integer modulo 2^width, as a signed or unsigned integer of that width.
Integer wrapped(const Integer &value, unsigned width, bool isUnsigned) {
  Integer result;
  mpz_fdiv_r_2exp(result, value, width);
  if (!isUnsigned && mpz_tstbit(result, width - 1))
    mpz_sub(result, result, Integer::powerOfTwo(width));
  return result;
}

bool fits(const Integer &value, unsigned width, bool isUnsigned) {
  return mpz_cmp(value, wrapped(value, width, isUnsigned)) == 0;
}

// Truncates a borrowed operand, made by makeOperand or a compiled constant,
// with every compiled truncation, and checks that it is neither modified,
// counted, nor freed.
void checkTruncations(void *x, const Integer &value) {
  std::optional<std::uintptr_t> count;
  if (!Handle(x).isImmediate())
    count = references(x, Counting::nonatomic);
  AllocationCounter counter;
  for (const Width &width : widths) {
    SCOPED_TRACE("i" + std::to_string(width.width) + " of " + value.hex());
    EXPECT_EQ(fromInt128(width.trunci(x)), wrapped(value, width.width, false));
    for (bool isUnsigned : {false, true}) {
      __int128 result = 0;
      bool overflow = (isUnsigned ? width.overflowUnsigned
                                  : width.overflowSigned)(x, &result) != 0;
      EXPECT_EQ(fromInt128(result), wrapped(value, width.width, false));
      EXPECT_EQ(overflow, !fits(value, width.width, isUnsigned))
          << (isUnsigned ? "unsigned" : "signed");
    }
  }
  for (const Assumed &truncation : assumed)
    if (fits(value, truncation.width, truncation.isUnsigned))
      EXPECT_EQ(fromInt128(truncation.trunci(x)), value)
          << (truncation.isUnsigned ? "nuw" : "nsw") << " i" << truncation.width
          << " of " << value.hex();
  EXPECT_EQ(counter.allocations(), 0u) << "truncation allocated";
  EXPECT_EQ(counter.deallocations(), 0u) << "truncation freed";
  EXPECT_THAT(x, Represents(value)) << "borrowed operand was modified";
  if (count)
    EXPECT_EQ(references(x, Counting::nonatomic), *count)
        << "borrowed operand was counted";
}

void checkOperand(const Operand &operand, bool immortal) {
  AllocationCounter balance;
  void *x = makeOperand(operand, Counting::nonatomic, immortal);
  checkTruncations(x, operand.value);
  dropOperand(x, Counting::nonatomic, false, immortal);
  EXPECT_TRUE(balance.balanced());
}

// Integers near the signed and unsigned bounds of each width.
fuzztest::Domain<Integer> AnyTruncationInput() {
  return fuzztest::OneOf(AnyInteger(),
                         NearPowersOfTwo({0, 7, 8, 31, 32, 60, 61, 62, 63, 64,
                                          65, 126, 127, 128}));
}

void TruncationMatchesGmp(const Integer &value, std::size_t spare,
                          bool immortal) {
  checkOperand({value, spare}, immortal);
}
FUZZ_TEST(Differential, TruncationMatchesGmp)
    .WithDomains(AnyTruncationInput(), AnySpare(), fuzztest::Arbitrary<bool>());

TEST(Truncation, MatchesGmpOnBoundaries) {
  std::vector<Integer> values = boundaries();
  for (unsigned exponent : {7, 8, 31, 32, 61, 62, 63, 64, 65, 127, 128, 129})
    for (long delta : {-1L, 0L, 1L}) {
      Integer value = Integer::powerOfTwo(exponent) + delta;
      values.push_back(value);
      values.push_back(-value);
    }
  for (const auto &[value, immortal] :
       std::views::cartesian_product(values, std::array{false, true}))
    checkOperand({value, 2}, immortal);
}

TEST(Truncation, MatchesGmpOnConstants) {
  for (std::size_t i = 0; i < constantValues.size(); ++i)
    checkTruncations(constants[0](i), constantValue(i));
}

// Atomic integers truncate without counting.
TEST(Truncation, ReadsAtomicIntegers) {
  Integer value = -(Integer::powerOfTwo(100) + 12345);
  void *x = makeOperand({value, 1}, Counting::atomic);
  EXPECT_EQ(trunci_atomic_64(x), -12345);
  EXPECT_EQ(references(x, Counting::atomic), 1u);
  releases[1](x);
}

} // namespace
