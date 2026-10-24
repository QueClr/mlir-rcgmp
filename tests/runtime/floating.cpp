// Compiled floating-point conversions compared with exact GMP arithmetic.
#include "RCGMP/Runtime/Mpz.h"
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>
import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.allocator;

extern "C" {
double sitofp_f64_local(void *);
double sitofp_f64_atomic(void *);
float sitofp_f32_local(void *);
float sitofp_f32_atomic(void *);
void *fptosi_f64_local(double);
void *fptosi_f64_atomic(double);
void *fptosi_f32_local(float);
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

// An integral floating-point value as an exact integer.
Integer exact(double value) {
  Integer integer;
  mpz_set_d(integer, value);
  return integer;
}

// The nearest Float to value, ties to even, chosen by exact comparison with
// the neighbours of its truncation. Beyond the largest finite value, the
// upper neighbour is 2^(max exponent), whose tie with the largest finite
// value, which has an odd significand, goes to infinity.
template <typename Float> Float nearest(const Integer &value) {
  using Limits = std::numeric_limits<Float>;
  Integer magnitude;
  mpz_abs(magnitude, value);
  // The truncation, clamped to the largest finite value: GMP's conversion
  // truncates, and narrowing a double may round up by one place.
  Float lower = Limits::max();
  if (mpz_sizeinbase(magnitude, 2) <= Limits::max_exponent) {
    lower = std::min(static_cast<Float>(mpz_get_d(magnitude)), Limits::max());
    if (mpz_cmp(exact(lower), magnitude) > 0)
      lower = std::nextafter(lower, Float(0));
  }
  Float result = lower;
  if (exact(lower) != magnitude) {
    bool overflow = lower == Limits::max();
    Integer upper = overflow ? Integer::powerOfTwo(Limits::max_exponent)
                             : exact(std::nextafter(lower, Limits::infinity()));
    Integer below, above;
    mpz_sub(below, magnitude, exact(lower));
    mpz_sub(above, upper, magnitude);
    int order = mpz_cmp(below, above);
    bool odd =
        (std::bit_cast<std::conditional_t<sizeof(Float) == 4, std::uint32_t,
                                          std::uint64_t>>(lower) &
         1) != 0;
    if (order > 0 || (order == 0 && odd))
      result = overflow ? Limits::infinity()
                        : std::nextafter(lower, Limits::infinity());
  }
  return value.negative() ? -result : result;
}

// Rounding allocates and frees nothing, and leaves the operand and its count
// unchanged.
void checkRounding(const Integer &value, Counting counting) {
  bool atomic = counting == Counting::atomic;
  void *x = toRuntime(value);
  AllocationCounter counter;
  double f64 = nearest<double>(value);
  float f32 = nearest<float>(value);
  EXPECT_EQ((atomic ? sitofp_f64_atomic : sitofp_f64_local)(x), f64) << value;
  EXPECT_EQ((atomic ? sitofp_f32_atomic : sitofp_f32_local)(x), f32) << value;
  EXPECT_EQ((atomic ? rcgmp_mpz_to_f64_atomic : rcgmp_mpz_to_f64)(x), f64)
      << value;
  EXPECT_EQ((atomic ? rcgmp_mpz_to_f32_atomic : rcgmp_mpz_to_f32)(x), f32)
      << value;
  EXPECT_EQ(counter.allocations(), 0u);
  EXPECT_THAT(x, Represents(value));
  if (!Handle(x).isImmediate())
    EXPECT_EQ(references(x, counting), 1u);
  rcgmp_mpz_release(x, atomic);
}

// Integers around the immediate bound, the significand widths, where ties
// arise, and the largest finite values, where rounding overflows.
fuzztest::Domain<Integer> AnyRoundingInput() {
  return fuzztest::OneOf(AnyInteger(), AnyWords(17),
                         NearPowersOfTwo({24, 25, 53, 54, 64, 65, 127, 128, 129,
                                          1023, 1024, 1025}));
}

void RoundsToNearest(const Integer &value, Counting counting) {
  checkRounding(value, counting);
}
FUZZ_TEST(Floating, RoundsToNearest)
    .WithDomains(AnyRoundingInput(), AnyCounting());

// Ties just above the significand width, which go to the even neighbour, and
// the largest finite values and their halfway points to infinity.
TEST(Floating, RoundsOnBoundaries) {
  std::vector<Integer> values{Integer(0), Integer(Handle::immediateMin),
                              Integer(Handle::immediateMax)};
  for (unsigned exponent : {24u, 53u})
    for (long delta : {1L, 2L, 3L}) {
      Integer value = Integer::powerOfTwo(exponent) + delta;
      values.push_back(value);
      values.push_back(-value);
    }
  // The largest finite double is 2^1024 - 2^971, and halfway to 2^1024 is
  // 2^1024 - 2^970; likewise 2^128 - 2^104 and 2^128 - 2^103 for float.
  for (auto [top, low] : {std::pair{1024u, 970u}, {128u, 103u}})
    for (long delta : {-1L, 0L, 1L}) {
      Integer value;
      mpz_sub(value, Integer::powerOfTwo(top), Integer::powerOfTwo(low));
      values.push_back(value + delta);
    }
  for (const auto &[value, counting] :
       std::views::cartesian_product(values, countings))
    checkRounding(value, counting);
  void *beyond = toRuntime(-Integer::powerOfTwo(1024));
  EXPECT_EQ(sitofp_f64_local(beyond), -std::numeric_limits<double>::infinity());
  rcgmp_mpz_release(beyond, 0);
}

// Truncation is exact for every finite value.
void checkTruncation(double value, Counting counting) {
  bool atomic = counting == Counting::atomic;
  void *x = (atomic ? fptosi_f64_atomic : fptosi_f64_local)(value);
  EXPECT_THAT(x, Represents(exact(std::trunc(value)))) << value;
  if (!Handle(x).isImmediate())
    EXPECT_EQ(references(x, counting), 1u);
  rcgmp_mpz_release(x, atomic);
}

void TruncatesExactly(double value, Counting counting) {
  checkTruncation(value, counting);
}
FUZZ_TEST(Floating, TruncatesExactly)
    .WithDomains(
        fuzztest::Filter([](double value) { return std::isfinite(value); },
                         fuzztest::Arbitrary<double>()),
        AnyCounting());

void TruncatesSinglePrecision(float value) {
  void *x = fptosi_f32_local(value);
  EXPECT_THAT(x, Represents(exact(std::trunc(value)))) << value;
  rcgmp_mpz_release(x, 0);
}
FUZZ_TEST(Floating, TruncatesSinglePrecision)
    .WithDomains(
        fuzztest::Filter([](float value) { return std::isfinite(value); },
                         fuzztest::Arbitrary<float>()));

// Signed zeros, fractions, the immediate bound, the largest finite values,
// and the smallest subnormal.
TEST(Floating, TruncatesOnBoundaries) {
  double bound = std::ldexp(1.0, GMP_NUMB_BITS - 3);
  for (double value :
       {0.0, -0.0, 0.5, -0.99, 2.5, -2.5, bound, -bound,
        std::nextafter(bound, 0.0), -std::nextafter(bound, 0.0), 0x1p63, 0x1p64,
        std::numeric_limits<double>::max(), -std::numeric_limits<double>::max(),
        std::numeric_limits<double>::denorm_min()})
    for (Counting counting : countings)
      checkTruncation(value, counting);
}

// A NaN or an infinity has no integer value.
TEST(Floating, RejectsNonFiniteValues) {
  EXPECT_DEATH(fptosi_f64_local(std::numeric_limits<double>::quiet_NaN()), "");
  EXPECT_DEATH(fptosi_f64_atomic(std::numeric_limits<double>::infinity()), "");
  EXPECT_DEATH(fptosi_f32_local(-std::numeric_limits<float>::infinity()), "");
}

} // namespace
