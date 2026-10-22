// Compares integers that compiled from_int and constants, and the runtime
// constructor, produce with GMP.
#include "RCGMP/Runtime/Mpz.h"
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.allocator;
import rcgmp.testing.differential;

extern "C" {
void *from_signed_i61_local(std::uint64_t);
void *from_unsigned_i61_local(std::uint64_t);
void *from_signed_i62_local(std::uint64_t);
void *from_unsigned_i62_local(std::uint64_t);
void *from_signed_i63_local(std::uint64_t);
void *from_unsigned_i63_local(std::uint64_t);
void *from_signed_i64_local(std::uint64_t);
void *from_unsigned_i64_local(std::uint64_t);
void *from_signed_i65_local(std::uint64_t, std::uint64_t);
void *from_unsigned_i65_local(std::uint64_t, std::uint64_t);
void *from_signed_i128_local(std::uint64_t, std::uint64_t);
void *from_unsigned_i128_local(std::uint64_t, std::uint64_t);
void *from_signed_i129_local(std::uint64_t, std::uint64_t, std::uint64_t);
void *from_unsigned_i129_local(std::uint64_t, std::uint64_t, std::uint64_t);
void *from_signed_i64_atomic(std::uint64_t);
void *from_unsigned_i64_atomic(std::uint64_t);
void *from_signed_i129_atomic(std::uint64_t, std::uint64_t, std::uint64_t);
void *from_unsigned_i129_atomic(std::uint64_t, std::uint64_t, std::uint64_t);
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

using Words = std::array<std::uint64_t, 3>;

template <typename> constexpr std::size_t parameterCount = 0;
template <typename Result, typename... Parameters>
constexpr std::size_t parameterCount<Result (*)(Parameters...)> =
    sizeof...(Parameters);

// Passes as many leading words as the entry point takes.
template <auto entry> void *passWords(const Words &words) {
  return [&]<std::size_t... index>(std::index_sequence<index...>) {
    return entry(words[index]...);
  }(std::make_index_sequence<parameterCount<decltype(entry)>>());
}

// Each entry point converts an operand of the given width, passed as
// ceil(width / 64) 64-bit words, least significant first. The widths cover
// each way of deciding whether a value is immediate, and operands of one, two,
// and three words.
struct FromInt {
  std::string_view name;
  unsigned width;
  bool isUnsigned;
  Counting counting;
  void *(*apply)(const Words &);
};

constexpr std::array<FromInt, 18> fromInts{{
    {"from_signed_i61", 61, false, Counting::nonatomic,
     passWords<from_signed_i61_local>},
    {"from_unsigned_i61", 61, true, Counting::nonatomic,
     passWords<from_unsigned_i61_local>},
    {"from_signed_i62", 62, false, Counting::nonatomic,
     passWords<from_signed_i62_local>},
    {"from_unsigned_i62", 62, true, Counting::nonatomic,
     passWords<from_unsigned_i62_local>},
    {"from_signed_i63", 63, false, Counting::nonatomic,
     passWords<from_signed_i63_local>},
    {"from_unsigned_i63", 63, true, Counting::nonatomic,
     passWords<from_unsigned_i63_local>},
    {"from_signed_i64", 64, false, Counting::nonatomic,
     passWords<from_signed_i64_local>},
    {"from_unsigned_i64", 64, true, Counting::nonatomic,
     passWords<from_unsigned_i64_local>},
    {"from_signed_i65", 65, false, Counting::nonatomic,
     passWords<from_signed_i65_local>},
    {"from_unsigned_i65", 65, true, Counting::nonatomic,
     passWords<from_unsigned_i65_local>},
    {"from_signed_i128", 128, false, Counting::nonatomic,
     passWords<from_signed_i128_local>},
    {"from_unsigned_i128", 128, true, Counting::nonatomic,
     passWords<from_unsigned_i128_local>},
    {"from_signed_i129", 129, false, Counting::nonatomic,
     passWords<from_signed_i129_local>},
    {"from_unsigned_i129", 129, true, Counting::nonatomic,
     passWords<from_unsigned_i129_local>},
    {"from_signed_i64", 64, false, Counting::atomic,
     passWords<from_signed_i64_atomic>},
    {"from_unsigned_i64", 64, true, Counting::atomic,
     passWords<from_unsigned_i64_atomic>},
    {"from_signed_i129", 129, false, Counting::atomic,
     passWords<from_signed_i129_atomic>},
    {"from_unsigned_i129", 129, true, Counting::atomic,
     passWords<from_unsigned_i129_atomic>},
}};

// Passes the low words of input, which the entry point truncates further to
// the operand width.
void checkFromInt(const FromInt &from, const Integer &input) {
  unsigned count = (from.width + 63) / 64;
  Integer expected;
  mpz_fdiv_r_2exp(expected, input, 64 * count);
  Words words{};
  mpz_export(words.data(), nullptr, -1, sizeof(std::uint64_t), 0, 0, expected);
  mpz_fdiv_r_2exp(expected, expected, from.width);
  // A set sign bit makes the ceiling remainder the value minus 2^width.
  if (!from.isUnsigned && mpz_tstbit(expected, from.width - 1))
    mpz_cdiv_r_2exp(expected, expected, from.width);
  AllocationCounter balance;
  void *result = from.apply(words);
  EXPECT_THAT(result, Represents(expected))
      << std::format("{} ({}): words = {:#x} {:#x} {:#x}", from.name,
                     rcgmp::name(from.counting), words[0], words[1], words[2]);
  releases[std::to_underlying(from.counting)](result);
  EXPECT_TRUE(balance.balanced());
}

// Values near the immediate bounds, word boundaries, and the operand width,
// with long runs of ones and zeros, and bits above the operand width.
fuzztest::Domain<Integer> AnyFromIntInput(unsigned width) {
  return fuzztest::OneOf(
      NearPowersOfTwo({word - 3, word - 2, word - 1, word, width - 1, width}),
      AnyWords((width + 63) / 64 + 1));
}

void FromIntMatchesGmp(const std::pair<std::size_t, Integer> &test) {
  checkFromInt(fromInts[test.first], test.second);
}
FUZZ_TEST(Differential, FromIntMatchesGmp)
    .WithDomains(fuzztest::FlatMap(
        [](std::size_t index) {
          return fuzztest::PairOf(fuzztest::Just(index),
                                  AnyFromIntInput(fromInts[index].width));
        },
        fuzztest::InRange<std::size_t>(0, fromInts.size() - 1)));

class FromIntTest : public ::testing::TestWithParam<FromInt> {};

INSTANTIATE_TEST_SUITE_P(Widths, FromIntTest, ::testing::ValuesIn(fromInts),
                         [](const ::testing::TestParamInfo<FromInt> &info) {
                           return std::string(info.param.name) + "_" +
                                  std::string(rcgmp::name(info.param.counting));
                         });

// Zero and one with either sign; both sides of the immediate bounds and of
// 2^(word-2), 2^(word-1), and 2^word; powers of two at two-word boundaries;
// and the operand width's signed minimum and maximum and unsigned maximum.
// Inputs wider than the operand are truncated.
TEST_P(FromIntTest, MatchesGmpOnBoundaries) {
  unsigned width = GetParam().width;
  struct Point {
    unsigned exponent;
    long delta;
    bool negate;
  };
  std::vector<Point> grid{
      {0, -1, false},          {0, 0, false},          {0, 0, true},
      {word - 3, -1, false},   {word - 3, 0, false},   {word - 3, 0, true},
      {word - 3, 1, true},     {word - 2, -1, false},  {word - 2, 0, false},
      {word - 1, -1, false},   {word - 1, 0, false},   {word, -1, false},
      {word, 0, false},        {word, 0, true},        {2 * word - 1, 0, false},
      {2 * word - 1, 0, true}, {2 * word, 0, false},   {2 * word, 0, true},
      {width - 1, 0, true},    {width - 1, -1, false}, {width, -1, false},
  };
  for (auto [exponent, delta, negate] : grid) {
    Integer input = Integer::powerOfTwo(exponent) + delta;
    checkFromInt(GetParam(), negate ? -input : input);
  }
}

// The runtime constructor trims most significant zero words, and a negative
// sign with a zero magnitude makes zero. A new heap integer has one reference.
void FromWordsMatchesGmp(Counting counting, const Integer &magnitude,
                         bool negative, std::size_t padding) {
  std::vector<std::uintptr_t> words = magnitude.words();
  words.resize(words.size() + padding);
  Integer expected = Integer::fromWords(words, negative);
  AllocationCounter balance;
  void *result = rcgmp_mpz_from_words(words.data(), words.size(), negative,
                                      counting == Counting::atomic);
  EXPECT_THAT(result, Represents(expected));
  if (!Handle(result).isImmediate())
    EXPECT_EQ(references(result, counting), 1u);
  releases[std::to_underlying(counting)](result);
  EXPECT_TRUE(balance.balanced());
}
FUZZ_TEST(Differential, FromWordsMatchesGmp)
    .WithDomains(AnyCounting(), AnyInteger(), fuzztest::Arbitrary<bool>(),
                 fuzztest::InRange<std::size_t>(0, 2));

// Each compiled constant holds its value, and one outside the immediate range
// is the same static object with a zero count on every call.
TEST(Differential, ConstantsHoldTheirValues) {
  for (Counting counting : countings) {
    for (std::size_t i = 0; i < constantValues.size(); ++i) {
      SCOPED_TRACE(std::format("constant {} ({})", constantValues[i],
                               rcgmp::name(counting)));
      void *value = constants[std::to_underlying(counting)](i);
      EXPECT_THAT(value, Represents(constantValue(i)));
      if (!Handle(value).isImmediate()) {
        EXPECT_EQ(references(value, counting), 0u);
        EXPECT_EQ(constants[std::to_underlying(counting)](i), value);
      }
    }
  }
}

} // namespace
