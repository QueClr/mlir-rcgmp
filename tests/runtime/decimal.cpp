// Decimal conversion through the runtime entry points, compared with GMP.
#include "RCGMP/Runtime/Mpz.h"
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <string>
#include <string_view>
#include <vector>
import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.allocator;

namespace {

using rcgmp::Handle;
using namespace rcgmp::test;

std::string gmpDecimal(const Integer &value) {
  std::string digits(mpz_sizeinbase(value, 10) + 2, '\0');
  mpz_get_str(digits.data(), 10, value);
  digits.resize(digits.find('\0'));
  return digits;
}

// Writing allocates nothing, stays within the reported size, and leaves the
// operand and its count unchanged. Parsing the text yields the value again,
// allocating only the storage it returns.
void checkRoundTrip(const Integer &value) {
  constexpr char canary = '\x7f';
  void *x = toRuntime(value);
  std::size_t size = rcgmp_mpz_decimal_size(x);
  std::string buffer(size + 16, canary);
  AllocationCounter writing;
  std::size_t length = rcgmp_mpz_to_decimal(x, buffer.data());
  EXPECT_EQ(writing.allocations(), 0u);
  ASSERT_LT(length, size);
  EXPECT_EQ(buffer.substr(0, length), gmpDecimal(value));
  EXPECT_TRUE(std::ranges::all_of(buffer.substr(size),
                                  [](char byte) { return byte == canary; }));
  EXPECT_THAT(x, Represents(value));
  if (!Handle(x).isImmediate())
    EXPECT_EQ(references(x, rcgmp::Counting::nonatomic), 1u);
  rcgmp_mpz_release(x, 0);

  AllocationCounter parsing;
  void *y = rcgmp_mpz_from_decimal(buffer.data(), length, 0);
  EXPECT_THAT(y, Represents(value));
  EXPECT_EQ(parsing.allocations() - parsing.deallocations(),
            Handle(y).isImmediate() ? 0u : 1u);
  rcgmp_mpz_release(y, 0);
}

void RoundTripsThroughGmp(const Integer &value) { checkRoundTrip(value); }
// Up to 80 words reach the parser's heap buffer for more than 1024 digits.
FUZZ_TEST(Decimal, RoundTripsThroughGmp)
    .WithDomains(fuzztest::OneOf(AnyInteger(), AnyWords(80)));

// An optional minus sign followed by one or more digits.
bool wellFormed(std::string_view text) {
  if (text.starts_with('-'))
    text.remove_prefix(1);
  return !text.empty() && std::ranges::all_of(text, [](char digit) {
    return digit >= '0' && digit <= '9';
  });
}

// Text of any form parses exactly when it is well formed, to the value GMP
// reads from it.
void ParsesLikeGmp(const std::string &text) {
  AllocationCounter counter;
  void *value = rcgmp_mpz_from_decimal(text.data(), text.size(), 0);
  if (!wellFormed(text)) {
    EXPECT_EQ(value, nullptr) << text;
    EXPECT_TRUE(counter.balanced());
    return;
  }
  Integer expected;
  ASSERT_EQ(mpz_set_str(expected, text.c_str(), 10), 0) << text;
  EXPECT_THAT(value, Represents(expected)) << text;
  rcgmp_mpz_release(value, 0);
  EXPECT_TRUE(counter.balanced());
}
FUZZ_TEST(Decimal, ParsesLikeGmp)
    .WithDomains(fuzztest::StringOf(
        fuzztest::ElementOf({'-', '+', ' ', 'a', '0', '0', '0', '1', '2', '3',
                             '4', '5', '6', '7', '8', '9'})));

TEST(Decimal, RejectsMalformedText) {
  for (std::string_view text :
       {"", "-", "+1", " 1", "1 ", "1-", "--1", "0x1", "1_000", "-a", "12a3"})
    EXPECT_EQ(rcgmp_mpz_from_decimal(text.data(), text.size(), 0), nullptr)
        << text;
}

// Zero in its spellings, leading zeros, the immediate bounds, the word
// boundaries in digits and in bits, and a value beyond the parser's stack
// buffer.
TEST(Decimal, MatchesGmpOnBoundaries) {
  constexpr unsigned word = GMP_NUMB_BITS;
  Integer bound = Integer::powerOfTwo(word - 3);
  Integer tenToDigits(1);
  for (int digit = 0; digit < std::numeric_limits<std::uintptr_t>::digits10;
       ++digit)
    mpz_mul_ui(tenToDigits, tenToDigits, 10);
  Integer huge;
  mpz_ui_pow_ui(huge, 7, 4000);
  std::vector<Integer> values{Integer(0),
                              bound - 1,
                              bound,
                              -bound,
                              -bound - 1,
                              tenToDigits - 1,
                              tenToDigits,
                              -tenToDigits,
                              huge,
                              -huge,
                              Integer::powerOfTwo(word) - 1,
                              Integer::powerOfTwo(word)};
  for (const Integer &value : values)
    checkRoundTrip(value);
  std::string padded = std::string(40, '0') + "1";
  for (std::string_view text : {"-0", "000", "-000123", padded.c_str()}) {
    void *value = rcgmp_mpz_from_decimal(text.data(), text.size(), 0);
    Integer expected;
    mpz_set_str(expected, std::string(text).c_str(), 10);
    EXPECT_THAT(value, Represents(expected)) << text;
    rcgmp_mpz_release(value, 0);
  }
}

// Parsed integers start with one reference whatever their counting.
TEST(Decimal, ParsesAtomicIntegers) {
  std::string text = gmpDecimal(Integer::powerOfTwo(200));
  void *value = rcgmp_mpz_from_decimal(text.data(), text.size(), 1);
  EXPECT_THAT(value, Represents(Integer::powerOfTwo(200)));
  EXPECT_EQ(references(value, rcgmp::Counting::atomic), 1u);
  rcgmp_mpz_release(value, 1);
}

} // namespace
