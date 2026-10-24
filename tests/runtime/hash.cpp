// Compiled hashes, which borrow their operands, compared with the runtime and
// with a reference built on GMP.
#include "RCGMP/Runtime/Hash.h"
#include "RCGMP/Runtime/Mpz.h"
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <ranges>
#include <utility>
#include <vector>
import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.allocator;

extern "C" {
std::int64_t hashi_local(void *);
std::int64_t hashi_atomic(void *);
std::int64_t hashi_constant();
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

constexpr std::array hashi{hashi_local, hashi_atomic};

// The hash from the value: itself in the range of int64_t, and otherwise the
// sign and the 64-bit chunks of the magnitude, mixed.
std::int64_t expectedHash(const Integer &value) {
  if (mpz_fits_slong_p(value))
    return mpz_get_si(value);
  Integer magnitude;
  mpz_abs(magnitude, value);
  std::size_t chunks = (mpz_sizeinbase(magnitude, 2) + 63) / 64;
  std::vector<std::uint64_t> words(chunks);
  mpz_export(words.data(), nullptr, -1, sizeof(std::uint64_t), 0, 0, magnitude);
  std::uint64_t state = rcgmp::hashing::start(value.negative());
  for (std::uint64_t word : words)
    state = rcgmp::hashing::absorb(state, word);
  return static_cast<std::int64_t>(rcgmp::hashing::finish(state, chunks));
}

// Hashing allocates and frees nothing, and leaves the operand and its count
// unchanged.
void checkHash(const Integer &value, Counting counting) {
  bool atomic = counting == Counting::atomic;
  void *x = toRuntime(value);
  AllocationCounter counter;
  std::int64_t expected = expectedHash(value);
  EXPECT_EQ(hashi[std::to_underlying(counting)](x), expected) << value;
  EXPECT_EQ((atomic ? rcgmp_mpz_hash_atomic : rcgmp_mpz_hash)(x), expected)
      << value;
  EXPECT_EQ(counter.allocations(), 0u);
  EXPECT_EQ(counter.deallocations(), 0u);
  EXPECT_THAT(x, Represents(value));
  if (!Handle(x).isImmediate())
    EXPECT_EQ(references(x, counting), 1u);
  rcgmp_mpz_release(x, atomic);
}

void MatchesReference(const Integer &value, Counting counting) {
  checkHash(value, counting);
}
FUZZ_TEST(Hash, MatchesReference).WithDomains(AnyInteger(), AnyCounting());

// The immediate bounds, the int64_t bounds, which heap integers in its range
// reach, and multiples of 2^64 that differ only in their high chunk.
TEST(Hash, MatchesReferenceOnBoundaries) {
  std::vector<Integer> values{Integer(0), Integer(1), Integer(-1),
                              Integer(Handle::immediateMin),
                              Integer(Handle::immediateMax)};
  for (unsigned exponent : {62u, 63u, 64u, 127u, 128u, 192u})
    for (long delta : {-1L, 0L, 1L}) {
      Integer value = Integer::powerOfTwo(exponent) + delta;
      values.push_back(value);
      values.push_back(-value);
    }
  for (const auto &[value, counting] :
       std::views::cartesian_product(values, countings))
    checkHash(value, counting);
  EXPECT_EQ(expectedHash(Integer(std::numeric_limits<long>::min())),
            std::numeric_limits<std::int64_t>::min());
}

// Values differing only in their sign or in chunks above the lowest hash
// differently, and an immortal constant hashes as its heap value.
TEST(Hash, SeparatesSignsAndHighChunks) {
  Integer high = Integer::powerOfTwo(128) + 1;
  std::vector<std::int64_t> hashes;
  for (const Integer &value :
       {high, -high, Integer::powerOfTwo(192) + 1, Integer::powerOfTwo(64) + 1})
    hashes.push_back(expectedHash(value));
  EXPECT_NE(hashes[0], hashes[1]);
  EXPECT_NE(hashes[0], hashes[2]);
  EXPECT_NE(hashes[0], hashes[3]);
  Integer constant;
  mpz_set_str(constant, "-340282366920938463463374607431768211457", 10);
  EXPECT_EQ(hashi_constant(), expectedHash(constant));
}

} // namespace
