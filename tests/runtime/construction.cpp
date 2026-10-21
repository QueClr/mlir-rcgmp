// Construction and reference counting in compiled code, interoperating with
// the runtime entry points.
#include "RCGMP/Runtime/Mpz.h"
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
import rcgmp.runtime;
import rcgmp.testing;
#ifdef RCGMP_TEST_ALLOCATOR
import rcgmp.testing.allocator;
#endif

extern "C" {
void *from_signed(std::int64_t);
void *from_unsigned(std::uint64_t);
void *from_atomic(std::int64_t);
void *retain_local(void *);
void *retain_atomic(void *);
void release_local(void *);
void release_atomic(void *);
void *from_min129();
void *from_max129();
void *from_signed_i1();
void *from_unsigned_i1();
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

void ConstructsAndCountsReferences(std::int64_t n, Counting counting) {
  bool atomic = counting == Counting::atomic;
  Integer expected(n);
  void *x = atomic ? from_atomic(n) : from_signed(n);
  EXPECT_EQ(Handle(x).isImmediate(),
            n >= Handle::immediateMin && n <= Handle::immediateMax);
  EXPECT_THAT(x, Represents(expected));
  void *alias = atomic ? retain_atomic(x) : retain_local(x);
  EXPECT_EQ(alias, x);
  // Interoperate between compiled reference counting and the runtime.
  rcgmp_mpz_release(x, atomic);
  EXPECT_THAT(alias, Represents(expected));
  rcgmp_mpz_retain(alias, atomic);
  void (*release)(void *) = atomic ? release_atomic : release_local;
  release(alias);
  release(alias);
}
FUZZ_TEST(Construction, ConstructsAndCountsReferences)
    .WithDomains(fuzztest::Arbitrary<std::int64_t>().WithSeeds(
                     {0, 1, -1, Handle::immediateMax, Handle::immediateMin,
                      Handle::immediateMax + 1, Handle::immediateMin - 1,
                      std::numeric_limits<std::int64_t>::max(),
                      std::numeric_limits<std::int64_t>::min()}),
                 AnyCounting());

TEST(Construction, FromUnsignedMaximum) {
  void *x = from_unsigned(std::numeric_limits<std::uint64_t>::max());
  EXPECT_THAT(x, Represents(Integer::powerOfTwo(64) - 1));
  release_local(x);
}

TEST(Construction, FromWideIntegers) {
  void *x = from_min129();
  EXPECT_THAT(x, Represents(-Integer::powerOfTwo(128)));
  release_local(x);
  x = from_max129();
  EXPECT_THAT(x, Represents(Integer::powerOfTwo(129) - 1));
  release_local(x);
}

TEST(Construction, FromOneBit) {
  EXPECT_THAT(from_signed_i1(), Represents(Integer(-1)));
  EXPECT_THAT(from_unsigned_i1(), Represents(Integer(1)));
}

// Zero is the immediate 0 regardless of the sign flag or zero words.
TEST(Construction, ZeroIgnoresSignAndZeroWords) {
  std::uintptr_t zeros[] = {0, 0};
  EXPECT_EQ(rcgmp_mpz_from_words(zeros, 0, 1, 0),
            Handle::immediate(0).pointer());
  EXPECT_EQ(rcgmp_mpz_from_words(zeros, 2, 1, 1),
            Handle::immediate(0).pointer());
}

#ifdef RCGMP_TEST_ALLOCATOR
TEST(Construction, BalancesAllocations) {
  AllocationCounter counter;
  ConstructsAndCountsReferences(std::numeric_limits<std::int64_t>::min(),
                                Counting::nonatomic);
  ConstructsAndCountsReferences(std::numeric_limits<std::int64_t>::max(),
                                Counting::atomic);
  EXPECT_GT(counter.allocations(), 0u);
  EXPECT_TRUE(counter.balanced());
}

TEST(Construction, CountsImmediatesWithoutAllocation) {
  AllocationCounter counter;
  void *x = from_signed(123);
  release_local(retain_local(x));
  release_local(x);
  EXPECT_EQ(counter.allocations(), 0u);
}
#endif

} // namespace
