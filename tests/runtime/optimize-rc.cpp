// Reference counting optimized by -rcgmp-optimize-rc: an operation that
// receives the last reference reuses its storage, and one that does not leaves
// the value intact.
#include "RCGMP/Runtime/Mpz.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.allocator;

extern "C" {
void *increment(void *);
void *increment_if(void *, std::int32_t);
std::int32_t increment_compare(void *);
}

namespace {

using namespace rcgmp::test;

class OptimizeRCTest : public ::testing::Test {
protected:
  void TearDown() override { EXPECT_TRUE(allocations.balanced()); }

  Integer value = Integer::powerOfTwo(130);
  AllocationCounter allocations;
};

TEST_F(OptimizeRCTest, LastUseReusesStorage) {
  for (bool branch : {false, true}) {
    void *x = toRuntime(value);
    AllocationCounter counter;
    void *sum = branch ? increment_if(x, 1) : increment(x);
    EXPECT_EQ(sum, x);
    EXPECT_EQ(counter.allocations(), 0u);
    EXPECT_THAT(sum, Represents(value + 1));
    rcgmp_mpz_release(sum, false);
  }
}

TEST_F(OptimizeRCTest, SharedValueIsNotReused) {
  void *x = toRuntime(value);
  rcgmp_mpz_retain(x, false);
  void *sum = increment(x);
  EXPECT_NE(sum, x);
  EXPECT_THAT(x, Represents(value));
  EXPECT_THAT(sum, Represents(value + 1));
  rcgmp_mpz_release(sum, false);
  rcgmp_mpz_release(x, false);
}

TEST_F(OptimizeRCTest, UntakenBranchKeepsValue) {
  void *x = toRuntime(value);
  void *same = increment_if(x, 0);
  EXPECT_EQ(same, x);
  EXPECT_THAT(same, Represents(value));
  rcgmp_mpz_release(same, false);
}

TEST_F(OptimizeRCTest, ReadAfterConsumerSeesValue) {
  EXPECT_EQ(increment_compare(toRuntime(value)), 1);
}

} // namespace
