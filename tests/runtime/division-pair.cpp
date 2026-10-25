// A quotient and a remainder of the same operands, computed by one division
// after canonicalization, compared with GMP over the operands' ownership.
#include "RCGMP/Runtime/Mpz.h"

#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <tuple>
#include <vector>

import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.allocator;

extern "C" {
rcgmp_mpz_qr truncated(void *, void *);
rcgmp_mpz_qr floored_lean_atomic(void *, void *);
rcgmp_mpz_qr euclidean_kept(void *, void *);
rcgmp_mpz_qr natural_by_ten(void *);
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

constexpr unsigned word = GMP_NUMB_BITS;

// The expected results of an entry point, given its operands.
using Expected = void (*)(mpz_ptr, mpz_ptr, mpz_srcptr, mpz_srcptr);

void euclideanKept(mpz_ptr quotient, mpz_ptr sum, mpz_srcptr lhs,
                   mpz_srcptr rhs) {
  gmpEuclidean(quotient, sum, lhs, rhs);
  mpz_add(sum, sum, lhs);
}

// A lean division by zero gives the quotient zero and the remainder lhs.
void flooredLean(mpz_ptr quotient, mpz_ptr remainder, mpz_srcptr lhs,
                 mpz_srcptr rhs) {
  if (mpz_sgn(rhs)) {
    mpz_fdiv_qr(quotient, remainder, lhs, rhs);
  } else {
    mpz_set_ui(quotient, 0);
    mpz_set(remainder, lhs);
  }
}

// Calls apply with lhs and rhs, either shared if the caller keeps an extra
// reference, and compares both results with expected. Shared operands keep
// their values and references, and nothing leaks or is freed twice.
void checkPair(rcgmp_mpz_qr (*apply)(void *, void *), Expected expected,
               const Integer &lhs, const Integer &rhs, Counting counting,
               bool lhsShared, bool rhsShared) {
  SCOPED_TRACE(lhs.hex() + " / " + rhs.hex() +
               (lhsShared ? " (lhs shared)" : "") +
               (rhsShared ? " (rhs shared)" : ""));
  bool atomic = counting == Counting::atomic;
  AllocationCounter balance;
  void *x = toRuntime(lhs), *y = toRuntime(rhs);
  if (lhsShared)
    rcgmp_mpz_retain(x, atomic);
  if (rhsShared)
    rcgmp_mpz_retain(y, atomic);
  Integer first, second;
  expected(first, second, lhs, rhs);
  auto [q, r] = apply(x, y);
  EXPECT_THAT(q, Represents(first));
  EXPECT_THAT(r, Represents(second));
  for (auto [operand, value, shared] :
       {std::tuple{x, &lhs, lhsShared}, {y, &rhs, rhsShared}}) {
    if (!shared)
      continue;
    EXPECT_THAT(operand, Represents(*value));
    // A result may be the operand itself, as a remainder of a smaller
    // dividend is.
    if (!Handle(operand).isImmediate())
      EXPECT_EQ(references(operand, counting),
                1u + (q == operand) + (r == operand));
    rcgmp_mpz_release(operand, atomic);
  }
  rcgmp_mpz_release(q, atomic);
  rcgmp_mpz_release(r, atomic);
  EXPECT_TRUE(balance.balanced());
}

// Immediates, one to three limbs, and 40 limbs, of either sign.
std::vector<Integer> operands() {
  std::vector<Integer> values{Integer(1),
                              Integer(-1),
                              Integer(7),
                              Integer(-7),
                              Integer(Handle::immediateMin),
                              Integer(Handle::immediateMax)};
  for (unsigned exponent : {word - 1, 2 * word, 3 * word - 5, 40 * word - 1})
    for (long delta : {-1L, 3L}) {
      Integer value = Integer::powerOfTwo(exponent) + delta;
      values.push_back(value);
      values.push_back(-value);
    }
  return values;
}

TEST(DivisionPair, MatchesGmp) {
  std::vector<Integer> values = operands();
  for (const auto &[lhs, rhs, lhsShared, rhsShared] :
       std::views::cartesian_product(values, values, std::array{false, true},
                                     std::array{false, true})) {
    checkPair(truncated, mpz_tdiv_qr, lhs, rhs, Counting::nonatomic, lhsShared,
              rhsShared);
    checkPair(floored_lean_atomic, flooredLean, lhs, rhs, Counting::atomic,
              lhsShared, rhsShared);
    checkPair(euclidean_kept, euclideanKept, lhs, rhs, Counting::nonatomic,
              lhsShared, rhsShared);
  }
  // A lean division by zero.
  for (const auto &[lhs, shared] :
       std::views::cartesian_product(values, std::array{false, true}))
    checkPair(floored_lean_atomic, flooredLean, lhs, Integer(0),
              Counting::atomic, shared, false);
}

TEST(DivisionPair, DividesNaturalsByTen) {
  std::vector<Integer> values = operands();
  std::erase_if(values, [](const Integer &value) { return value.sign() < 0; });
  values.push_back(Integer(0));
  auto byTen = [](void *x, void *) { return natural_by_ten(x); };
  for (const auto &[lhs, shared] :
       std::views::cartesian_product(values, std::array{false, true}))
    checkPair(byTen, mpz_tdiv_qr, lhs, Integer(10), Counting::nonatomic, shared,
              false);
}

} // namespace
