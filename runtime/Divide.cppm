module;
#include "RCGMP/Runtime/Mpz.h"
#include <gmp.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <optional>
#include <span>
export module rcgmp.runtime:divide;
import :binary;
import :mpz;

namespace rcgmp {

export enum class Rounding { truncate, floor, euclidean };

// The results of a division. One that was not requested is zero.
export struct Division {
  Handle quotient = Handle::immediate(0);
  Handle remainder = Handle::immediate(0);
};

// The canonical integer for a word, which may lie outside the immediate range.
inline Handle fromInteger(std::intptr_t value) {
  bool negative = value < 0;
  std::uintptr_t magnitude =
      negative ? -static_cast<std::uintptr_t>(value) : value;
  if (std::optional<Handle> small = Handle::immediate(magnitude, negative))
    return *small;
  return fromWords(std::span(&magnitude, 1), negative);
}

// Divides lhs by rhs, computing the requested results. Truncating division
// rounds the quotient toward zero, and the remainder has the sign of lhs.
// Floor division rounds it toward negative infinity, and the remainder has the
// sign of rhs. Euclidean division rounds it so that the remainder is
// nonnegative. A zero divisor terminates the program.
export template <Counting counting, Rounding rounding, bool wantQuotient,
                 bool wantRemainder>
Division divide(Handle lhs, Handle rhs, Reuse reuse) {
  Division result;
  if (lhs.isImmediate() && rhs.isImmediate()) {
    std::intptr_t x = lhs.value(), y = rhs.value();
    if (!y)
      std::abort();
    // Immediates are narrower than the word, so only the quotient of the
    // immediate minimum by -1 leaves their range, and it fits the word.
    std::intptr_t quotient = x / y, remainder = x % y;
    if (rounding == Rounding::floor && remainder &&
        (remainder < 0) != (y < 0)) {
      --quotient;
      remainder += y;
    }
    if (rounding == Rounding::euclidean && remainder < 0) {
      quotient -= y < 0 ? -1 : 1;
      remainder += y < 0 ? -y : y;
    }
    if (wantQuotient)
      result.quotient = fromInteger(quotient);
    if (wantRemainder)
      result.remainder = Handle::immediate(remainder);
    return result;
  }
  Magnitude n(lhs), d(rhs);
  if (!d.count())
    std::abort();
  std::size_t nn = n.count(), dn = d.count();
  bool negative = n.negative != d.negative;
  // Floor division moves an inexact negative quotient down by one, and
  // Euclidean division moves the inexact quotient of a negative dividend away
  // from zero. Either takes the remainder's magnitude from that of the
  // divisor, the floored remainder with the divisor's sign and the Euclidean
  // one nonnegative.
  auto adjusts = [&](bool inexact) {
    switch (rounding) {
    case Rounding::truncate:
      return false;
    case Rounding::floor:
      return negative && inexact;
    case Rounding::euclidean:
      return n.negative && inexact;
    }
  };
  auto remainderNegative = [&](bool adjust) {
    return adjust ? rounding == Rounding::floor && d.negative : n.negative;
  };
  const Limb *np = n.limbs.data(), *dp = d.limbs.data();
  // Whether an operand's reference passed to a result, which then holds or
  // has freed its storage.
  bool keepLhs = false, keepRhs = false;
  auto take = [](bool &kept, Handle operand) -> Mpz & {
    kept = true;
    return operand.heap();
  };

  if (nn < dn) {
    // The quotient is zero, or one of its sign after adjustment.
    bool adjust = adjusts(nn != 0);
    result.quotient = Handle::immediate(adjust ? negative ? -1 : 1 : 0);
    if (wantRemainder && !adjust) {
      result.remainder = lhs;
      keepLhs = true;
    } else if (wantRemainder) {
      // |d| - |n|. Subtracting in the lhs storage replaces each limb of n
      // only after reading it.
      Mpz &out = reuse.rhs                                ? take(keepRhs, rhs)
                 : reuse.lhs && lhs.heap().capacity >= dn ? take(keepLhs, lhs)
                                                          : allocateResult(dn);
      Limb *rp = out.limbs().data();
      Limb borrow = mpn_sub_n(rp, dp, np, nn);
      if (dn > nn)
        mpn_sub_1(rp + nn, dp + nn, dn - nn, borrow);
      out.setMagnitude(dn, remainderNegative(true));
      result.remainder = normalize<counting>(out);
    }
  } else if (dn == 1) {
    Limb divisor = dp[0], rest;
    Mpz *quotient = nullptr;
    if constexpr (wantQuotient) {
      // Dividing by one limb permits writing the quotient over the dividend.
      quotient = reuse.lhs ? &take(keepLhs, lhs) : &allocateResult(nn);
      rest = mpn_divrem_1(quotient->limbs().data(), 0, np, nn, divisor);
    } else {
      rest = mpn_mod_1(np, nn, divisor);
    }
    bool adjust = adjusts(rest != 0);
    if (adjust)
      rest = divisor - rest;
    if constexpr (wantQuotient) {
      // A quotient of all ones has divisor one, so it is exact and does not
      // carry here.
      Limb *qp = quotient->limbs().data();
      if (adjust)
        mpn_add_1(qp, qp, nn, 1);
      quotient->setMagnitude(nn, negative);
      result.quotient = normalize<counting>(*quotient);
    }
    if constexpr (wantRemainder) {
      bool sign = remainderNegative(adjust);
      if (std::optional<Handle> small = Handle::immediate(rest, sign)) {
        result.remainder = *small;
      } else {
        Mpz &out = reuse.rhs                    ? take(keepRhs, rhs)
                   : reuse.lhs && !wantQuotient ? take(keepLhs, lhs)
                                                : allocateResult(1);
        out.limbs()[0] = rest;
        out.setMagnitude(1, sign);
        result.remainder = Handle(out);
      }
    }
  } else {
    std::size_t qn = nn - dn + 1;
    // mpn_tdiv_qr may write the remainder over the dividend, but no result
    // may overlap the divisor, and the quotient may overlap neither operand.
    // A result that is not requested goes to a reusable dividend, which its
    // release then frees, or to scratch storage, on the stack when short.
    Mpz *quotient = wantQuotient ? &allocateResult(qn + 1) : nullptr;
    Mpz *remainder = reuse.lhs       ? &lhs.heap()
                     : wantRemainder ? &allocateResult(dn)
                                     : nullptr;
    std::size_t scratchCount = !quotient ? qn + 1 : !remainder ? dn : 0;
    Limb stack[16];
    Mpz *spill =
        scratchCount > std::size(stack) ? &allocate(scratchCount) : nullptr;
    Limb *scratch = spill ? spill->limbs().data() : stack;
    Limb *qp = quotient ? quotient->limbs().data() : scratch;
    Limb *rp = remainder ? remainder->limbs().data() : scratch;
    mpn_tdiv_qr(qp, rp, 0, np, nn, dp, dn);
    bool adjust = adjusts(!mpn_zero_p(rp, dn));
    qp[qn] = adjust ? mpn_add_1(qp, qp, qn, 1) : 0;
    if (adjust)
      mpn_sub_n(rp, dp, rp, dn);
    if (spill)
      deallocate<counting>(*spill);
    if (wantQuotient) {
      quotient->setMagnitude(qn + 1, negative);
      result.quotient = normalize<counting>(*quotient);
    }
    if (wantRemainder) {
      keepLhs = reuse.lhs;
      remainder->setMagnitude(dn, remainderNegative(adjust));
      result.remainder = normalize<counting>(*remainder);
    }
  }
  // Release each operand whose storage holds no result, after all reads.
  if (!keepLhs)
    release<counting>(lhs);
  if (!keepRhs)
    release<counting>(rhs);
  return result;
}

// Divides as divide does, except that a zero divisor gives the quotient zero
// and the remainder lhs, as the integer division of Lean does.
template <Counting counting, Rounding rounding, bool wantQuotient,
          bool wantRemainder>
Division divideLean(Handle lhs, Handle rhs, Reuse reuse) {
  if (rhs != Handle::immediate(0))
    return divide<counting, rounding, wantQuotient, wantRemainder>(lhs, rhs,
                                                                   reuse);
  // The remainder takes over the reference to lhs. Zero is immediate, so the
  // divisor needs no release.
  Division result;
  if (wantRemainder)
    result.remainder = lhs;
  else
    release<counting>(lhs);
  return result;
}

// Adapts a division with one result to the runtime ABI.
template <auto divide, Handle Division::*result>
void *divisionEntryPoint(void *lhs, void *rhs, std::uint32_t reuse) {
  return (divide(Handle(lhs), Handle(rhs), Reuse::fromBits(reuse)).*result)
      .pointer();
}

// Adapts a division with both results to the runtime ABI.
template <auto divide>
rcgmp_mpz_qr divisionPairEntryPoint(void *lhs, void *rhs, std::uint32_t reuse) {
  Division result = divide(Handle(lhs), Handle(rhs), Reuse::fromBits(reuse));
  return {result.quotient.pointer(), result.remainder.pointer()};
}

} // namespace rcgmp

extern "C" {

#define RCGMP_DIVISION(name, divide, rounding, quotient, remainder, member)    \
  void *rcgmp_mpz_##name(void *lhs, void *rhs, std::uint32_t reuse) {          \
    return rcgmp::divisionEntryPoint<                                          \
        rcgmp::divide<rcgmp::Counting::nonatomic, rounding, quotient,          \
                      remainder>,                                              \
        &rcgmp::Division::member>(lhs, rhs, reuse);                            \
  }                                                                            \
  void *rcgmp_mpz_##name##_atomic(void *lhs, void *rhs, std::uint32_t reuse) { \
    return rcgmp::divisionEntryPoint<                                          \
        rcgmp::divide<rcgmp::Counting::atomic, rounding, quotient, remainder>, \
        &rcgmp::Division::member>(lhs, rhs, reuse);                            \
  }
RCGMP_DIVISION(tdiv_q, divide, rcgmp::Rounding::truncate, true, false, quotient)
RCGMP_DIVISION(tdiv_r, divide, rcgmp::Rounding::truncate, false, true,
               remainder)
RCGMP_DIVISION(fdiv_q, divide, rcgmp::Rounding::floor, true, false, quotient)
RCGMP_DIVISION(fdiv_r, divide, rcgmp::Rounding::floor, false, true, remainder)
RCGMP_DIVISION(ediv_q, divide, rcgmp::Rounding::euclidean, true, false,
               quotient)
RCGMP_DIVISION(ediv_r, divide, rcgmp::Rounding::euclidean, false, true,
               remainder)
RCGMP_DIVISION(tdiv_q_lean, divideLean, rcgmp::Rounding::truncate, true, false,
               quotient)
RCGMP_DIVISION(tdiv_r_lean, divideLean, rcgmp::Rounding::truncate, false, true,
               remainder)
RCGMP_DIVISION(fdiv_q_lean, divideLean, rcgmp::Rounding::floor, true, false,
               quotient)
RCGMP_DIVISION(fdiv_r_lean, divideLean, rcgmp::Rounding::floor, false, true,
               remainder)
RCGMP_DIVISION(ediv_q_lean, divideLean, rcgmp::Rounding::euclidean, true, false,
               quotient)
RCGMP_DIVISION(ediv_r_lean, divideLean, rcgmp::Rounding::euclidean, false, true,
               remainder)
#undef RCGMP_DIVISION

#define RCGMP_DIVISION_PAIR(name, divide, rounding)                            \
  rcgmp_mpz_qr rcgmp_mpz_##name(void *lhs, void *rhs, std::uint32_t reuse) {   \
    return rcgmp::divisionPairEntryPoint<                                      \
        rcgmp::divide<rcgmp::Counting::nonatomic, rounding, true, true>>(      \
        lhs, rhs, reuse);                                                      \
  }                                                                            \
  rcgmp_mpz_qr rcgmp_mpz_##name##_atomic(void *lhs, void *rhs,                 \
                                         std::uint32_t reuse) {                \
    return rcgmp::divisionPairEntryPoint<                                      \
        rcgmp::divide<rcgmp::Counting::atomic, rounding, true, true>>(         \
        lhs, rhs, reuse);                                                      \
  }
RCGMP_DIVISION_PAIR(tdiv_qr, divide, rcgmp::Rounding::truncate)
RCGMP_DIVISION_PAIR(fdiv_qr, divide, rcgmp::Rounding::floor)
RCGMP_DIVISION_PAIR(ediv_qr, divide, rcgmp::Rounding::euclidean)
RCGMP_DIVISION_PAIR(tdiv_qr_lean, divideLean, rcgmp::Rounding::truncate)
RCGMP_DIVISION_PAIR(fdiv_qr_lean, divideLean, rcgmp::Rounding::floor)
RCGMP_DIVISION_PAIR(ediv_qr_lean, divideLean, rcgmp::Rounding::euclidean)
#undef RCGMP_DIVISION_PAIR
}
