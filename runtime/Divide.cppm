module;
#include "RCGMP/Runtime/Mpz.h"
#include <gmp.h>

#include <algorithm>
#include <bit>
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

// Whether mpz_tdiv_q, which computes a quotient of qn limbs by a divisor of dn
// limbs without most of the remainder, is faster than mpn_tdiv_qr. It costs a
// few hundred instructions more to call, for the quotient it allocates and
// rcgmp copies, so it pays off only for divisors of 16 limbs or more. It also
// needs a quotient of 3 limbs or more, and saves little on one more than 4
// times longer than the divisor, whose division the remainder hardly slows.
constexpr bool skipsRemainder(std::size_t qn, std::size_t dn) {
  return dn >= 16 && qn >= 3 && qn <= 4 * dn;
}

// The truncated quotient of magnitudes, with the given sign, from mpz_tdiv_q
// on read-only views of them. GMP allocates the quotient, which is copied
// out, as mpz_tdiv_q may reallocate its destination with GMP's allocation
// functions, which must not see storage from rcgmp's allocation hooks. Kept
// out of line, its GMP integers take no stack in divide's other paths.
[[gnu::noinline]] Handle quotientOnly(std::span<const Limb> n,
                                      std::span<const Limb> d, bool negative) {
  mpz_t dividend, divisor, quotient;
  mpz_init(quotient);
  mpz_tdiv_q(quotient, mpz_roinit_n(dividend, n.data(), n.size()),
             mpz_roinit_n(divisor, d.data(), d.size()));
  Handle result = fromWords(
      std::span(mpz_limbs_read(quotient), mpz_size(quotient)), negative);
  mpz_clear(quotient);
  return result;
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

  if (nn < dn || (nn == dn && mpn_cmp(np, dp, nn) < 0)) {
    // |n| < |d|, so the quotient is zero, or one of its sign after
    // adjustment.
    bool adjust = adjusts(nn != 0);
    result.quotient = Handle::immediate(adjust ? negative ? -1 : 1 : 0);
    if (wantRemainder && !adjust) {
      result.remainder = lhs;
      keepLhs = true;
    } else if (wantRemainder) {
      // |d| - |n|. Subtracting in the lhs storage replaces each limb of n
      // only after reading it. A unique dividend with room holds the
      // remainder, as it does for longer dividends.
      Mpz &out = reuse.lhs && lhs.heap().capacity >= dn ? take(keepLhs, lhs)
                 : reuse.rhs                            ? take(keepRhs, rhs)
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
  } else if (!wantRemainder && !adjusts(true) &&
             skipsRemainder(nn - dn + 1, dn)) {
    // A quotient that no rounding moves from the truncated one needs no
    // remainder, which mpz_tdiv_q mostly skips computing.
    result.quotient = quotientOnly(n.limbs, d.limbs, negative);
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

// The inverse of an odd limb modulo 2^GMP_NUMB_BITS. (3 odd) xor 2 is correct
// in its five low bits, and each Newton step doubles the correct bits.
constexpr Limb inverse(Limb odd) {
  // The step x (2 - odd x) is written x (1 + e) with the error e = 1 - odd x,
  // which the step squares, so that the products of a step are independent.
  Limb inverse = (3 * odd) ^ 2, error = 1 - odd * inverse;
  for (unsigned bits = 5; bits < GMP_NUMB_BITS; bits *= 2) {
    inverse *= 1 + error;
    error *= error;
  }
  return inverse;
}

// Replaces {qp, qn} by its quotient by the odd {dp, dn} modulo 2^(qn W), from
// the low end: each step chooses the quotient limb that clears the lowest
// remaining limb, and subtracts that multiple of the divisor from the limbs
// above, which matter up to the qn-th limb only. When {qp, qn} is a multiple
// of {dp, dn} with a quotient of qn limbs, this is that quotient.
void divideFromLowEnd(Limb *qp, std::size_t qn, const Limb *dp,
                      std::size_t dn) {
  Limb dinv = inverse(dp[0]);
  for (std::size_t i = 0; i + 1 < qn; ++i) {
    Limb q = qp[i] * dinv;
    std::size_t count = std::min(dn, qn - i);
    Limb borrow = mpn_submul_1(qp + i, dp, count, q);
    if (i + count < qn)
      mpn_sub_1(qp + i + count, qp + i + count, qn - i - count, borrow);
    qp[i] = q;
  }
  // The last quotient limb clears no limb above.
  qp[qn - 1] *= dinv;
}

// Division from the low end of more limb products than this, of operands of
// about 180 limbs, is slower than mpn_tdiv_qr.
constexpr std::size_t maxLowEndProducts = 1 << 15;

// The quotient of lhs by rhs, of which lhs must be a multiple; otherwise the
// result is some integer no longer than lhs. A zero divisor terminates the
// program. As in mpz_divexact, the zero limbs ending the divisor, which also
// end the dividend, are dropped, and a divisor of several limbs is made odd by
// shifting both operands right by its trailing zero bits. Only the lowest
// limbs of the operands then determine the quotient, which is computed from
// the low end, in the storage of a unique dividend if there is one, unless the
// operands are long enough for mpn_tdiv_qr to be faster.
export template <Counting counting>
Handle divideExact(Handle lhs, Handle rhs, Reuse reuse) {
  if (lhs.isImmediate() && rhs.isImmediate()) {
    std::intptr_t divisor = rhs.value();
    if (!divisor)
      std::abort();
    // Only the quotient of the immediate minimum by -1 leaves the immediate
    // range.
    return fromInteger(lhs.value() / divisor);
  }
  Magnitude n(lhs), d(rhs);
  if (!d.count())
    std::abort();
  std::size_t nn = n.count(), dn = d.count();
  // Only zero is a multiple shorter than the divisor.
  if (nn < dn)
    return consume<counting>(lhs, rhs, Handle::immediate(0));
  bool negative = n.negative != d.negative;
  const Limb *np = n.limbs.data(), *dp = d.limbs.data();
  std::size_t zeros = 0;
  while (!dp[zeros])
    ++zeros;
  np += zeros;
  nn -= zeros;
  dp += zeros;
  dn -= zeros;
  // The quotient is less than 2^(qn W). Dividing from the low end takes about
  // qn min(qn, dn) limb products, beyond which the subquadratic division of
  // mpn_tdiv_qr is faster, although it also computes a remainder. Its quotient
  // may not overlap the dividend.
  std::size_t qn = nn - dn + 1, oddCount = std::min(dn, qn);
  bool fromLowEnd = dn == 1 || qn * oddCount <= maxLowEndProducts;
  bool inPlace = reuse.lhs && fromLowEnd;
  // A quotient of a single limb, which is often immediate, needs no storage
  // until it is known.
  Limb single;
  Mpz *quotient = inPlace  ? &lhs.heap()
                  : qn > 1 ? &allocateResult(qn)
                           : nullptr;
  Limb *qp = quotient ? quotient->limbs().data() : &single;
  if (!fromLowEnd) {
    Mpz &remainder = allocate(dn);
    mpn_tdiv_qr(qp, remainder.limbs().data(), 0, np, nn, dp, dn);
    deallocate<counting>(remainder);
  } else if (nn == 1) {
    qp[0] = np[0] / dp[0];
  } else if (dn == 1) {
    // mpn_divexact_1 accepts an even divisor, but not a quotient that
    // partially overlaps the dividend, as in a unique dividend with zero limbs
    // dropped.
    if (qp != np && inPlace) {
      std::copy_n(np, nn, qp);
      np = qp;
    }
    mpn_divexact_1(qp, np, nn, dp[0]);
  } else {
    // The quotient's limbs come from the low qn limbs of the dividend shifted
    // right, whose highest takes bits from the limb above. A unique dividend
    // moves toward its low end, which mpn_rshift permits, and the limb above
    // lies beyond the limbs it writes.
    unsigned shift = std::countr_zero(dp[0]);
    if (shift) {
      mpn_rshift(qp, np, qn, shift);
      qp[qn - 1] |= np[qn] << (GMP_NUMB_BITS - shift);
    } else if (qp != np) {
      std::copy_n(np, qn, qp);
    }
    // Likewise only the low qn limbs of the divisor matter.
    const Limb *odd = dp;
    Limb stack[16];
    Mpz *spill = nullptr;
    if (shift) {
      spill = oddCount > std::size(stack) ? &allocate(oddCount) : nullptr;
      Limb *op = spill ? spill->limbs().data() : stack;
      mpn_rshift(op, dp, oddCount, shift);
      if (oddCount < dn)
        op[oddCount - 1] |= dp[oddCount] << (GMP_NUMB_BITS - shift);
      odd = op;
    }
    divideFromLowEnd(qp, qn, odd, oddCount);
    if (spill)
      deallocate<counting>(*spill);
  }
  Handle result = Handle::immediate(0);
  if (quotient) {
    quotient->setMagnitude(qn, negative);
    result = normalize<counting>(*quotient);
  } else {
    result = fromWords(std::span(&single, 1), negative);
  }
  // Release each operand whose storage holds no result, after all reads.
  if (!inPlace)
    release<counting>(lhs);
  release<counting>(rhs);
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

void *rcgmp_mpz_tdiv_q_exact(void *lhs, void *rhs, std::uint32_t reuse) {
  return rcgmp::binaryEntryPoint<
      rcgmp::divideExact<rcgmp::Counting::nonatomic>>(lhs, rhs, reuse);
}

void *rcgmp_mpz_tdiv_q_exact_atomic(void *lhs, void *rhs, std::uint32_t reuse) {
  return rcgmp::binaryEntryPoint<rcgmp::divideExact<rcgmp::Counting::atomic>>(
      lhs, rhs, reuse);
}
}
