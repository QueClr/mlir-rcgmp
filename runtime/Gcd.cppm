module;
#include "RCGMP/Runtime/Mpz.h"
#include <gmp.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <span>
export module rcgmp.runtime:gcd;
import :binary;
import :mpz;
import :unary;

namespace rcgmp {

// The number of trailing zero bits of a nonzero magnitude.
inline std::size_t trailingZeros(std::span<const Limb> limbs) {
  std::size_t index = 0;
  while (!limbs[index])
    ++index;
  return index * GMP_NUMB_BITS + std::countr_zero(limbs[index]);
}

// Writes a nonzero magnitude shifted right by its zeros trailing zero bits to
// out, and returns its limb count.
inline std::size_t oddPart(Limb *out, std::span<const Limb> limbs,
                           std::size_t zeros) {
  std::span<const Limb> rest = limbs.subspan(zeros / GMP_NUMB_BITS);
  unsigned bits = zeros % GMP_NUMB_BITS;
  if (!bits) {
    std::ranges::copy(rest, out);
    return rest.size();
  }
  mpn_rshift(out, rest.data(), rest.size(), bits);
  return rest.size() - !out[rest.size() - 1];
}

// Consumes both operands and returns a nonnegative result of one limb, in the
// storage of a reusable operand when it is not immediate.
template <Counting counting>
Handle limbResult(Limb value, Handle lhs, Handle rhs, Reuse reuse) {
  if (std::optional<Handle> small = Handle::immediate(value, false))
    return consume<counting>(lhs, rhs, *small);
  Mpz &result = destination(lhs, rhs, reuse, 1);
  result.limbs()[0] = value;
  result.setMagnitude(1, false);
  return finish<counting>(result, lhs, rhs);
}

// The greatest common divisor of the absolute values of lhs and rhs. It is no
// larger than either nonzero magnitude, so a reusable operand always has room
// for it.
export template <Counting counting>
Handle gcd(Handle lhs, Handle rhs, Reuse reuse) {
  Magnitude left(lhs), right(rhs);
  // A zero operand is an immediate, which needs no release.
  if (!left.count())
    return absolute<counting>(rhs, reuse.rhs);
  if (!right.count())
    return absolute<counting>(lhs, reuse.lhs);
  bool swapped = left.count() < right.count();
  Magnitude &x = swapped ? right : left;
  Magnitude &y = swapped ? left : right;
  std::size_t xn = x.count(), yn = y.count();
  if (yn == 1)
    return limbResult<counting>(mpn_gcd_1(x.limbs.data(), xn, y.limbs[0]), lhs,
                                rhs, reuse);

  // mpn_gcd destroys both operands and needs one of them odd. As in mpz_gcd,
  // the powers of two that both share are removed first and restored in the
  // result, here by making the shorter operand odd: the odd part v shares
  // every odd divisor of y and no factor of two.
  std::size_t xZeros = trailingZeros(x.limbs), yZeros = trailingZeros(y.limbs);
  std::size_t shift = std::min(xZeros, yZeros);
  // Scratch for v, the greatest common divisor, and either a copy of x or a
  // remainder and quotient of x by v, on the stack when short.
  std::size_t scratchCount = 2 * yn + xn + 1;
  Limb stack[64];
  Mpz *spill =
      scratchCount > std::size(stack) ? &allocate(scratchCount) : nullptr;
  Limb *vp = spill ? spill->limbs().data() : stack;
  Limb *gp = vp + yn, *wp = gp + yn;
  std::size_t vn = oddPart(vp, y.limbs, yZeros);
  const Limb *g = gp;
  std::size_t gn = 1;
  if (vn == 1) {
    gp[0] = mpn_gcd_1(x.limbs.data(), xn, vp[0]);
  } else if (xn > vn) {
    // Dividing x by v reads x without copying it, and leaves a remainder no
    // longer than v for mpn_gcd.
    Limb *rp = wp, *qp = wp + vn;
    mpn_tdiv_qr(qp, rp, 0, x.limbs.data(), xn, vp, vn);
    std::size_t rn = trimmed(std::span<const Limb>(rp, vn)).size();
    if (!rn) {
      g = vp;
      gn = vn;
    } else if (rn == 1) {
      gp[0] = mpn_gcd_1(vp, vn, rp[0]);
    } else {
      gn = mpn_gcd(gp, vp, vn, rp, rn);
    }
  } else {
    // Of operands of one length, mpz_gcd passes the larger first, which takes
    // fewer steps.
    std::ranges::copy(x.limbs, wp);
    gn = wp[xn - 1] < vp[vn - 1] ? mpn_gcd(gp, vp, vn, wp, xn)
                                 : mpn_gcd(gp, wp, xn, vp, vn);
  }

  std::size_t zeroLimbs = shift / GMP_NUMB_BITS;
  unsigned bits = shift % GMP_NUMB_BITS;
  std::size_t count =
      gn + zeroLimbs + (bits && g[gn - 1] >> (GMP_NUMB_BITS - bits));
  Handle result = Handle::immediate(0);
  if (count == 1) {
    result = limbResult<counting>(g[0] << bits, lhs, rhs, reuse);
  } else {
    // All operand limbs have been read.
    Mpz &out = destination(lhs, rhs, reuse, count);
    Limb *rp = out.limbs().data();
    std::fill_n(rp, zeroLimbs, Limb{0});
    Limb carry = 0;
    if (bits)
      carry = mpn_lshift(rp + zeroLimbs, g, gn, bits);
    else
      std::copy_n(g, gn, rp + zeroLimbs);
    if (carry)
      rp[count - 1] = carry;
    out.setMagnitude(count, false);
    result = finish<counting>(out, lhs, rhs);
  }
  if (spill)
    deallocate<counting>(*spill);
  return result;
}

} // namespace rcgmp

extern "C" {

void *rcgmp_mpz_gcd(void *lhs, void *rhs, std::uint32_t reuse) {
  return rcgmp::binaryEntryPoint<rcgmp::gcd<rcgmp::Counting::nonatomic>>(
      lhs, rhs, reuse);
}

void *rcgmp_mpz_gcd_atomic(void *lhs, void *rhs, std::uint32_t reuse) {
  return rcgmp::binaryEntryPoint<rcgmp::gcd<rcgmp::Counting::atomic>>(lhs, rhs,
                                                                      reuse);
}
}
