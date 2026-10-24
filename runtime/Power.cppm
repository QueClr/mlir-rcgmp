module;
#include "RCGMP/Runtime/Mpz.h"
#include <gmp.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <optional>
#include <span>
export module rcgmp.runtime:power;
import :binary;
import :mpz;
import :shift;

namespace rcgmp {

// m^exponent as (m^k)^q * m^r, for exponent = q * k + r and the k that
// bit_width guarantees to keep m^k within a limb.
struct LimbPowers {
  Limb power, rest;
  std::uintptr_t quotient;
};

// The k for each bit width of m, which spares a division.
constexpr auto limbExponents = [] {
  std::array<unsigned char, GMP_NUMB_BITS + 1> exponents{};
  for (unsigned bits = 1; bits <= GMP_NUMB_BITS; ++bits)
    exponents[bits] = GMP_NUMB_BITS / bits;
  return exponents;
}();

// The bits of k and r select from the same squares of m, and m^k is computed
// only when q is not zero.
constexpr LimbPowers limbPowers(Limb m, std::uintptr_t exponent) {
  unsigned k = limbExponents[std::bit_width(m)];
  // Small quotients, the common ones, need no division.
  std::uintptr_t quotient = exponent < k       ? 0
                            : exponent < 2 * k ? 1
                                               : exponent / k;
  LimbPowers powers{1, 1, quotient};
  unsigned bits = quotient ? k : 0, r = exponent - quotient * k;
  for (Limb square = m;; square *= square) {
    if (bits & 1)
      powers.power *= square;
    if (r & 1)
      powers.rest *= square;
    bits >>= 1;
    r >>= 1;
    if (!bits && !r)
      return powers;
  }
}

// Squares a magnitude that overflows a limb when squared.
template <Counting counting>
Handle square(Handle base, std::span<const Limb> magnitude) {
  std::size_t count = 2 * magnitude.size();
  Mpz &result = allocateResult(count);
  Limb *out = result.limbs().data();
  mpn_sqr(out, magnitude.data(), magnitude.size());
  count -= !out[count - 1];
  result.setMagnitude(count, false);
  release<counting>(base);
  return Handle(result);
}

// Raises base to the power exponent, at least two, when the result overflows
// a limb. The trailing zero bits of the base are factored out as a final left
// shift of the power of its remaining factor, which is computed by squaring
// and multiplying, left to right.
template <Counting counting>
[[gnu::noinline]] Handle largePower(Handle base, std::uintptr_t exponent,
                                    bool reuse) {
  Magnitude magnitude(base);
  // A square needs none of the steps below, which would make it slower than
  // the same product.
  if (exponent == 2)
    return square<counting>(base, magnitude.limbs);
  bool negative = magnitude.negative && exponent & 1;

  // Zero limbs are skipped in place, and zero bits are shifted out of a
  // remaining factor of one limb, or two that shift into one, locally.
  std::span<const Limb> limbs = magnitude.limbs;
  std::size_t skipped =
      std::ranges::find_if(limbs, std::identity{}) - limbs.begin();
  std::span<const Limb> odd = limbs.subspan(skipped);
  std::uintptr_t bits =
      (limbs.size() - 1) * GMP_NUMB_BITS + std::bit_width(limbs.back());
  std::uintptr_t zeros = skipped * GMP_NUMB_BITS;
  unsigned shift = std::countr_zero(odd[0]);
  std::size_t count = odd.size() - (shift && !(odd.back() >> shift));
  Limb single;
  if (count == 1) {
    single = odd[0] >> shift;
    if (shift && odd.size() == 2)
      single |= odd[1] << (GMP_NUMB_BITS - shift);
    odd = std::span(&single, 1);
    zeros += shift;
    shift = 0;
  }

  // The power has at most exponent * bits bits, of which the shift supplies
  // exponent * zeros. Allocation rejects a saturated count.
  std::uintptr_t total = std::mul_sat(exponent, zeros + shift);
  if (odd.size() == 1 && odd[0] == 1) {
    release<counting>(base);
    return shiftLeft<counting>(Handle::immediate(negative ? -1 : 1), total,
                               false);
  }
  std::size_t shiftLimbs = total / GMP_NUMB_BITS;
  unsigned shiftBits = total % GMP_NUMB_BITS;
  // Each buffer holds the power with a limb to spare, for the most
  // significant limb that mpn_sqr and mpn_mul write and may leave zero, or for
  // the final carry.
  std::size_t size =
      std::mul_sat(exponent, bits - zeros - shift) / GMP_NUMB_BITS + 2;
  std::size_t required = std::add_sat(shiftLimbs, size);

  // A one-limb factor takes fewer steps as the power of m^k, which then
  // multiplies m^r.
  Limb rest = 1;
  if (odd.size() == 1) {
    LimbPowers powers = limbPowers(single, exponent);
    single = powers.power;
    rest = powers.rest;
    exponent = powers.quotient;
  }
  unsigned squarings = exponent ? std::bit_width(exponent) - 1 : 0;
  unsigned switches = squarings + (count > 1 ? std::popcount(exponent) - 1 : 0);
  // The result is written over a reusable base only after its odd factor is
  // copied out, as are its zero bits when shifted out.
  bool reusable =
      reuse && !base.isImmediate() && base.heap().capacity >= required;
  bool copy = count > 1 && (shift || reusable);

  // mpn_sqr and mpn_mul cannot write over their operands, so a second buffer,
  // on the stack when short, alternates with the result.
  std::size_t alternate = switches > 1 ? size : 0;
  std::size_t scratchCount = alternate + (copy ? odd.size() : 0);
  Limb stack[16];
  Mpz *spill =
      scratchCount > std::size(stack) ? &allocate(scratchCount) : nullptr;
  Limb *scratch = spill ? spill->limbs().data() : stack;
  if (copy) {
    Limb *factor = scratch + alternate;
    if (shift)
      mpn_rshift(factor, odd.data(), odd.size(), shift);
    else
      std::ranges::copy(odd, factor);
    odd = std::span(factor, count);
  }
  Mpz &result = reusable ? base.heap() : allocateResult(required);
  Limb *out = result.limbs().data() + shiftLimbs;

  // Squaring and multiplying by a factor of several limbs write to the other
  // buffer, starting with the one that makes the last write land in out.
  // Multiplying by one limb permits writing over the power.
  const Limb *power = odd.data();
  std::size_t n = odd.size();
  Limb *next = switches % 2 ? out : scratch, *current = nullptr;
  auto advance = [&] {
    current = next;
    power = current;
    next = next == out ? scratch : out;
  };
  for (unsigned bit = squarings; bit-- > 0;) {
    mpn_sqr(next, power, n);
    n = 2 * n - !next[2 * n - 1];
    advance();
    if (!(exponent >> bit & 1))
      continue;
    if (count == 1) {
      current[n] = mpn_mul_1(current, current, n, single);
      n += current[n] != 0;
      continue;
    }
    mpn_mul(next, power, n, odd.data(), odd.size());
    n += odd.size();
    n -= !next[n - 1];
    advance();
  }
  // Without squarings, the power of m^k is m^k itself or one.
  if (!switches)
    *out = exponent ? single : 1;
  if (rest != 1) {
    out[n] = mpn_mul_1(out, out, n, rest);
    n += out[n] != 0;
  }
  if (shiftBits) {
    out[n] = mpn_lshift(out, out, n, shiftBits);
    n += out[n] != 0;
  }
  std::fill(result.limbs().data(), out, Limb{0});
  if (spill)
    deallocate<counting>(*spill);
  result.setMagnitude(shiftLimbs + n, negative);
  if (Handle(result) != base)
    release<counting>(base);
  // A power that overflows a limb has at least two limbs.
  return Handle(result);
}

// Raises base to the power exponent. Small powers of immediates stay out of
// the general path, which has more state to set up.
export template <Counting counting>
Handle power(Handle base, std::uintptr_t exponent, bool reuse) {
  if (exponent == 1)
    return base; // The consumed reference becomes the result's.
  if (!exponent) {
    release<counting>(base);
    return Handle::immediate(1);
  }
  if (base.isImmediate()) {
    std::intptr_t value = base.value();
    bool negative = value < 0 && exponent & 1;
    Limb magnitude = value < 0 ? -static_cast<Limb>(value) : value;
    // Zero, one, and -1 are their own powers, but for the even powers of -1.
    if (magnitude <= 1)
      return Handle::immediate(negative ? -1 : value != 0);
    // Powers of two are shifts of one, which allocation rejects when the
    // amount saturates.
    if (std::has_single_bit(magnitude)) {
      std::uintptr_t amount =
          std::mul_sat<std::uintptr_t>(exponent, std::countr_zero(magnitude));
      if (amount < GMP_NUMB_BITS)
        if (std::optional<Handle> small =
                Handle::immediate(Limb{1} << amount, negative))
          return *small;
      return shiftLeft<counting>(Handle::immediate(negative ? -1 : 1), amount,
                                 false);
    }
    // (m^k)^q * m^r has at most two limbs for q below two. Otherwise m^k has
    // at least 32 bits, so the power has at least two limbs.
    LimbPowers powers = limbPowers(magnitude, exponent);
    if (powers.quotient < 2) {
      unsigned __int128 product = powers.rest;
      if (powers.quotient)
        product *= powers.power;
      std::array<std::uintptr_t, 2> words{
          static_cast<std::uintptr_t>(product),
          static_cast<std::uintptr_t>(product >> GMP_NUMB_BITS)};
      if (!words[1])
        if (std::optional<Handle> small = Handle::immediate(words[0], negative))
          return *small;
      return fromWords(words, negative);
    }
  }
  return largePower<counting>(base, exponent, reuse);
}

// Adapts a power to the runtime ABI, where bit 0 of reuse permits reusing the
// base.
template <auto power>
void *powerEntryPoint(void *base, std::uintptr_t exponent,
                      std::uint32_t reuse) {
  return power(Handle(base), exponent, (reuse & 1) != 0).pointer();
}

} // namespace rcgmp

extern "C" {

void *rcgmp_mpz_pow(void *base, std::uintptr_t exponent, std::uint32_t reuse) {
  return rcgmp::powerEntryPoint<rcgmp::power<rcgmp::Counting::nonatomic>>(
      base, exponent, reuse);
}

void *rcgmp_mpz_pow_atomic(void *base, std::uintptr_t exponent,
                           std::uint32_t reuse) {
  return rcgmp::powerEntryPoint<rcgmp::power<rcgmp::Counting::atomic>>(
      base, exponent, reuse);
}
}
