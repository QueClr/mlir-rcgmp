module;
#include "RCGMP/Runtime/Mpz.h"
#include <gmp.h>

#include <algorithm>
#include <array>
#include <bit>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <span>
#include <type_traits>
export module rcgmp.runtime:floating;
import :mpz;

namespace rcgmp {

// The 64 bits of a magnitude from bit position up, zero beyond its limbs.
std::uint64_t bitsAt(std::span<const Limb> limbs, std::size_t position) {
  std::uint64_t bits = 0;
  for (std::size_t offset = 0; offset < 64;) {
    std::size_t index = (position + offset) / GMP_NUMB_BITS,
                shift = (position + offset) % GMP_NUMB_BITS;
    if (index >= limbs.size())
      break;
    bits |= static_cast<std::uint64_t>(limbs[index] >> shift) << offset;
    offset += GMP_NUMB_BITS - shift;
  }
  return bits;
}

// Whether any of the bits of a magnitude below position is set.
bool anyBelow(std::span<const Limb> limbs, std::size_t position) {
  std::size_t index = position / GMP_NUMB_BITS,
              shift = position % GMP_NUMB_BITS;
  if (shift && (limbs[index] & ((Limb{1} << shift) - 1)))
    return true;
  return std::ranges::any_of(limbs.first(index),
                             [](Limb limb) { return limb != 0; });
}

// The IEEE binary representation of Float.
template <typename Float>
using FloatBits =
    std::conditional_t<sizeof(Float) == 4, std::uint32_t, std::uint64_t>;
template <typename Float>
constexpr int significandBits = std::numeric_limits<Float>::digits - 1;

// A positive normal value times 2^shift, adding to its exponent field, which
// is exact, or infinity when the exponent overflows.
template <typename Float> Float scaled(Float value, std::size_t shift) {
  using Bits = FloatBits<Float>;
  constexpr Bits infiniteExponent =
      std::bit_cast<Bits>(std::numeric_limits<Float>::infinity()) >>
      significandBits<Float>;
  Bits bits = std::bit_cast<Bits>(value);
  if (shift >= infiniteExponent - (bits >> significandBits<Float>))
    return std::numeric_limits<Float>::infinity();
  return std::bit_cast<Float>(
      bits + (static_cast<Bits>(shift) << significandBits<Float>));
}

// Rounds a borrowed integer to the nearest Float, ties to even.
export template <typename Float> Float toFloating(Handle value) {
  if (value.isImmediate())
    return static_cast<Float>(value.value());
  Mpz &heap = value.heap();
  std::span<const Limb> limbs = heap.magnitude();
  std::size_t length =
      (limbs.size() - 1) * GMP_NUMB_BITS + std::bit_width(limbs.back());
  Float magnitude;
  if (length <= 64) {
    magnitude = static_cast<Float>(bitsAt(limbs, 0));
  } else {
    // Converting the top 64 bits rounds once. The lowest of them lies below
    // the rounding position of every Float, so setting it when any lower bit
    // is set breaks ties exactly as the full magnitude does. Scaling is then
    // exact, or overflows to infinity.
    std::size_t shift = length - 64;
    std::uint64_t top = bitsAt(limbs, shift) | anyBelow(limbs, shift);
    magnitude = scaled(static_cast<Float>(top), shift);
  }
  return heap.negative() ? -magnitude : magnitude;
}

// The integer a finite value truncates to, owning one reference. A NaN or an
// infinity terminates the program.
export Handle fromFloating(double value) {
  if (!std::isfinite(value))
    std::abort();
  // A nonzero truncated magnitude is a normal value, its significand with the
  // implicit bit times 2^shift, and an integer.
  using Bits = FloatBits<double>;
  constexpr int precision = significandBits<double>;
  Bits bits = std::bit_cast<Bits>(std::fabs(std::trunc(value)));
  if (!bits)
    return Handle::immediate(0);
  std::uint64_t mantissa =
      (bits & ((Bits{1} << precision) - 1)) | (Bits{1} << precision);
  int shift = static_cast<int>(bits >> precision) - 1023 - precision;
  if (shift < 0) {
    mantissa >>= -shift;
    shift = 0;
  }
  constexpr std::size_t width = sizeof(std::uintptr_t) * CHAR_BIT;
  std::array<std::uintptr_t, (1024 + 64) / width + 1> words{};
  for (std::size_t offset = 0; offset < 64; offset += width) {
    std::size_t index = (shift + offset) / width,
                within = (shift + offset) % width;
    auto part = static_cast<std::uintptr_t>(mantissa >> offset);
    words[index] |= part << within;
    if (within)
      words[index + 1] |= part >> (width - within);
  }
  return fromWords(words, std::signbit(value));
}

} // namespace rcgmp

extern "C" {

// Rounding reads no reference count, so atomicity makes no difference.
double rcgmp_mpz_to_f64(void *value) {
  return rcgmp::toFloating<double>(rcgmp::Handle(value));
}
double rcgmp_mpz_to_f64_atomic(void *value) { return rcgmp_mpz_to_f64(value); }
float rcgmp_mpz_to_f32(void *value) {
  return rcgmp::toFloating<float>(rcgmp::Handle(value));
}
float rcgmp_mpz_to_f32_atomic(void *value) { return rcgmp_mpz_to_f32(value); }

// A new integer has no other owner, so its counting needs no atomicity.
void *rcgmp_mpz_from_f64(double value) {
  return rcgmp::fromFloating(value).pointer();
}
void *rcgmp_mpz_from_f64_atomic(double value) {
  return rcgmp_mpz_from_f64(value);
}
}
