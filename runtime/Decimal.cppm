module;
#include "RCGMP/Runtime/Allocation.h"
#include "RCGMP/Runtime/Mpz.h"
#include <gmp.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>
export module rcgmp.runtime:decimal;
import :mpz;

namespace rcgmp {

// The most decimal digits of an immediate: those of its bound, 2^(W-3).
constexpr std::size_t immediateDigits =
    std::numeric_limits<std::intptr_t>::digits10 + 1;

// A read-only GMP integer sharing the limbs of a heap integer.
mpz_srcptr readOnly(Mpz &heap, mpz_ptr storage) {
  return mpz_roinit_n(storage, heap.limbs().data(), heap.size);
}

// Room for the digits, a sign, and the terminator mpz_get_str writes.
export std::size_t decimalSize(Handle value) {
  if (value.isImmediate())
    return immediateDigits + 2;
  mpz_t storage;
  return mpz_sizeinbase(readOnly(value.heap(), storage), 10) + 2;
}

export std::size_t toDecimal(Handle value, char *buffer) {
  if (value.isImmediate())
    return std::to_chars(buffer, buffer + immediateDigits + 1, value.value())
               .ptr -
           buffer;
  mpz_t storage;
  return std::strlen(mpz_get_str(buffer, 10, readOnly(value.heap(), storage)));
}

// Parses an optional minus sign and one or more decimal digits. The result
// owns one reference, and is null if text has another form.
export void *fromDecimal(std::string_view text) {
  bool negative = text.starts_with('-');
  if (negative)
    text.remove_prefix(1);
  if (text.empty() || !std::ranges::all_of(text, [](char digit) {
        return digit >= '0' && digit <= '9';
      }))
    return nullptr;
  text.remove_prefix(std::min(text.find_first_not_of('0'), text.size()));
  // Up to digits10 digits fit a word, and its immediate when they fit.
  if (text.size() <= std::numeric_limits<std::uintptr_t>::digits10) {
    std::uintptr_t magnitude = 0;
    std::from_chars(text.data(), text.data() + text.size(), magnitude);
    return rcgmp_mpz_from_word(magnitude, negative, 0);
  }
  // mpn_set_str reads digit values rather than characters, needs room for
  // the largest value of that many digits plus one limb, and returns no
  // leading zero limbs for a leading digit that is not zero. 3402 / 1024
  // exceeds log2(10).
  std::size_t count = text.size();
  if (count > std::numeric_limits<std::size_t>::max() / 3402)
    std::abort();
  Mpz &value = allocate((count * 3402 / 1024 + 1) / GMP_NUMB_BITS + 2);
  constexpr std::size_t stackDigits = 1024;
  std::array<unsigned char, stackDigits> stack;
  auto *digits = count <= stackDigits
                     ? stack.data()
                     : static_cast<unsigned char *>(rcgmp_allocate(count, 1));
  if (!digits)
    std::abort();
  std::ranges::transform(text, digits, [](char digit) {
    return static_cast<unsigned char>(digit - '0');
  });
  mp_size_t limbs = mpn_set_str(value.limbs().data(), digits, count, 10);
  if (digits != stack.data())
    rcgmp_dealloc(digits, count, 1);
  value.setMagnitude(limbs, negative);
  return Handle(value).pointer();
}

} // namespace rcgmp

extern "C" {

std::size_t rcgmp_mpz_decimal_size(void *value) {
  return rcgmp::decimalSize(rcgmp::Handle(value));
}

std::size_t rcgmp_mpz_to_decimal(void *value, char *buffer) {
  return rcgmp::toDecimal(rcgmp::Handle(value), buffer);
}

void *rcgmp_mpz_from_decimal(const char *text, std::size_t length,
                             std::uint32_t) {
  return rcgmp::fromDecimal({text, length});
}
}
