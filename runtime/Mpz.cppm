module;
#include "RCGMP/Runtime/Allocation.h"
#include "RCGMP/Runtime/Mpz.h"
#include <gmp.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <optional>
#include <span>
#include <type_traits>
export module rcgmp.runtime:mpz;

namespace rcgmp {

export using Limb = mp_limb_t;

// The reference counting of an integer, fixed by its MLIR type.
export enum class Counting : bool { nonatomic, atomic };

// The heap representation shared with compiled code: a reference count, the
// limb count negated for negative values, and the limb capacity, followed by
// the limbs, least significant first.
export struct Mpz {
  std::uintptr_t references;
  std::intptr_t size;
  std::uintptr_t capacity;

  inline std::span<Limb> limbs() {
    return {reinterpret_cast<Limb *>(this + 1), capacity};
  }
  constexpr bool negative() const { return size < 0; }
  constexpr std::size_t count() const {
    return static_cast<std::size_t>(negative() ? -size : size);
  }
  inline std::span<const Limb> magnitude() { return limbs().first(count()); }
  constexpr void setMagnitude(std::size_t count, bool negative) {
    auto signedCount = static_cast<std::intptr_t>(count);
    size = negative ? -signedCount : signedCount;
  }
};

// An integer as passed to and returned from compiled code: an immediate n
// encoded as 4*n + 1, or a pointer to a heap integer.
export class Handle {
public:
  static constexpr std::intptr_t immediateMax =
      std::numeric_limits<std::intptr_t>::max() / 4;
  static constexpr std::intptr_t immediateMin = -immediateMax - 1;

  inline explicit Handle(void *pointer)
      : bits(std::bit_cast<std::uintptr_t>(pointer)) {}
  inline explicit Handle(Mpz &heap) : Handle(&heap) {}

  // value must lie in the immediate range.
  static constexpr Handle immediate(std::intptr_t value) {
    return Handle((static_cast<std::uintptr_t>(value) << 2) | 1);
  }
  // The immediate with the given sign and magnitude, if it fits.
  static constexpr std::optional<Handle> immediate(std::uintptr_t magnitude,
                                                   bool negative) {
    if (magnitude > static_cast<std::uintptr_t>(immediateMax) + negative)
      return std::nullopt;
    return Handle(((negative ? -magnitude : magnitude) << 2) | 1);
  }

  constexpr bool isImmediate() const { return bits & 1; }
  // The value of an immediate.
  constexpr std::intptr_t value() const {
    return static_cast<std::intptr_t>(bits) >> 2;
  }
  // The integer a heap handle points to.
  inline Mpz &heap() const { return *std::bit_cast<Mpz *>(bits); }
  inline void *pointer() const { return std::bit_cast<void *>(bits); }

  constexpr bool operator==(const Handle &) const = default;

private:
  constexpr explicit Handle(std::uintptr_t bits) : bits(bits) {}

  std::uintptr_t bits;
};

// Drops the most significant zero words of a magnitude.
template <typename Word>
constexpr std::span<const Word> trimmed(std::span<const Word> words) {
  while (!words.empty() && !words.back())
    words = words.first(words.size() - 1);
  return words;
}

// Returns a heap integer with one reference, no limbs in use, and room for
// capacity limbs. Terminates if allocation fails.
export Mpz &allocate(std::size_t capacity) {
  // The limit keeps both the signed size and the allocation size in range.
  constexpr std::size_t maxCapacity = std::min<std::size_t>(
      std::numeric_limits<std::intptr_t>::max(),
      (std::numeric_limits<std::size_t>::max() - sizeof(Mpz)) / sizeof(Limb));
  if (capacity > maxCapacity)
    std::abort();
  void *storage =
      rcgmp_allocate(sizeof(Mpz) + capacity * sizeof(Limb), alignof(Mpz));
  if (!storage)
    std::abort();
  return *::new(storage) Mpz{.references = 1, .size = 0, .capacity = capacity};
}

// Only for a heap integer whose final release has already completed.
export template <Counting counting> void deallocate(Mpz &value) {
  // Make the other owners' accesses, ordered by their releases, happen first.
  if constexpr (counting == Counting::atomic)
    std::atomic_thread_fence(std::memory_order::acquire);
  rcgmp_dealloc(&value, sizeof(Mpz) + value.capacity * sizeof(Limb),
                alignof(Mpz));
}

// A counted integer holds at least one reference while it is live, so a zero
// count identifies a static immortal integer for its whole lifetime.
template <Counting counting> bool isImmortal(Mpz &value) {
  if constexpr (counting == Counting::atomic)
    return std::atomic_ref<std::uintptr_t>(value.references)
               .load(std::memory_order::relaxed) == 0;
  else
    return value.references == 0;
}

export template <Counting counting> void retain(Handle value) {
  if (value.isImmediate() || isImmortal<counting>(value.heap()))
    return;
  std::uintptr_t &references = value.heap().references;
  std::uintptr_t old;
  if constexpr (counting == Counting::atomic)
    old = std::atomic_ref<std::uintptr_t>(references)
              .fetch_add(1, std::memory_order::relaxed);
  else
    old = references++;
  if (old == std::numeric_limits<std::uintptr_t>::max())
    std::abort();
}

export template <Counting counting> void release(Handle value) {
  if (value.isImmediate())
    return;
  // The only reference is released without updating the count, which no
  // other owner can observe. The acquire load orders the accesses of earlier
  // owners before the deallocation.
  std::uintptr_t &references = value.heap().references;
  std::uintptr_t count;
  if constexpr (counting == Counting::atomic)
    count = std::atomic_ref<std::uintptr_t>(references)
                .load(std::memory_order::acquire);
  else
    count = references;
  if (count == 1) {
    deallocate<counting>(value.heap());
    return;
  }
  if (count == 0)
    return;
  std::uintptr_t old;
  if constexpr (counting == Counting::atomic)
    old = std::atomic_ref<std::uintptr_t>(references)
              .fetch_sub(1, std::memory_order::release);
  else
    old = references--;
  if (old == 1)
    deallocate<counting>(value.heap());
}

// Returns the canonical integer with the given sign and magnitude, least
// significant word first. The result owns one reference.
export Handle fromWords(std::span<const std::uintptr_t> words, bool negative) {
  words = trimmed(words);
  if (words.size() <= 1)
    if (std::optional<Handle> small =
            Handle::immediate(words.empty() ? 0 : words[0], negative))
      return *small;
  // One spare limb allows the first growing addition to reuse this allocation.
  Mpz &value = allocate(words.size() + 1);
  std::ranges::copy(words, value.limbs().begin());
  value.setMagnitude(words.size(), negative);
  return Handle(value);
}

// Canonicalizes a heap result that may have most significant zero limbs,
// replacing it with an immediate when its value fits.
template <Counting counting> Handle normalize(Mpz &value) {
  bool negative = value.negative();
  std::span<const Limb> magnitude = trimmed(value.magnitude());
  if (magnitude.size() <= 1)
    if (std::optional<Handle> small =
            Handle::immediate(magnitude.empty() ? 0 : magnitude[0], negative)) {
      deallocate<counting>(value);
      return *small;
    }
  value.setMagnitude(magnitude.size(), negative);
  return Handle(value);
}

} // namespace rcgmp

static_assert(sizeof(rcgmp::Limb) == sizeof(std::uintptr_t),
              "GMP limb/handle ABI mismatch");
static_assert(GMP_NAIL_BITS == 0, "GMP nail bits are unsupported");
static_assert(std::is_standard_layout_v<rcgmp::Mpz> &&
                  sizeof(rcgmp::Mpz) == 3 * sizeof(std::uintptr_t),
              "MPZ ABI mismatch");
static_assert(alignof(rcgmp::Mpz) >= 4,
              "MPZ tagging requires four-byte alignment");
// Compiled code updates the same count with native atomic instructions.
static_assert(std::atomic_ref<std::uintptr_t>::is_always_lock_free &&
                  std::atomic_ref<std::uintptr_t>::required_alignment <=
                      alignof(std::uintptr_t),
              "RC requires lock-free native-word atomics");

extern "C" {

void *rcgmp_mpz_from_words(const std::uintptr_t *words, std::size_t count,
                           std::uint32_t negative, std::uint32_t) {
  // A new integer has no other owner, so its counting needs no atomicity.
  return rcgmp::fromWords({words, count}, negative).pointer();
}

void *rcgmp_mpz_from_word(std::uintptr_t magnitude, std::uint32_t negative,
                          std::uint32_t) {
  if (std::optional<rcgmp::Handle> small =
          rcgmp::Handle::immediate(magnitude, negative))
    return small->pointer();
  // One spare limb, as fromWords leaves.
  rcgmp::Mpz &value = rcgmp::allocate(2);
  value.limbs()[0] = magnitude;
  value.setMagnitude(1, negative);
  return rcgmp::Handle(value).pointer();
}

void *rcgmp_mpz_retain(void *value, std::uint32_t atomic) {
  if (atomic)
    rcgmp::retain<rcgmp::Counting::atomic>(rcgmp::Handle(value));
  else
    rcgmp::retain<rcgmp::Counting::nonatomic>(rcgmp::Handle(value));
  return value;
}

void rcgmp_mpz_release(void *value, std::uint32_t atomic) {
  if (atomic)
    rcgmp::release<rcgmp::Counting::atomic>(rcgmp::Handle(value));
  else
    rcgmp::release<rcgmp::Counting::nonatomic>(rcgmp::Handle(value));
}

void rcgmp_mpz_free(void *value, std::uint32_t atomic) {
  rcgmp::Mpz &heap = rcgmp::Handle(value).heap();
  if (atomic)
    rcgmp::deallocate<rcgmp::Counting::atomic>(heap);
  else
    rcgmp::deallocate<rcgmp::Counting::nonatomic>(heap);
}
}
