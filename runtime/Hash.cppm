module;
#include "RCGMP/Runtime/Hash.h"
#include "RCGMP/Runtime/Mpz.h"
#include <gmp.h>

#include <cstddef>
#include <cstdint>
#include <span>
export module rcgmp.runtime:hash;
import :mpz;

namespace rcgmp {

// Hashes a borrowed integer by its value, as rcgmp.hashi defines it.
export std::int64_t hash(Handle value) {
  if (value.isImmediate())
    return value.value();
  Mpz &heap = value.heap();
  std::span<const Limb> limbs = heap.magnitude();
  // The magnitude in 64-bit chunks, whatever the limb width.
  constexpr std::size_t perChunk = 64 / GMP_NUMB_BITS;
  std::size_t chunks = (limbs.size() + perChunk - 1) / perChunk;
  auto chunk = [&](std::size_t index) {
    std::uint64_t bits = 0;
    for (std::size_t part = 0;
         part < perChunk && index * perChunk + part < limbs.size(); ++part)
      bits |= static_cast<std::uint64_t>(limbs[index * perChunk + part])
              << (part * GMP_NUMB_BITS);
    return bits;
  };
  bool negative = heap.negative();
  constexpr std::uint64_t signBit = std::uint64_t{1} << 63;
  if (chunks == 1)
    if (std::uint64_t magnitude = chunk(0);
        negative ? magnitude <= signBit : magnitude < signBit)
      return static_cast<std::int64_t>(negative ? 0 - magnitude : magnitude);
  std::uint64_t state = hashing::start(negative);
  for (std::size_t index = 0; index < chunks; ++index)
    state = hashing::absorb(state, chunk(index));
  return static_cast<std::int64_t>(hashing::finish(state, chunks));
}

} // namespace rcgmp

extern "C" {

std::int64_t rcgmp_mpz_hash(void *value) {
  return rcgmp::hash(rcgmp::Handle(value));
}

// Hashing reads no reference count, so atomicity makes no difference.
std::int64_t rcgmp_mpz_hash_atomic(void *value) {
  return rcgmp_mpz_hash(value);
}
}
