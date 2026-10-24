#ifndef RCGMP_RUNTIME_HASH_H
#define RCGMP_RUNTIME_HASH_H

#include <cstdint>

// The hash of an integer outside the range of int64_t, shared by the runtime
// and constant folding: a state started from the sign absorbs the magnitude in
// 64-bit chunks, least significant first, and is finished with the number of
// chunks. Integers in the range of int64_t hash to themselves.
namespace rcgmp::hashing {

constexpr std::uint64_t start(bool negative) {
  return negative ? 0x9e3779b97f4a7c15 : 0;
}

constexpr std::uint64_t absorb(std::uint64_t state, std::uint64_t chunk) {
  state = (state ^ chunk) * 0xff51afd7ed558ccd;
  return state ^ (state >> 32);
}

// MurmurHash3's finalizer, which every bit of the state affects.
constexpr std::uint64_t finish(std::uint64_t state, std::uint64_t chunks) {
  state ^= chunks;
  state = (state ^ (state >> 33)) * 0xff51afd7ed558ccd;
  state = (state ^ (state >> 33)) * 0xc4ceb9fe1a85ec53;
  return state ^ (state >> 33);
}

} // namespace rcgmp::hashing

#endif
