module;
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
export module rcgmp.runtime:binary;
import :mpz;

namespace rcgmp {

// The operand positions whose storage a binary operation may reuse. Only a
// heap operand proven uniquely owned may be reused.
export struct Reuse {
  bool lhs = false;
  bool rhs = false;

  // Decodes the runtime ABI flags: bit 0 for lhs and bit 1 for rhs.
  static constexpr Reuse fromBits(std::uint32_t bits) {
    return {.lhs = (bits & 1) != 0, .rhs = (bits & 2) != 0};
  }
};

// The sign and magnitude of an operand. An immediate's magnitude is stored in
// the view itself, so mixed operations never box it.
class Magnitude {
public:
  inline explicit Magnitude(Handle value) {
    if (value.isImmediate()) {
      std::intptr_t integer = value.value();
      negative = integer < 0;
      small = negative ? -static_cast<Limb>(integer) : integer;
      limbs = {&small, integer != 0};
    } else {
      negative = value.heap().negative();
      limbs = value.heap().magnitude();
    }
  }
  Magnitude(const Magnitude &) =
      delete("an immediate's limbs refer to the view's own storage");

  constexpr std::size_t count() const { return limbs.size(); }
  // The limb at index, which is zero beyond the magnitude.
  constexpr Limb limb(std::size_t index) const {
    return index < count() ? limbs[index] : 0;
  }

  std::span<const Limb> limbs;
  bool negative;

private:
  Limb small;
};

// A new integer for a result of up to required limbs. Geometric headroom
// amortizes repeated growth. Allocation rejects the saturated capacity.
inline Mpz &allocateResult(std::size_t required) {
  return allocate(std::add_sat(required, required / 2 + 1));
}

// Returns storage for a result of up to required limbs: a reusable operand with
// enough capacity, or else a new integer.
inline Mpz &destination(Handle lhs, Handle rhs, Reuse reuse,
                        std::size_t required) {
  if (reuse.lhs && lhs.heap().capacity >= required)
    return lhs.heap();
  if (reuse.rhs && rhs.heap().capacity >= required)
    return rhs.heap();
  return allocateResult(required);
}

// Consumes both operands of a result that needs no storage of its own.
template <Counting counting>
Handle consume(Handle lhs, Handle rhs, Handle result) {
  release<counting>(lhs);
  release<counting>(rhs);
  return result;
}

// Consumes the operands after all source limbs have been read, then
// canonicalizes the result.
template <Counting counting>
Handle finish(Mpz &result, Handle lhs, Handle rhs) {
  if (Handle(result) != lhs)
    release<counting>(lhs);
  if (Handle(result) != rhs)
    release<counting>(rhs);
  return normalize<counting>(result);
}

// Adapts a binary operation to the runtime ABI.
template <auto operation>
void *binaryEntryPoint(void *lhs, void *rhs, std::uint32_t reuse) {
  return operation(Handle(lhs), Handle(rhs), Reuse::fromBits(reuse)).pointer();
}

} // namespace rcgmp
