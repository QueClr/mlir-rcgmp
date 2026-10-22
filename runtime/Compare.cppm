module;
#include "RCGMP/Runtime/Mpz.h"
#include <gmp.h>

#include <compare>
#include <cstdint>
export module rcgmp.runtime:compare;
import :binary;
import :mpz;

namespace rcgmp {

// Compares two borrowed integers.
export std::strong_ordering compare(Handle lhs, Handle rhs) {
  if (lhs.isImmediate() && rhs.isImmediate())
    return lhs.value() <=> rhs.value();
  Magnitude left(lhs), right(rhs);
  if (left.negative != right.negative)
    return left.negative ? std::strong_ordering::less
                         : std::strong_ordering::greater;
  // Canonical magnitudes have no leading zero limbs, so the longer one is
  // larger. A heap operand has at least one limb, so equal counts are nonzero.
  std::strong_ordering order = left.count() <=> right.count();
  if (order == 0)
    order = mpn_cmp(left.limbs.data(), right.limbs.data(), left.count()) <=> 0;
  return left.negative ? 0 <=> order : order;
}

} // namespace rcgmp

extern "C" {

std::int32_t rcgmp_mpz_cmp(void *lhs, void *rhs) {
  std::strong_ordering order =
      rcgmp::compare(rcgmp::Handle(lhs), rcgmp::Handle(rhs));
  return (order > 0) - (order < 0);
}

// Comparison reads no reference count, so atomicity makes no difference.
std::int32_t rcgmp_mpz_cmp_atomic(void *lhs, void *rhs) {
  return rcgmp_mpz_cmp(lhs, rhs);
}
}
