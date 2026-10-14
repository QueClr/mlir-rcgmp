#include "MpzArithmetic.h"

static void *multiply(void *lhs, void *rhs, uint32_t reuse, uint32_t atomic) {
  Magnitude left, right;
  magnitude(&left, lhs);
  magnitude(&right, rhs);
  if (!left.count || !right.count) {
    rcgmp_mpz_release(lhs, atomic);
    rcgmp_mpz_release(rhs, atomic);
    return rcgmp_encode(0);
  }
  Magnitude *a = &left, *b = &right;
  if (a->count < b->count) {
    a = &right;
    b = &left;
  }
  size_t count = a->count + b->count;
  // mpn_mul_1 permits exact overlap. General mpn_mul/mpn_sqr do not.
  RCGMPMpz *result =
      destination(lhs, rhs, b->count == 1 ? reuse : 0, count, atomic);
  if (b->count == 1) {
    mp_limb_t factor = b->limbs[0];
    mp_limb_t carry;
    if (factor == 1) {
      if (result->limbs != a->limbs)
        mpn_copyi(result->limbs, a->limbs, a->count);
      carry = 0;
    } else if ((factor & (factor - 1)) == 0) {
      unsigned shift = 0;
      for (mp_limb_t bits = factor; bits > 1; bits >>= 1)
        ++shift;
      carry = mpn_lshift(result->limbs, a->limbs, a->count, shift);
    } else {
      carry = mpn_mul_1(result->limbs, a->limbs, a->count, factor);
    }
    result->limbs[a->count] = carry;
  } else if (a->limbs == b->limbs) {
    mpn_sqr(result->limbs, a->limbs, a->count);
  } else {
    mpn_mul(result->limbs, a->limbs, a->count, b->limbs, b->count);
  }
  result->size =
      a->negative != b->negative ? -(intptr_t)count : (intptr_t)count;
  return finish(result, lhs, rhs, atomic);
}

void *rcgmp_mpz_mul(void *lhs, void *rhs, uint32_t reuse) {
  return multiply(lhs, rhs, reuse, 0);
}

void *rcgmp_mpz_mul_atomic(void *lhs, void *rhs, uint32_t reuse) {
  return multiply(lhs, rhs, reuse, 1);
}
