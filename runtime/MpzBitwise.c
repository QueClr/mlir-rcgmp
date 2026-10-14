#include "MpzArithmetic.h"

// Stream a negative magnitude as ~magnitude + 1, including sign extension.
static mp_limb_t twosComplementLimb(Magnitude *value, size_t index,
                                    mp_limb_t *carry) {
  mp_limb_t limb = index < value->count ? value->limbs[index] : 0;
  if (!value->negative)
    return limb;
  limb = ~limb + *carry;
  *carry = *carry && limb == 0;
  return limb;
}

static void *bitwise(void *lhs, void *rhs, uint32_t reuse, uint32_t atomic,
                     int exclusiveOr) {
  Magnitude left, right;
  magnitude(&left, lhs);
  magnitude(&right, rhs);
  int negative = exclusiveOr ? left.negative != right.negative
                             : left.negative && right.negative;
  size_t count;
  if (exclusiveOr || negative) {
    count = left.count > right.count ? left.count : right.count;
    // Converting a negative result back to magnitude may carry into a new limb.
    count += negative;
  } else if (left.negative) {
    count = right.count;
  } else if (right.negative) {
    count = left.count;
  } else {
    count = left.count < right.count ? left.count : right.count;
  }
  if (!count) {
    rcgmp_mpz_release(lhs, atomic);
    rcgmp_mpz_release(rhs, atomic);
    return rcgmp_encode(0);
  }
  RCGMPMpz *result = destination(lhs, rhs, reuse, count, atomic);
  if (!left.negative && !right.negative) {
    size_t common = left.count < right.count ? left.count : right.count;
    if (common) {
      if (exclusiveOr)
        mpn_xor_n(result->limbs, left.limbs, right.limbs, common);
      else
        mpn_and_n(result->limbs, left.limbs, right.limbs, common);
    }
    if (count > common) {
      Magnitude *longer = left.count > right.count ? &left : &right;
      if (result->limbs != longer->limbs)
        mpn_copyi(result->limbs + common, longer->limbs + common,
                  count - common);
    }
  } else {
    // Read both source limbs before writing, so either unique input can be
    // reused without a temporary two's-complement allocation.
    mp_limb_t leftCarry = 1, rightCarry = 1, resultCarry = 1;
    for (size_t i = 0; i < count; ++i) {
      mp_limb_t a = twosComplementLimb(&left, i, &leftCarry);
      mp_limb_t b = twosComplementLimb(&right, i, &rightCarry);
      mp_limb_t limb = exclusiveOr ? a ^ b : a & b;
      if (negative) {
        limb = ~limb + resultCarry;
        resultCarry = resultCarry && limb == 0;
      }
      result->limbs[i] = limb;
    }
  }
  result->size = negative ? -(intptr_t)count : (intptr_t)count;
  return finish(result, lhs, rhs, atomic);
}

void *rcgmp_mpz_and(void *lhs, void *rhs, uint32_t reuse) {
  return bitwise(lhs, rhs, reuse, 0, 0);
}

void *rcgmp_mpz_and_atomic(void *lhs, void *rhs, uint32_t reuse) {
  return bitwise(lhs, rhs, reuse, 1, 0);
}

void *rcgmp_mpz_xor(void *lhs, void *rhs, uint32_t reuse) {
  return bitwise(lhs, rhs, reuse, 0, 1);
}

void *rcgmp_mpz_xor_atomic(void *lhs, void *rhs, uint32_t reuse) {
  return bitwise(lhs, rhs, reuse, 1, 1);
}
