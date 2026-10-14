#include "MpzArithmetic.h"

static void *addsub(void *lhs, void *rhs, uint32_t reuse, uint32_t atomic,
                    int subtract) {
  Magnitude left, right;
  magnitude(&left, lhs);
  magnitude(&right, rhs);
  if (subtract && right.count)
    right.negative = !right.negative;
  if (!left.count && !subtract)
    return rhs; // The consumed zero has no reference count.
  if (!right.count)
    return lhs;

  // The larger magnitude leads both addition and subtraction. Swap views, not
  // their storage: immediate limb pointers refer to fields in left/right.
  Magnitude *a = &left, *b = &right;
  int sameSign = a->negative == b->negative;
  int order = a->count > b->count ? 1 : (a->count < b->count ? -1 : 0);
  if (!sameSign && !order)
    order = mpn_cmp(a->limbs, b->limbs, a->count);
  if (!sameSign && !order) {
    rcgmp_mpz_release(lhs, atomic);
    rcgmp_mpz_release(rhs, atomic);
    return rcgmp_encode(0);
  }
  if (order < 0) {
    a = &right;
    b = &left;
  }

  if (a->count == 1 && rcgmp_is_immediate(lhs) && rcgmp_is_immediate(rhs)) {
    // The sum or difference of two signed W-2-bit payloads fits intptr_t.
    intptr_t sum = subtract ? rcgmp_decode(lhs) - rcgmp_decode(rhs)
                            : rcgmp_decode(lhs) + rcgmp_decode(rhs);
    uintptr_t word = sum < 0 ? (uintptr_t)-sum : (uintptr_t)sum;
    return rcgmp_mpz_from_words(&word, 1, sum < 0, atomic);
  }

  size_t count = a->count;
  RCGMPMpz *result = destination(lhs, rhs, reuse, count + sameSign, atomic);
  if (sameSign) {
    mp_limb_t carry;
    if (b->count <= 1)
      carry = mpn_add_1(result->limbs, a->limbs, count, b->limbs[0]);
    else if (a->limbs == b->limbs)
      carry = mpn_lshift(result->limbs, a->limbs, count, 1);
    else {
      carry = mpn_add_n(result->limbs, a->limbs, b->limbs, b->count);
      if (count > b->count)
        carry = mpn_add_1(result->limbs + b->count, a->limbs + b->count,
                          count - b->count, carry);
    }
    result->limbs[count] = carry;
    count += carry;
  } else {
    mp_limb_t borrow;
    if (b->count <= 1)
      borrow = mpn_sub_1(result->limbs, a->limbs, count, b->limbs[0]);
    else {
      borrow = mpn_sub_n(result->limbs, a->limbs, b->limbs, b->count);
      if (count > b->count)
        borrow = mpn_sub_1(result->limbs + b->count, a->limbs + b->count,
                           count - b->count, borrow);
    }
    (void)borrow; // Magnitude ordering guarantees no final borrow.
  }
  result->size = a->negative ? -(intptr_t)count : (intptr_t)count;
  return finish(result, lhs, rhs, atomic);
}

void *rcgmp_mpz_add(void *lhs, void *rhs, uint32_t reuse) {
  return addsub(lhs, rhs, reuse, 0, 0);
}

void *rcgmp_mpz_add_atomic(void *lhs, void *rhs, uint32_t reuse) {
  return addsub(lhs, rhs, reuse, 1, 0);
}

void *rcgmp_mpz_sub(void *lhs, void *rhs, uint32_t reuse) {
  return addsub(lhs, rhs, reuse, 0, 1);
}

void *rcgmp_mpz_sub_atomic(void *lhs, void *rhs, uint32_t reuse) {
  return addsub(lhs, rhs, reuse, 1, 1);
}
