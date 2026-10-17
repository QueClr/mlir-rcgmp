#include "Test.h"

void *sub_local(void *, void *);
void *sub_atomic(void *, void *);
void *mul_local(void *, void *);
void *mul_atomic(void *, void *);
void *and_local(void *, void *);
void *and_atomic(void *, void *);
void *xor_local(void *, void *);
void *xor_atomic(void *, void *);

typedef struct {
  void *(*apply[2])(void *, void *);
  void (*oracle)(mpz_ptr, mpz_srcptr, mpz_srcptr);
} Binary;

static Binary operations[] = {
    {{sub_local, sub_atomic}, mpz_sub},
    {{mul_local, mul_atomic}, mpz_mul},
    {{and_local, and_atomic}, mpz_and},
    {{xor_local, xor_atomic}, mpz_xor},
};

static void check_pair(Binary *op, mpz_t lhs, mpz_t rhs, uint32_t atomic,
                       unsigned shared) {
  void *x = from_mpz(lhs, atomic), *y = from_mpz(rhs, atomic);
  if (shared & 1)
    rcgmp_mpz_retain(x, atomic);
  if (shared & 2)
    rcgmp_mpz_retain(y, atomic);
  mpz_t expected;
  mpz_init(expected);
  op->oracle(expected, lhs, rhs);
  void *result = op->apply[atomic](x, y);
  check_value(result, expected);
  if (shared & 1) {
    check_value(x, lhs);
    rcgmp_mpz_release(x, atomic);
  }
  if (shared & 2) {
    check_value(y, rhs);
    rcgmp_mpz_release(y, atomic);
  }
  rcgmp_mpz_release(result, atomic);
  mpz_clear(expected);
}

int main(void) {
  mpz_t lhs, rhs, expected;
  mpz_inits(lhs, rhs, expected, NULL);
  gmp_randstate_t random;
  gmp_randinit_default(random);
  gmp_randseed_ui(random, 2718);
  intptr_t root = (intptr_t)1 << (sizeof(uintptr_t) * CHAR_BIT / 2 - 1);
  intptr_t boundaries[] = {0,
                           1,
                           -1,
                           2,
                           -2,
                           root - 1,
                           root,
                           -root,
                           RCGMP_IMM_MIN,
                           RCGMP_IMM_MAX,
                           RCGMP_IMM_MIN - 1,
                           RCGMP_IMM_MAX + 1};
  for (size_t operation = 0; operation < 4; ++operation) {
    Binary *op = &operations[operation];
    for (uint32_t atomic = 0; atomic < 2; ++atomic) {
      for (size_t i = 0; i < sizeof(boundaries) / sizeof(*boundaries); ++i) {
        for (size_t j = 0; j < sizeof(boundaries) / sizeof(*boundaries); ++j) {
          mpz_set_si(lhs, boundaries[i]);
          mpz_set_si(rhs, boundaries[j]);
          for (unsigned shared = 0; shared < 4; ++shared)
            check_pair(op, lhs, rhs, atomic, shared);
        }
      }
      for (unsigned i = 0; i < 1200; ++i) {
        mpz_urandomb(lhs, random, (i * 73) % 1025);
        mpz_urandomb(rhs, random, (i * 131) % 1025);
        if (i & 1)
          mpz_neg(lhs, lhs);
        if (i & 2)
          mpz_neg(rhs, rhs);
        if (i % 7 == 0) {
          mpz_set(rhs, lhs);
          mpz_add_ui(rhs, rhs, i % 3);
        }
        for (unsigned shared = 0; shared < 4; ++shared)
          check_pair(op, lhs, rhs, atomic, shared);
      }

      // Long zero/one chains, negative sign extension, and an extra result
      // limb when a negative bitwise result is converted back to magnitude.
      for (unsigned bits = GMP_NUMB_BITS; bits <= 4 * GMP_NUMB_BITS; ++bits) {
        mpz_set_ui(lhs, 1);
        mpz_mul_2exp(lhs, lhs, bits);
        mpz_sub_ui(lhs, lhs, 1);
        mpz_neg(lhs, lhs);
        mpz_set_si(rhs, -2);
        for (unsigned shared = 0; shared < 4; ++shared) {
          check_pair(op, lhs, rhs, atomic, shared);
          check_pair(op, rhs, lhs, atomic, shared);
        }
      }

      // Both operand positions can reuse the heap allocation, with either
      // sign. AND uses -1 so the result remains a heap integer.
      for (unsigned sign = 0; sign < 2; ++sign) {
        for (unsigned reverse = 0; reverse < 2; ++reverse) {
          mpz_set_ui(lhs, 1);
          mpz_mul_2exp(lhs, lhs, 130);
          mpz_add_ui(lhs, lhs, 17);
          if (sign)
            mpz_neg(lhs, lhs);
          intptr_t small = operation == 2 ? -1 : 3;
          mpz_set_si(rhs, small);
          void *x = from_mpz(lhs, atomic);
          size_t before = allocations;
          void *result = reverse ? op->apply[atomic](rcgmp_encode(small), x)
                                 : op->apply[atomic](x, rcgmp_encode(small));
          CHECK(result == x && allocations == before);
          op->oracle(expected, reverse ? rhs : lhs, reverse ? lhs : rhs);
          check_value(result, expected);
          rcgmp_mpz_release(result, atomic);
        }
      }

      // Two references to the same heap operand, optionally with an external
      // alias. Multiplication exercises mpn_sqr's disjoint output requirement.
      for (unsigned shared = 0; shared < 2; ++shared) {
        void *x = from_mpz(lhs, atomic);
        rcgmp_mpz_retain(x, atomic);
        if (shared)
          rcgmp_mpz_retain(x, atomic);
        void *result = op->apply[atomic](x, x);
        op->oracle(expected, lhs, lhs);
        check_value(result, expected);
        if (shared) {
          check_value(x, lhs);
          rcgmp_mpz_release(x, atomic);
        }
        rcgmp_mpz_release(result, atomic);
      }

      size_t before = allocations;
      void *small = op->apply[atomic](rcgmp_encode(-7), rcgmp_encode(9));
      CHECK(rcgmp_is_immediate(small) && allocations == before);
      rcgmp_mpz_release(small, atomic);
    }
  }
  CHECK(allocations == deallocations);
  gmp_randclear(random);
  mpz_clears(lhs, rhs, expected, NULL);
  return 0;
}
