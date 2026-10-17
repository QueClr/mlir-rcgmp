#include "Test.h"
#include <pthread.h>

void *add_local(void *, void *);
void *add_atomic(void *, void *);
void *double_local(void *);
void *double_atomic(void *);

static void *add(void *lhs, void *rhs, uint32_t atomic) {
  return atomic ? add_atomic(lhs, rhs) : add_local(lhs, rhs);
}

static void check_sum(mpz_t lhs, mpz_t rhs, uint32_t atomic, unsigned shared) {
  void *x = from_mpz(lhs, atomic), *y = from_mpz(rhs, atomic);
  if (shared & 1)
    rcgmp_mpz_retain(x, atomic);
  if (shared & 2)
    rcgmp_mpz_retain(y, atomic);
  mpz_t expected;
  mpz_init(expected);
  mpz_add(expected, lhs, rhs);
  void *sum = add(x, y, atomic);
  check_value(sum, expected);
  if (shared & 1) {
    check_value(x, lhs);
    rcgmp_mpz_release(x, atomic);
  }
  if (shared & 2) {
    check_value(y, rhs);
    rcgmp_mpz_release(y, atomic);
  }
  rcgmp_mpz_release(sum, atomic);
  mpz_clear(expected);
}

typedef struct {
  void *shared;
  mpz_t expected;
} ThreadInput;

static void *add_shared(void *argument) {
  ThreadInput *input = argument;
  for (unsigned i = 0; i < 1000; ++i) {
    void *owned = rcgmp_mpz_retain(input->shared, 1);
    void *sum = add_atomic(owned, rcgmp_encode(1));
    check_value(sum, input->expected);
    rcgmp_mpz_release(sum, 1);
  }
  return NULL;
}

int main(void) {
  mpz_t lhs, rhs, expected;
  mpz_inits(lhs, rhs, expected, NULL);
  gmp_randstate_t random;
  gmp_randinit_default(random);
  gmp_randseed_ui(random, 1729);
  intptr_t boundaries[] = {0,
                           1,
                           -1,
                           RCGMP_IMM_MIN,
                           RCGMP_IMM_MAX,
                           RCGMP_IMM_MIN - 1,
                           RCGMP_IMM_MAX + 1};
  for (uint32_t atomic = 0; atomic < 2; ++atomic) {
    for (size_t i = 0; i < sizeof(boundaries) / sizeof(*boundaries); ++i) {
      for (size_t j = 0; j < sizeof(boundaries) / sizeof(*boundaries); ++j) {
        mpz_set_si(lhs, boundaries[i]);
        mpz_set_si(rhs, boundaries[j]);
        for (unsigned shared = 0; shared < 4; ++shared)
          check_sum(lhs, rhs, atomic, shared);
      }
    }
    for (unsigned i = 0; i < 1500; ++i) {
      mpz_urandomb(lhs, random, (i * 73) % 1025);
      mpz_urandomb(rhs, random, (i * 131) % 1025);
      if (i & 1)
        mpz_neg(lhs, lhs);
      if (i & 2)
        mpz_neg(rhs, rhs);
      // Exercise cancellation, normalization, and long carry/borrow chains.
      if (i % 7 == 0) {
        mpz_neg(rhs, lhs);
        mpz_add_ui(rhs, rhs, i % 3);
      }
      for (unsigned shared = 0; shared < 4; ++shared)
        check_sum(lhs, rhs, atomic, shared);
    }

    // Reuse either operand with no temporary promotion of the small operand.
    mpz_set_ui(lhs, 1);
    mpz_mul_2exp(lhs, lhs, 130);
    mpz_add_ui(expected, lhs, 1);
    for (unsigned reverse = 0; reverse < 2; ++reverse) {
      void *x = from_mpz(lhs, atomic);
      size_t before = allocations;
      void *sum = reverse ? add(rcgmp_encode(1), x, atomic)
                          : add(x, rcgmp_encode(1), atomic);
      CHECK(sum == x && allocations == before);
      check_value(sum, expected);
      rcgmp_mpz_release(sum, atomic);
    }

    // A shorter, unique operand can supply storage while the larger is shared.
    mpz_set_ui(lhs, 1);
    mpz_mul_2exp(lhs, lhs, GMP_NUMB_BITS - 1);
    mpz_set_ui(rhs, 1);
    mpz_mul_2exp(rhs, rhs, GMP_NUMB_BITS);
    mpz_neg(rhs, rhs);
    void *x = from_mpz(lhs, atomic), *y = from_mpz(rhs, atomic);
    rcgmp_mpz_retain(y, atomic);
    size_t before = allocations;
    void *sum = add(x, y, atomic);
    CHECK(sum == x && allocations == before);
    mpz_add(expected, lhs, rhs);
    check_value(sum, expected);
    check_value(y, rhs);
    rcgmp_mpz_release(sum, atomic);
    rcgmp_mpz_release(y, atomic);

    // Aliased inputs consume two references and preserve any external alias.
    x = from_mpz(rhs, atomic);
    rcgmp_mpz_retain(x, atomic);
    sum = atomic ? double_atomic(x) : double_local(x);
    mpz_mul_2exp(expected, rhs, 1);
    check_value(sum, expected);
    check_value(x, rhs);
    rcgmp_mpz_release(sum, atomic);
    rcgmp_mpz_release(x, atomic);

    // Carry fills the spare limb; subsequent growth must allocate safely.
    uintptr_t words[] = {UINTPTR_MAX, UINTPTR_MAX};
    x = rcgmp_mpz_from_words(words, 2, 0, atomic);
    y = rcgmp_mpz_from_words(words, 2, 0, atomic);
    mpz_import(expected, 2, -1, sizeof(*words), 0, 0, words);
    mpz_mul_2exp(expected, expected, 1);
    before = allocations;
    sum = add(x, y, atomic);
    CHECK(sum == x && allocations == before);
    check_value(sum, expected);
    before = allocations;
    sum = add(sum, rcgmp_encode(1), atomic);
    CHECK(allocations == before + 1);
    mpz_add_ui(expected, expected, 1);
    check_value(sum, expected);
    rcgmp_mpz_release(sum, atomic);
  }
  size_t before = allocations;
  CHECK(rcgmp_decode(add_local(rcgmp_encode(-7), rcgmp_encode(9))) == 2);
  CHECK(allocations == before);
  // Concurrent readers keep independent references; no thread may mutate
  // shared.
  mpz_set_ui(lhs, 1);
  mpz_mul_2exp(lhs, lhs, 200);
  ThreadInput input = {.shared = from_mpz(lhs, 1)};
  mpz_init(input.expected);
  mpz_add_ui(input.expected, lhs, 1);
  pthread_t threads[4];
  for (unsigned i = 0; i < 4; ++i)
    CHECK(pthread_create(&threads[i], NULL, add_shared, &input) == 0);
  for (unsigned i = 0; i < 4; ++i)
    CHECK(pthread_join(threads[i], NULL) == 0);
  check_value(input.shared, lhs);
  // Once the workers finish, the remaining reference is eligible for reuse.
  void *sum = add_atomic(input.shared, rcgmp_encode(1));
  CHECK(sum == input.shared);
  check_value(sum, input.expected);
  rcgmp_mpz_release(sum, 1);
  mpz_clear(input.expected);
  CHECK(allocations == deallocations);
  gmp_randclear(random);
  mpz_clears(lhs, rhs, expected, NULL);
  return 0;
}
