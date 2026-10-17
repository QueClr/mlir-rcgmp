#include "Test.h"

void *from_signed(int64_t);
void *from_unsigned(uint64_t);
void *from_atomic(int64_t);
void *retain_local(void *);
void *retain_atomic(void *);
void release_local(void *);
void release_atomic(void *);
void *from_min129(void);
void *from_max129(void);
void *from_signed_i1(void);
void *from_unsigned_i1(void);

int main(void) {
  mpz_t expected;
  mpz_init(expected);
  int64_t cases[] = {0,
                     1,
                     -1,
                     RCGMP_IMM_MAX,
                     RCGMP_IMM_MIN,
                     (int64_t)RCGMP_IMM_MAX + 1,
                     (int64_t)RCGMP_IMM_MIN - 1,
                     INT64_MAX,
                     INT64_MIN};
  for (uint32_t atomic = 0; atomic < 2; ++atomic) {
    for (size_t i = 0; i < sizeof(cases) / sizeof(*cases); ++i) {
      int64_t n = cases[i];
      uint64_t magnitude = n < 0 ? -(uint64_t)n : (uint64_t)n;
      mpz_import(expected, 1, -1, sizeof(magnitude), 0, 0, &magnitude);
      if (n < 0)
        mpz_neg(expected, expected);
      void *x = atomic ? from_atomic(n) : from_signed(n);
      CHECK(rcgmp_is_immediate(x) ==
            (n >= RCGMP_IMM_MIN && n <= RCGMP_IMM_MAX));
      check_value(x, expected);
      void *alias = atomic ? retain_atomic(x) : retain_local(x);
      CHECK(alias == x);
      // Interoperate between generated RC code and the C runtime helpers.
      rcgmp_mpz_release(x, atomic);
      check_value(alias, expected);
      rcgmp_mpz_retain(alias, atomic);
      if (atomic) {
        release_atomic(alias);
        release_atomic(alias);
      } else {
        release_local(alias);
        release_local(alias);
      }
    }
  }
  uint64_t max = UINT64_MAX;
  mpz_import(expected, 1, -1, sizeof(max), 0, 0, &max);
  void *x = from_unsigned(max);
  check_value(x, expected);
  release_local(x);
  x = from_min129();
  mpz_set_ui(expected, 1);
  mpz_mul_2exp(expected, expected, 128);
  mpz_neg(expected, expected);
  check_value(x, expected);
  release_local(x);
  x = from_max129();
  mpz_set_ui(expected, 1);
  mpz_mul_2exp(expected, expected, 129);
  mpz_sub_ui(expected, expected, 1);
  check_value(x, expected);
  release_local(x);
  CHECK(rcgmp_decode(from_signed_i1()) == -1);
  CHECK(rcgmp_decode(from_unsigned_i1()) == 1);
  // Zero is the immediate 0 regardless of the sign flag or zero words.
  uintptr_t zeros[] = {0, 0};
  CHECK((uintptr_t)rcgmp_mpz_from_words(zeros, 0, 1, 0) == 1);
  CHECK((uintptr_t)rcgmp_mpz_from_words(zeros, 2, 1, 1) == 1);
#ifdef RCGMP_TEST_ALLOCATOR
  CHECK(allocations > 0 && allocations == deallocations);
  size_t before = allocations;
  x = from_signed(123);
  release_local(retain_local(x));
  release_local(x);
  CHECK(allocations == before);
#endif
  mpz_clear(expected);
  return 0;
}
