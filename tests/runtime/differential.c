// Compares compiled operations with GMP on boundary and random operands.
// Usage: rcgmp-differential [seed [iterations]]
#include "Test.h"
#include <ctype.h>
#include <errno.h>
#include <string.h>

void *addi_local(void *, void *);
void *addi_atomic(void *, void *);
void *subi_local(void *, void *);
void *subi_atomic(void *, void *);
void *muli_local(void *, void *);
void *muli_atomic(void *, void *);
void *andi_local(void *, void *);
void *andi_atomic(void *, void *);
void *xori_local(void *, void *);
void *xori_atomic(void *, void *);
void *retain_local(void *);
void *retain_atomic(void *);
void release_local(void *);
void release_atomic(void *);

// Compiled reference counting, indexed by atomicity. Operands are built by the
// C runtime, so both implementations operate on the same objects.
static void *(*retain[2])(void *) = {retain_local, retain_atomic};
static void (*release[2])(void *) = {release_local, release_atomic};

// Each operation lists its compiled entry points, indexed by atomicity, and the
// GMP function it must agree with.
typedef struct {
  const char *name;
  void *(*apply[2])(void *, void *);
  void (*oracle)(mpz_ptr, mpz_srcptr, mpz_srcptr);
} BinaryOperation;

static BinaryOperation binaryOperations[] = {
    {"addi", {addi_local, addi_atomic}, mpz_add},
    {"subi", {subi_local, subi_atomic}, mpz_sub},
    {"muli", {muli_local, muli_atomic}, mpz_mul},
    {"andi", {andi_local, andi_atomic}, mpz_and},
    {"xori", {xori_local, xori_atomic}, mpz_xor},
};

// A shared operand has an extra reference held by the caller, which must
// survive with its value. An aliased rhs is a retained reference to the lhs.
enum { SharedLhs = 1, SharedRhs = 2, Aliased = 4 };
typedef struct {
  unsigned flags;
  const char *name;
} OwnershipMode;

static OwnershipMode pairModes[] = {
    {0, "unique operands"},
    {SharedLhs, "shared lhs"},
    {SharedRhs, "shared rhs"},
    {SharedLhs | SharedRhs, "shared operands"},
};
static OwnershipMode aliasModes[] = {
    {Aliased, "aliased operands"},
    {Aliased | SharedLhs, "aliased shared operand"},
};

static unsigned long seed = 31415;
static unsigned long iterations = 20000;

// Where the run is, so that a failure can be located without a rerun: a phase
// and, within a random stream, the one-based position and the stream length.
static const char *phase;
static unsigned long position, positionCount;

static void print_position(void) {
  if (positionCount)
    fprintf(stderr, "  %s %lu of %lu\n", phase, position, positionCount);
  else
    fprintf(stderr, "  %s\n", phase);
  fprintf(stderr, "  seed = %lu, iterations = %lu\n", seed, iterations);
}

// A heap operand is allocated with spare limbs beyond its magnitude, as results
// of earlier operations often are, so that an operation may reuse it in place.
// Like the stale limbs of such results, the spare limbs are nonzero, so reading
// past the magnitude changes the value.
typedef struct {
  mpz_srcptr value;
  size_t spare;
} Operand;

typedef struct {
  BinaryOperation *operation;
  uint32_t atomic;
  OwnershipMode *mode;
  Operand lhs, rhs;
  // Each heap operand has room for the result, so a unique one must hold it.
  int roomy;
} BinaryCase;

static int fits_immediate(mpz_srcptr value) {
  return mpz_size(value) <= 1 &&
         mpz_getlimbn(value, 0) <=
             (uintptr_t)RCGMP_IMM_MAX + (mpz_sgn(value) < 0);
}

static void print_operand(const char *name, Operand operand) {
  if (fits_immediate(operand.value))
    gmp_fprintf(stderr, "  %s = %#Zx (immediate)\n", name, operand.value);
  else
    gmp_fprintf(stderr, "  %s = %#Zx (%zu spare limbs)\n", name, operand.value,
                operand.spare);
}

static void fail(BinaryCase *test, const char *problem) {
  fprintf(stderr, "%s (%s, %s): %s\n", test->operation->name,
          test->atomic ? "atomic" : "local", test->mode->name, problem);
  print_operand("lhs", test->lhs);
  print_operand("rhs", test->rhs);
  print_position();
  abort();
}

static void *make_operand(Operand operand, uint32_t atomic) {
  void *handle = from_mpz(operand.value, atomic);
  if (rcgmp_is_immediate(handle))
    return handle;
  RCGMPMpz *exact = handle;
  size_t count = mpz_size(operand.value);
  RCGMPMpz *padded = rcgmp_mpz_allocate(count + operand.spare, atomic);
  memcpy(padded->limbs, exact->limbs, count * sizeof(mp_limb_t));
  for (size_t i = 0; i < operand.spare; ++i)
    padded->limbs[count + i] =
        i % 2 ? GMP_NUMB_MAX / 0xff * 0xa5 : GMP_NUMB_MAX;
  padded->size = exact->size;
  rcgmp_mpz_release(exact, atomic);
  return padded;
}

static uintptr_t references(void *value, uint32_t atomic) {
  RCGMPMpz *heap = value;
  return atomic ? atomic_load(&heap->refs.atomic) : heap->refs.local;
}

// Each operand position holding operand consumed one of its references, and the
// caller's reference kept its value. The result may be another reference to it.
static void release_shared(BinaryCase *test, void *operand, mpz_srcptr value,
                           void *result) {
  if (!represents(operand, value))
    fail(test, "shared operand was modified");
  if (!rcgmp_is_immediate(operand) &&
      references(operand, test->atomic) != 1 + (operand == result))
    fail(test, "operand positions did not each consume one reference");
  release[test->atomic](operand);
}

static void retain_operand(BinaryCase *test, void *operand) {
  if (retain[test->atomic](operand) != operand)
    fail(test, "retain returned a different handle");
}

static void check_binary(BinaryCase *test) {
  uint32_t atomic = test->atomic;
  unsigned flags = test->mode->flags;
  mpz_srcptr lhs = test->lhs.value, rhs = test->rhs.value;
  void *x = make_operand(test->lhs, atomic);
  void *y = x;
  if (flags & Aliased)
    retain_operand(test, x);
  else
    y = make_operand(test->rhs, atomic);
  if (!represents(x, lhs) || !represents(y, rhs))
    fail(test, "operand construction is not canonical");
  if (flags & SharedLhs)
    retain_operand(test, x);
  if (flags & SharedRhs)
    retain_operand(test, y);
  mpz_t expected;
  mpz_init(expected);
  test->operation->oracle(expected, lhs, rhs);
  size_t before = allocations;
  void *result = test->operation->apply[atomic](x, y);
  if (!represents(result, expected))
    fail(test, "result differs from GMP or is not canonical");
  // General multiplication cannot write over its operands.
  int reusable = test->operation->oracle != mpz_mul || mpz_size(lhs) <= 1 ||
                 mpz_size(rhs) <= 1;
  if (test->roomy && !flags && reusable && !rcgmp_is_immediate(result) &&
      (!rcgmp_is_immediate(x) || !rcgmp_is_immediate(y)) &&
      allocations != before)
    fail(test, "result did not reuse a unique operand");
  if (flags & SharedLhs)
    release_shared(test, x, lhs, result);
  if (flags & SharedRhs)
    release_shared(test, y, rhs, result);
  release[atomic](result);
  mpz_clear(expected);
  if (allocations != deallocations)
    fail(test, "allocations and deallocations do not balance");
}

static void check_pair(BinaryOperation *operation, uint32_t atomic, Operand lhs,
                       Operand rhs, int roomy) {
  for (size_t mode = 0; mode < sizeof(pairModes) / sizeof(*pairModes); ++mode) {
    BinaryCase test = {operation, atomic, &pairModes[mode], lhs, rhs, roomy};
    check_binary(&test);
  }
}

static void check_aliased(BinaryOperation *operation, uint32_t atomic,
                          Operand operand) {
  for (size_t mode = 0; mode < sizeof(aliasModes) / sizeof(*aliasModes);
       ++mode) {
    BinaryCase test = {operation, atomic,  &aliasModes[mode],
                       operand,   operand, 0};
    check_binary(&test);
  }
}

static void add_small(mpz_t value, long delta) {
  if (delta < 0)
    mpz_sub_ui(value, value, -delta);
  else
    mpz_add_ui(value, value, delta);
}

static long random_delta(gmp_randstate_t random, long bound) {
  return (long)gmp_urandomm_ui(random, 2 * bound + 1) - bound;
}

static void power_of_two(mpz_t value, unsigned exponent) {
  mpz_set_ui(value, 0);
  mpz_setbit(value, exponent);
}

static void random_operand(mpz_t value, gmp_randstate_t random) {
  unsigned word = GMP_NUMB_BITS;
  // Square roots of the immediate bound, so that products land near it; the
  // immediate bound; the signed word bound; and one to four full limbs.
  unsigned edges[] = {(word - 3) / 2, (word - 2) / 2, word - 3, word - 1,
                      word,           2 * word,       3 * word, 4 * word};
  switch (gmp_urandomm_ui(random, 4)) {
  case 0:
    power_of_two(
        value, edges[gmp_urandomm_ui(random, sizeof(edges) / sizeof(*edges))]);
    add_small(value, random_delta(random, 4));
    break;
  case 1:
    mpz_set_si(value, random_delta(random, 4));
    break;
  case 2:
    // Long runs of ones and zeros stress carries, borrows, and sign extension.
    mpz_rrandomb(value, random, 1 + gmp_urandomm_ui(random, 5 * word));
    break;
  default:
    mpz_urandomb(value, random, gmp_urandomm_ui(random, 5 * word + 1));
    break;
  }
  if (gmp_urandomb_ui(random, 1))
    mpz_neg(value, value);
}

// Derive rhs from lhs to reach cancellation, complementary bit patterns, and
// products at either immediate bound, which independent operands rarely hit.
static void related_operand(mpz_t rhs, mpz_srcptr lhs, gmp_randstate_t random) {
  switch (gmp_urandomm_ui(random, 3)) {
  case 0:
    mpz_set(rhs, lhs);
    if (gmp_urandomb_ui(random, 1))
      mpz_neg(rhs, rhs);
    add_small(rhs, random_delta(random, 2));
    break;
  case 1:
    mpz_com(rhs, lhs);
    break;
  default:
    if (!mpz_sgn(lhs)) {
      random_operand(rhs, random);
      break;
    }
    power_of_two(rhs, GMP_NUMB_BITS - 3);
    mpz_tdiv_q(rhs, rhs, lhs);
    if (gmp_urandomb_ui(random, 1))
      mpz_neg(rhs, rhs);
    add_small(rhs, random_delta(random, 1));
    break;
  }
}

// With a one-limb operand, a result needs at most one limb more than the other
// operand, so a one-limb heap operand can hold it in place when its spare limbs
// cover the other operand. Random operands have at most five limbs and derived
// ones at most six, so spares range up to six.
static size_t random_spare(gmp_randstate_t random) {
  return gmp_urandomm_ui(random, 7);
}

// Each operation and atomicity draws from its own stream, keyed by the seed and
// the operation name, so that a larger iteration count extends every stream
// and adding an operation leaves the others unchanged.
static void seed_stream(gmp_randstate_t random, BinaryOperation *operation,
                        uint32_t atomic) {
  mpz_t key;
  mpz_init(key);
  mpz_import(key, strlen(operation->name), 1, 1, 0, 0, operation->name);
  mpz_mul_2exp(key, key, 1);
  mpz_add_ui(key, key, atomic);
  mpz_mul_2exp(key, key, sizeof(seed) * CHAR_BIT);
  mpz_add_ui(key, key, seed);
  gmp_randseed(random, key);
  mpz_clear(key);
}

// Appends 2^exponent + delta and its negation.
static void add_boundary_pair(mpz_t *boundaries, size_t *count,
                              unsigned exponent, long delta) {
  mpz_ptr positive = boundaries[(*count)++];
  mpz_ptr negative = boundaries[(*count)++];
  mpz_init(positive);
  power_of_two(positive, exponent);
  add_small(positive, delta);
  mpz_init(negative);
  mpz_neg(negative, positive);
}

// Accepts a whole unsigned number in C notation. strtoul alone would accept a
// prefix of the text or wrap a negative number around.
static int parse_count(const char *text, unsigned long *value) {
  if (!isdigit((unsigned char)*text))
    return 0;
  char *end;
  errno = 0;
  *value = strtoul(text, &end, 0);
  return !*end && errno != ERANGE;
}

int main(int argc, char **argv) {
  if (argc > 3 || (argc > 1 && !parse_count(argv[1], &seed)) ||
      (argc > 2 && !parse_count(argv[2], &iterations))) {
    fputs("usage: rcgmp-differential [seed [iterations]]\n", stderr);
    return 2;
  }
  gmp_randstate_t random;
  gmp_randinit_default(random);

  // Zero, one, both sides of the immediate bounds, and one or two full limbs;
  // the square roots of the immediate bound, whose products land exactly on
  // it; and the signed word bound.
  mpz_t boundaries[21];
  size_t boundaryCount = 0;
  mpz_init_set_si(boundaries[boundaryCount++], 0);
  mpz_init_set_si(boundaries[boundaryCount++], 1);
  mpz_init_set_si(boundaries[boundaryCount++], -1);
  unsigned edges[] = {GMP_NUMB_BITS - 3, GMP_NUMB_BITS};
  for (size_t edge = 0; edge < sizeof(edges) / sizeof(*edges); ++edge)
    for (long delta = -1; delta <= 1; ++delta)
      add_boundary_pair(boundaries, &boundaryCount, edges[edge], delta);
  unsigned powers[] = {(GMP_NUMB_BITS - 3) / 2, (GMP_NUMB_BITS - 2) / 2,
                       GMP_NUMB_BITS - 1};
  for (size_t power = 0; power < sizeof(powers) / sizeof(*powers); ++power)
    add_boundary_pair(boundaries, &boundaryCount, powers[power], 0);
  // No spare limbs, so that every growing result needs a new allocation, and
  // enough for any result of two boundary operands to fit in either one.
  size_t boundarySpares[] = {0, 4};

  mpz_t lhs, rhs;
  mpz_inits(lhs, rhs, NULL);
  for (size_t index = 0;
       index < sizeof(binaryOperations) / sizeof(*binaryOperations); ++index) {
    BinaryOperation *operation = &binaryOperations[index];
    for (uint32_t atomic = 0; atomic < 2; ++atomic) {
      phase = "boundary sweep";
      positionCount = 0;
      for (size_t spare = 0; spare < 2; ++spare) {
        for (size_t i = 0; i < boundaryCount; ++i) {
          Operand x = {boundaries[i], boundarySpares[spare]};
          for (size_t j = 0; j < boundaryCount; ++j) {
            Operand y = {boundaries[j], boundarySpares[spare]};
            check_pair(operation, atomic, x, y, boundarySpares[spare] > 0);
          }
          check_aliased(operation, atomic, x);
        }
      }
      seed_stream(random, operation, atomic);
      phase = "random iteration";
      positionCount = iterations;
      for (unsigned long i = 0; i < iterations; ++i) {
        position = i + 1;
        random_operand(lhs, random);
        if (gmp_urandomm_ui(random, 4) == 0)
          related_operand(rhs, lhs, random);
        else
          random_operand(rhs, random);
        Operand x = {lhs, random_spare(random)};
        Operand y = {rhs, random_spare(random)};
        check_pair(operation, atomic, x, y, 0);
        check_aliased(operation, atomic, x);
      }
    }
  }

  for (size_t i = 0; i < boundaryCount; ++i)
    mpz_clear(boundaries[i]);
  mpz_clears(lhs, rhs, NULL);
  gmp_randclear(random);
  return 0;
}
