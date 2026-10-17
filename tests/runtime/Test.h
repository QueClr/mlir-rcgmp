#ifndef RCGMP_TEST_RUNTIME_H
#define RCGMP_TEST_RUNTIME_H

#include "MpzInternal.h"
#include "RCGMP/Runtime/Allocation.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition);          \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#ifdef RCGMP_TEST_ALLOCATOR
static _Atomic(size_t) allocations, deallocations;
typedef union {
  max_align_t align;
  struct {
    size_t bytes, alignment;
  } info;
} AllocationHeader;

void *rcgmp_allocate(size_t bytes, size_t alignment) {
  CHECK(alignment <= _Alignof(max_align_t));
  AllocationHeader *header = malloc(sizeof(*header) + bytes);
  CHECK(header);
  header->info.bytes = bytes;
  header->info.alignment = alignment;
  ++allocations;
  return header + 1;
}
void rcgmp_dealloc(void *pointer, size_t bytes, size_t alignment) {
  AllocationHeader *header = (AllocationHeader *)pointer - 1;
  CHECK(header->info.bytes == bytes);
  CHECK(header->info.alignment == alignment);
  ++deallocations;
  free(header);
}
#endif

// Whether value is the canonical representation of expected: the immediate
// 4*n + 1 exactly when expected is an n in the immediate range, and otherwise a
// heap integer without leading zero limbs.
static inline int represents(void *value, mpz_srcptr expected) {
  if (rcgmp_is_immediate(value) && ((uintptr_t)value & 3) != 1)
    return 0;
  mpz_t actual;
  mpz_init(actual);
  if (rcgmp_is_immediate(value)) {
    intptr_t integer = rcgmp_decode(value);
    uintptr_t magnitude =
        integer < 0 ? (uintptr_t)-integer : (uintptr_t)integer;
    mpz_import(actual, 1, -1, sizeof(magnitude), 0, 0, &magnitude);
    if (integer < 0)
      mpz_neg(actual, actual);
  } else {
    RCGMPMpz *heap = value;
    int negative = heap->size < 0;
    size_t count = negative ? -heap->size : heap->size;
    if (!count || count > heap->capacity || !heap->limbs[count - 1] ||
        (count == 1 && heap->limbs[0] <= (uintptr_t)RCGMP_IMM_MAX + negative)) {
      mpz_clear(actual);
      return 0;
    }
    mpz_import(actual, count, -1, sizeof(mp_limb_t), 0, 0, heap->limbs);
    if (negative)
      mpz_neg(actual, actual);
  }
  int equal = mpz_cmp(actual, expected) == 0;
  mpz_clear(actual);
  return equal;
}

static inline void check_value(void *value, mpz_srcptr expected) {
  CHECK(represents(value, expected));
}

static inline void *from_mpz(mpz_srcptr value, uint32_t atomic) {
  uintptr_t words[32];
  CHECK(mpz_sizeinbase(value, 2) <= sizeof(words) * CHAR_BIT);
  size_t count;
  mpz_export(words, &count, -1, sizeof(*words), 0, 0, value);
  return rcgmp_mpz_from_words(words, count, mpz_sgn(value) < 0, atomic);
}
#endif
