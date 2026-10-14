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
static size_t allocations, deallocations;
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

static void check_value(void *value, mpz_t expected) {
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
    size_t count = heap->size < 0 ? -heap->size : heap->size;
    CHECK(count && count <= heap->capacity && heap->limbs[count - 1]);
    mpz_import(actual, count, -1, sizeof(mp_limb_t), 0, 0, heap->limbs);
    if (heap->size < 0)
      mpz_neg(actual, actual);
  }
  CHECK(mpz_cmp(actual, expected) == 0);
  mpz_clear(actual);
}
#endif
