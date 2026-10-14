#include "MpzInternal.h"
#include "RCGMP/Runtime/Allocation.h"

#include <stdlib.h>

RCGMPMpz *rcgmp_mpz_allocate(size_t capacity, uint32_t atomic) {
  if (capacity > INTPTR_MAX ||
      capacity > (SIZE_MAX - offsetof(RCGMPMpz, limbs)) / sizeof(mp_limb_t))
    abort();
  size_t bytes = offsetof(RCGMPMpz, limbs) + capacity * sizeof(mp_limb_t);
  RCGMPMpz *value = rcgmp_allocate(bytes, _Alignof(RCGMPMpz));
  if (!value)
    abort();
  if (atomic)
    atomic_init(&value->refs.atomic, 1);
  else
    value->refs.local = 1;
  value->size = 0;
  value->capacity = capacity;
  return value;
}

void rcgmp_mpz_free(void *pointer, uint32_t atomic) {
  if (atomic)
    atomic_thread_fence(memory_order_acquire);
  RCGMPMpz *value = pointer;
  size_t bytes =
      offsetof(RCGMPMpz, limbs) + value->capacity * sizeof(mp_limb_t);
  rcgmp_dealloc(value, bytes, _Alignof(RCGMPMpz));
}

void *rcgmp_mpz_retain(void *pointer, uint32_t atomic) {
  if (!rcgmp_is_immediate(pointer)) {
    RCGMPMpz *value = pointer;
    uintptr_t old = atomic ? atomic_fetch_add_explicit(&value->refs.atomic, 1,
                                                       memory_order_relaxed)
                           : value->refs.local++;
    if (old == UINTPTR_MAX)
      abort();
  }
  return pointer;
}

void rcgmp_mpz_release(void *pointer, uint32_t atomic) {
  if (!rcgmp_is_immediate(pointer)) {
    RCGMPMpz *value = pointer;
    uintptr_t old = atomic ? atomic_fetch_sub_explicit(&value->refs.atomic, 1,
                                                       memory_order_release)
                           : value->refs.local--;
    if (old == 1)
      rcgmp_mpz_free(pointer, atomic);
  }
}

void *rcgmp_mpz_from_words(const uintptr_t *words, size_t count,
                           uint32_t negative, uint32_t atomic) {
  while (count && words[count - 1] == 0)
    --count;
  if (count == 0)
    return rcgmp_encode(0);
  if (count == 1 && words[0] <= (uintptr_t)RCGMP_IMM_MAX + negative) {
    intptr_t small = (intptr_t)words[0];
    return rcgmp_encode(negative ? -small : small);
  }
  // One spare limb allows the first growing addition to reuse this allocation.
  if (count >= (size_t)INTPTR_MAX)
    abort();
  RCGMPMpz *value = rcgmp_mpz_allocate(count + 1, atomic);
  for (size_t i = 0; i < count; ++i)
    value->limbs[i] = words[i];
  value->size = negative ? -(intptr_t)count : (intptr_t)count;
  return value;
}

void *rcgmp_mpz_normalize(RCGMPMpz *value, uint32_t atomic) {
  int negative = value->size < 0;
  size_t count = negative ? (size_t)-value->size : (size_t)value->size;
  while (count && value->limbs[count - 1] == 0)
    --count;
  if (count == 0 ||
      (count == 1 && value->limbs[0] <= (uintptr_t)RCGMP_IMM_MAX + negative)) {
    intptr_t small = count ? (intptr_t)value->limbs[0] : 0;
    rcgmp_mpz_free(value, atomic);
    return rcgmp_encode(negative ? -small : small);
  }
  value->size = negative ? -(intptr_t)count : (intptr_t)count;
  return value;
}
