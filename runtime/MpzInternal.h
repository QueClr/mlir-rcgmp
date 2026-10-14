#ifndef RCGMP_RUNTIME_MPZ_INTERNAL_H
#define RCGMP_RUNTIME_MPZ_INTERNAL_H

#include "RCGMP/Runtime/Mpz.h"
#include <gmp.h>
#include <limits.h>
#include <stdatomic.h>

typedef union {
  uintptr_t local;
  _Atomic(uintptr_t) atomic;
} RCGMPRefCount;

typedef struct {
  RCGMPRefCount refs;
  intptr_t size;
  uintptr_t capacity;
  mp_limb_t limbs[];
} RCGMPMpz;

_Static_assert(sizeof(uintptr_t) == sizeof(mp_limb_t),
               "GMP limb/handle ABI mismatch");
_Static_assert(GMP_NAIL_BITS == 0, "GMP nail bits are unsupported");
_Static_assert(sizeof(RCGMPRefCount) == sizeof(uintptr_t), "RC ABI mismatch");
_Static_assert(_Alignof(RCGMPRefCount) == _Alignof(uintptr_t),
               "RC alignment mismatch");
_Static_assert(offsetof(RCGMPMpz, limbs) == 3 * sizeof(uintptr_t),
               "MPZ ABI mismatch");
_Static_assert(_Alignof(RCGMPRefCount) >= sizeof(uintptr_t),
               "RC requires native-word alignment");
_Static_assert(_Alignof(RCGMPMpz) >= 4,
               "MPZ tagging requires four-byte alignment");

#define RCGMP_IMM_MAX (INTPTR_MAX / 4)
#define RCGMP_IMM_MIN (-RCGMP_IMM_MAX - 1)

static inline int rcgmp_is_immediate(void *value) {
  return ((uintptr_t)value & 1) != 0;
}

static inline void *rcgmp_encode(intptr_t value) {
  return (void *)(((uintptr_t)value << 2) | 1);
}

static inline intptr_t rcgmp_decode(void *value) {
  // Unsigned arithmetic avoids relying on a right shift of a negative value.
  uintptr_t payload = (uintptr_t)value >> 2;
  uintptr_t sign = (uintptr_t)1 << (sizeof(uintptr_t) * CHAR_BIT - 3);
  return (payload & sign) ? -(intptr_t)((sign << 1) - payload)
                          : (intptr_t)payload;
}

RCGMPMpz *rcgmp_mpz_allocate(size_t capacity, uint32_t atomic);
void *rcgmp_mpz_normalize(RCGMPMpz *value, uint32_t atomic);

#endif
