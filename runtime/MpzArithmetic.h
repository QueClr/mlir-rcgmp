#ifndef RCGMP_RUNTIME_MPZ_ARITHMETIC_H
#define RCGMP_RUNTIME_MPZ_ARITHMETIC_H

#include "MpzInternal.h"
#include <stdlib.h>

// Immediate magnitudes live on the stack; mixed operations never box them.
typedef struct {
  mp_srcptr limbs;
  size_t count;
  int negative;
  mp_limb_t small;
} Magnitude;

static inline void magnitude(Magnitude *view, void *pointer) {
  if (rcgmp_is_immediate(pointer)) {
    intptr_t value = rcgmp_decode(pointer);
    view->negative = value < 0;
    view->small = value < 0 ? (uintptr_t)-value : (uintptr_t)value;
    view->limbs = &view->small;
    view->count = value != 0;
  } else {
    RCGMPMpz *value = pointer;
    view->negative = value->size < 0;
    view->count = view->negative ? -value->size : value->size;
    view->limbs = value->limbs;
  }
}

static inline RCGMPMpz *destination(void *lhs, void *rhs, uint32_t reuse,
                                    size_t required, uint32_t atomic) {
  if ((reuse & 1) && ((RCGMPMpz *)lhs)->capacity >= required)
    return lhs;
  if ((reuse & 2) && ((RCGMPMpz *)rhs)->capacity >= required)
    return rhs;
  // Geometric headroom amortizes repeated growth. Allocation checks the bound.
  size_t extra = required / 2 + 1;
  if (required > SIZE_MAX - extra)
    abort();
  return rcgmp_mpz_allocate(required + extra, atomic);
}

// Consume inputs after all source limbs have been read, then canonicalize.
static inline void *finish(RCGMPMpz *result, void *lhs, void *rhs,
                           uint32_t atomic) {
  if (result != lhs)
    rcgmp_mpz_release(lhs, atomic);
  if (result != rhs)
    rcgmp_mpz_release(rhs, atomic);
  return rcgmp_mpz_normalize(result, atomic);
}

#endif
