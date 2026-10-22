#ifndef RCGMP_RUNTIME_MPZ_H
#define RCGMP_RUNTIME_MPZ_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Every handle passed to or returned by these functions is canonical, as
// defined by the !rcgmp.mpz type: an immediate exactly when its value fits the
// immediate range, and otherwise a heap integer without leading zero limbs.

// Magnitude words are least-significant first. Zero words are permitted.
// negative and atomic must be 0 or 1. The result owns one reference.
void *rcgmp_mpz_from_words(const uintptr_t *words, size_t count,
                           uint32_t negative, uint32_t atomic);
// Heap objects with a zero reference count are immortal: retain and release
// leave them unchanged, and they are never reused in place.
void *rcgmp_mpz_retain(void *value, uint32_t atomic);
void rcgmp_mpz_release(void *value, uint32_t atomic);

// Consume one reference per operand position and return one owned sum.
// reuse bit 0/1 permits reusing lhs/rhs respectively; set a bit only for a
// heap operand proven uniquely owned. Both operands must match the entry point.
void *rcgmp_mpz_add(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_add_atomic(void *lhs, void *rhs, uint32_t reuse);

// The same ownership and reuse contract applies to all binary operations.
// Subtraction and multiplication are exact. Bitwise operations use infinite
// two's-complement semantics.
void *rcgmp_mpz_sub(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_sub_atomic(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_mul(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_mul_atomic(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_and(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_and_atomic(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_xor(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_xor_atomic(void *lhs, void *rhs, uint32_t reuse);

// Only for a heap value whose final decrement has already completed.
void rcgmp_mpz_free(void *value, uint32_t atomic);

#ifdef __cplusplus
}
#endif

#endif
