#ifndef RCGMP_RUNTIME_MPZ_H
#define RCGMP_RUNTIME_MPZ_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Magnitude words are least-significant first. Zero words are permitted.
// negative and atomic must be 0 or 1. The result owns one reference.
void *rcgmp_mpz_from_words(const uintptr_t *words, size_t count,
                           uint32_t negative, uint32_t atomic);
void *rcgmp_mpz_retain(void *value, uint32_t atomic);
void rcgmp_mpz_release(void *value, uint32_t atomic);

// Only for a heap value whose final decrement has already completed.
void rcgmp_mpz_free(void *value, uint32_t atomic);

#ifdef __cplusplus
}
#endif

#endif
