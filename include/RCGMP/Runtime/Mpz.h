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
// The same for a magnitude of one word.
void *rcgmp_mpz_from_word(uintptr_t magnitude, uint32_t negative,
                          uint32_t atomic);
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
// Subtraction and multiplication are exact, and monus is the difference when
// positive and zero otherwise. Bitwise operations use infinite two's-complement
// semantics.
void *rcgmp_mpz_sub(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_sub_atomic(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_monus(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_monus_atomic(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_mul(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_mul_atomic(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_and(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_and_atomic(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_or(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_or_atomic(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_xor(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_xor_atomic(void *lhs, void *rhs, uint32_t reuse);

// Consume one reference and return one owned result. reuse bit 0 permits
// reusing the operand; set it only for a heap operand proven uniquely owned.
// The complement of x is -x - 1, as in infinite two's complement.
void *rcgmp_mpz_not(void *value, uint32_t reuse);
void *rcgmp_mpz_not_atomic(void *value, uint32_t reuse);
void *rcgmp_mpz_neg(void *value, uint32_t reuse);
void *rcgmp_mpz_neg_atomic(void *value, uint32_t reuse);
void *rcgmp_mpz_abs(void *value, uint32_t reuse);
void *rcgmp_mpz_abs_atomic(void *value, uint32_t reuse);

// Shift value by amount bits, with the same ownership and reuse contract as
// the unary operations. Shifting right rounds toward negative infinity.
void *rcgmp_mpz_shl(void *value, uintptr_t amount, uint32_t reuse);
void *rcgmp_mpz_shl_atomic(void *value, uintptr_t amount, uint32_t reuse);
void *rcgmp_mpz_shr(void *value, uintptr_t amount, uint32_t reuse);
void *rcgmp_mpz_shr_atomic(void *value, uintptr_t amount, uint32_t reuse);

// Divide lhs by rhs, with the same ownership and reuse contract as the binary
// operations. tdiv rounds the quotient toward zero, giving the remainder the
// sign of lhs, fdiv rounds it toward negative infinity, giving the remainder
// the sign of rhs, and ediv rounds it so that the remainder is nonnegative. A
// zero divisor terminates the program, except that the _lean entry points
// give the quotient zero and the remainder lhs, as Lean does.
void *rcgmp_mpz_tdiv_q(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_tdiv_q_atomic(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_tdiv_r(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_tdiv_r_atomic(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_fdiv_q(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_fdiv_q_atomic(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_fdiv_r(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_fdiv_r_atomic(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_ediv_q(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_ediv_q_atomic(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_ediv_r(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_ediv_r_atomic(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_tdiv_q_lean(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_tdiv_q_lean_atomic(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_tdiv_r_lean(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_tdiv_r_lean_atomic(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_fdiv_q_lean(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_fdiv_q_lean_atomic(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_fdiv_r_lean(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_fdiv_r_lean_atomic(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_ediv_q_lean(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_ediv_q_lean_atomic(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_ediv_r_lean(void *lhs, void *rhs, uint32_t reuse);
void *rcgmp_mpz_ediv_r_lean_atomic(void *lhs, void *rhs, uint32_t reuse);

// Both results of a division, each owning one reference.
typedef struct {
  void *quotient;
  void *remainder;
} rcgmp_mpz_qr;

rcgmp_mpz_qr rcgmp_mpz_tdiv_qr(void *lhs, void *rhs, uint32_t reuse);
rcgmp_mpz_qr rcgmp_mpz_tdiv_qr_atomic(void *lhs, void *rhs, uint32_t reuse);
rcgmp_mpz_qr rcgmp_mpz_fdiv_qr(void *lhs, void *rhs, uint32_t reuse);
rcgmp_mpz_qr rcgmp_mpz_fdiv_qr_atomic(void *lhs, void *rhs, uint32_t reuse);
rcgmp_mpz_qr rcgmp_mpz_ediv_qr(void *lhs, void *rhs, uint32_t reuse);
rcgmp_mpz_qr rcgmp_mpz_ediv_qr_atomic(void *lhs, void *rhs, uint32_t reuse);
rcgmp_mpz_qr rcgmp_mpz_tdiv_qr_lean(void *lhs, void *rhs, uint32_t reuse);
rcgmp_mpz_qr rcgmp_mpz_tdiv_qr_lean_atomic(void *lhs, void *rhs,
                                           uint32_t reuse);
rcgmp_mpz_qr rcgmp_mpz_fdiv_qr_lean(void *lhs, void *rhs, uint32_t reuse);
rcgmp_mpz_qr rcgmp_mpz_fdiv_qr_lean_atomic(void *lhs, void *rhs,
                                           uint32_t reuse);
rcgmp_mpz_qr rcgmp_mpz_ediv_qr_lean(void *lhs, void *rhs, uint32_t reuse);
rcgmp_mpz_qr rcgmp_mpz_ediv_qr_lean_atomic(void *lhs, void *rhs,
                                           uint32_t reuse);

// Borrow both operands and return a negative, zero, or positive value as lhs is
// less than, equal to, or greater than rhs. Reference counts are unchanged.
// Both entry points share one implementation.
int32_t rcgmp_mpz_cmp(void *lhs, void *rhs);
int32_t rcgmp_mpz_cmp_atomic(void *lhs, void *rhs);

// Only for a heap value whose final decrement has already completed.
void rcgmp_mpz_free(void *value, uint32_t atomic);

#ifdef __cplusplus
}
#endif

#endif
