#include <gmp.h>

int main(void) {
  if (mp_bits_per_limb != GMP_LIMB_BITS)
    return 1;

  mp_limb_t lhs[2] = {GMP_NUMB_MAX, GMP_NUMB_MAX};
  mp_limb_t rhs[2] = {1, 0};
  mp_limb_t sum[2];
  if (mpn_add_n(sum, lhs, rhs, 2) != 1 || sum[0] != 0 || sum[1] != 0)
    return 2;

  mp_limb_t a = 6, b = 7;
  mp_limb_t product[2];
  mpn_mul(product, &a, 1, &b, 1);
  if (product[0] != 42 || product[1] != 0)
    return 3;

  mp_limb_t quotient, remainder;
  mpn_tdiv_qr(&quotient, &remainder, 0, product, 1, &a, 1);
  return quotient == b && remainder == 0 ? 0 : 4;
}
