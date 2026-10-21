#include <gmp.h>
#include <gtest/gtest.h>

#include <array>

TEST(GmpLink, LimbWidthMatchesHeader) {
  EXPECT_EQ(mp_bits_per_limb, GMP_LIMB_BITS);
}

TEST(GmpLink, AddsWithCarry) {
  std::array<mp_limb_t, 2> lhs{GMP_NUMB_MAX, GMP_NUMB_MAX}, rhs{1, 0}, sum;
  EXPECT_EQ(mpn_add_n(sum.data(), lhs.data(), rhs.data(), 2), 1u);
  EXPECT_EQ(sum, (std::array<mp_limb_t, 2>{0, 0}));
}

TEST(GmpLink, MultipliesAndDivides) {
  mp_limb_t a = 6, b = 7;
  std::array<mp_limb_t, 2> product;
  mpn_mul(product.data(), &a, 1, &b, 1);
  EXPECT_EQ(product, (std::array<mp_limb_t, 2>{42, 0}));
  mp_limb_t quotient, remainder;
  mpn_tdiv_qr(&quotient, &remainder, 0, product.data(), 1, &a, 1);
  EXPECT_EQ(quotient, b);
  EXPECT_EQ(remainder, 0u);
}
