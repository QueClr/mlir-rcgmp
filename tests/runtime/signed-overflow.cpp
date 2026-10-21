// Compiled i8 arithmetic packs the wrapped result in bits 0-7 and the signed
// overflow flag in bit 8.
#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <limits>

extern "C" {
std::uint32_t checked_add_i8(std::int32_t lhs, std::int32_t rhs);
std::uint32_t checked_sub_i8(std::int32_t lhs, std::int32_t rhs);
std::uint32_t checked_mul_i8(std::int32_t lhs, std::int32_t rhs);
}

namespace {

constexpr std::uint32_t pack(std::int32_t value) {
  bool overflow = value < std::numeric_limits<std::int8_t>::min() ||
                  value > std::numeric_limits<std::int8_t>::max();
  return static_cast<std::uint8_t>(value) | overflow << 8;
}

template <typename Operation>
void checkAllPairs(std::uint32_t (*compiled)(std::int32_t, std::int32_t),
                   Operation operation) {
  for (std::int32_t lhs = -128; lhs <= 127; ++lhs)
    for (std::int32_t rhs = -128; rhs <= 127; ++rhs)
      ASSERT_EQ(compiled(lhs, rhs), pack(operation(lhs, rhs)))
          << "lhs = " << lhs << ", rhs = " << rhs;
}

TEST(SignedOverflow, Addition) { checkAllPairs(checked_add_i8, std::plus{}); }

TEST(SignedOverflow, Subtraction) {
  checkAllPairs(checked_sub_i8, std::minus{});
}

TEST(SignedOverflow, Multiplication) {
  checkAllPairs(checked_mul_i8, std::multiplies{});
}

} // namespace
