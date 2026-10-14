#include <stdint.h>
#include <stdio.h>

extern uint32_t checked_add_i8(int32_t lhs, int32_t rhs);

int main(void) {
  for (int32_t lhs = -128; lhs <= 127; ++lhs) {
    for (int32_t rhs = -128; rhs <= 127; ++rhs) {
      int32_t sum = lhs + rhs;
      uint32_t expected = (uint8_t)sum | ((sum < -128 || sum > 127) << 8);
      uint32_t actual = checked_add_i8(lhs, rhs);
      if (actual != expected) {
        fprintf(stderr, "%d + %d: got %u, expected %u\n", lhs, rhs, actual,
                expected);
        return 1;
      }
    }
  }
  return 0;
}
