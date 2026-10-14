#include <stdint.h>
#include <stdio.h>

extern uint32_t checked_add_i8(int32_t lhs, int32_t rhs);
extern uint32_t checked_sub_i8(int32_t lhs, int32_t rhs);
extern uint32_t checked_mul_i8(int32_t lhs, int32_t rhs);

int main(void) {
  for (int32_t lhs = -128; lhs <= 127; ++lhs) {
    for (int32_t rhs = -128; rhs <= 127; ++rhs) {
      int32_t values[] = {lhs + rhs, lhs - rhs, lhs * rhs};
      uint32_t actual[] = {checked_add_i8(lhs, rhs), checked_sub_i8(lhs, rhs),
                           checked_mul_i8(lhs, rhs)};
      for (unsigned op = 0; op < 3; ++op) {
        int32_t value = values[op];
        uint32_t expected =
            (uint8_t)value | ((value < -128 || value > 127) << 8);
        if (actual[op] != expected) {
          fprintf(stderr, "%d %c %d: got %u, expected %u\n", lhs, "+-*"[op],
                  rhs, actual[op], expected);
          return 1;
        }
      }
    }
  }
  return 0;
}
