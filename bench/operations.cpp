// Times compiled operations on the input classes that select their code
// paths, next to the GMP calls that compute the same results. A shared operand
// is retained before an operation consumes it, so the result needs new
// storage; GMP then initializes and clears a result integer, as value
// semantics require. A unique operand is updated in place by a pair of
// operations that restores its value, as GMP updates its own operand.
#include "RCGMP/Runtime/Mpz.h"
#include <benchmark/benchmark.h>
#include <gmp.h>

#include <array>
#include <cstdint>
#include <deque>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

extern "C" {
#define RCGMP_ENTRY(result, name, ...)                                         \
  result name(__VA_ARGS__);                                                    \
  result name##_atomic(__VA_ARGS__);
RCGMP_ENTRY(void *, addi, void *, void *)
RCGMP_ENTRY(void *, subi, void *, void *)
RCGMP_ENTRY(void *, monusi, void *, void *)
RCGMP_ENTRY(void *, muli, void *, void *)
RCGMP_ENTRY(void *, andi, void *, void *)
RCGMP_ENTRY(void *, ori, void *, void *)
RCGMP_ENTRY(void *, xori, void *, void *)
RCGMP_ENTRY(void *, noti, void *)
RCGMP_ENTRY(void *, negi, void *)
RCGMP_ENTRY(void *, absi, void *)
RCGMP_ENTRY(void *, shli, void *, std::int64_t)
RCGMP_ENTRY(void *, shrsi, void *, std::int64_t)
RCGMP_ENTRY(void *, floormod2expi, void *, std::int64_t)
RCGMP_ENTRY(void *, ipowi, void *, std::int64_t)
RCGMP_ENTRY(std::int32_t, cmpi, void *, void *)
RCGMP_ENTRY(void *, extsi, std::int64_t)
RCGMP_ENTRY(void *, retain, void *)
RCGMP_ENTRY(void, release, void *)
#undef RCGMP_ENTRY
void *guarded_subi(void *, void *);
void *floormod2expi_32(void *);
void *floormod2expi_64(void *);
void *andi_mask(void *, std::int64_t);
void *emodsi_power(void *, std::int64_t);
void *andi_32(void *);
void *andi_64(void *);
void *emodsi_32(void *);
void *emodsi_64(void *);
void *ipowi_two(std::int64_t);
void *shli_one(std::int64_t);
void *square_muli(void *);
void *fifth_muli(void *);
void *fifth_ipowi(void *);
void *power_loop(void *, std::int64_t);
std::int64_t successor(void *);
std::int64_t int_cases(void *);
std::int64_t int_cases_noti(void *);
void *divsi(void *, void *);
void *remsi(void *, void *);
void *floordivsi(void *, void *);
void *floormodsi(void *, void *);
void *edivsi(void *, void *);
void *emodsi(void *, void *);
void *composed_edivsi(void *, void *);
void *composed_emodsi(void *, void *);
void *divsi_lean(void *, void *);
void *guarded_divsi(void *, void *);
void *remsi_lean(void *, void *);
void *guarded_remsi(void *, void *);
void *edivsi_lean(void *, void *);
void *guarded_edivsi(void *, void *);
void *emodsi_lean(void *, void *);
void *guarded_emodsi(void *, void *);
rcgmp_mpz_qr divremsi(void *, void *);
std::int64_t trunci(void *);
std::int64_t trunci_overflow(void *);
std::int64_t log2i(void *);
std::int64_t shifted_trunci_log2(void *);
void *gcdi(void *, void *);
void *euclid(void *, void *);
void *divsi_exact(void *, void *);
void *divsi_exact_by_12(void *);
void *divsi_by_12(void *);
void *divsi_exact_by_8(void *);
void *divsi_by_8(void *);
}

namespace {

using Binary = void *(*)(void *, void *);
using Unary = void *(*)(void *);
using GmpBinary = void (*)(mpz_ptr, mpz_srcptr, mpz_srcptr);
using GmpUnary = void (*)(mpz_ptr, mpz_srcptr);

void *immediate(std::intptr_t value) {
  return reinterpret_cast<void *>((static_cast<std::uintptr_t>(value) << 2) |
                                  1);
}

// Two-limb magnitudes, least significant limb first.
constexpr std::array<std::uintptr_t, 2> lhsWords{0x9e3779b97f4a7c15, 0x1234567};
constexpr std::array<std::uintptr_t, 2> rhsWords{0xbf58476d1ce4e5b9, 0x89abcd};
constexpr std::intptr_t smallLhs = 1234567, smallRhs = 89;

void *heap(const std::array<std::uintptr_t, 2> &words) {
  return rcgmp_mpz_from_words(words.data(), words.size(), 0, 0);
}

// An owned GMP integer.
struct Integer {
  mpz_t value;
  explicit Integer(std::intptr_t small) { mpz_init_set_si(value, small); }
  explicit Integer(const std::array<std::uintptr_t, 2> &words) {
    mpz_init(value);
    mpz_import(value, words.size(), -1, sizeof(std::uintptr_t), 0, 0,
               words.data());
  }
  Integer(const Integer &) = delete;
  ~Integer() { mpz_clear(value); }
};

// The compiled entry points of each counting mode.
struct Counted {
  Binary binary;
  void *(*retain)(void *);
  void (*release)(void *);
};

void binary(benchmark::State &state, Binary op, void *x, void *y) {
  for (auto _ : state) {
    benchmark::DoNotOptimize(x);
    benchmark::DoNotOptimize(y);
    benchmark::DoNotOptimize(op(x, y));
  }
}

void binaryShared(benchmark::State &state, Counted counted, void *x, void *y) {
  for (auto _ : state) {
    counted.retain(x);
    counted.retain(y);
    counted.release(counted.binary(x, y));
  }
  counted.release(x);
  counted.release(y);
}

void binaryUnique(benchmark::State &state, Counted counted, void *x,
                  std::intptr_t first, std::intptr_t second) {
  for (auto _ : state) {
    x = counted.binary(x, immediate(first));
    x = counted.binary(x, immediate(second));
    benchmark::DoNotOptimize(x);
  }
  counted.release(x);
}

void gmpBinary(benchmark::State &state, GmpBinary op, mpz_srcptr x,
               mpz_srcptr y) {
  for (auto _ : state) {
    mpz_t result;
    mpz_init(result);
    op(result, x, y);
    benchmark::DoNotOptimize(result);
    mpz_clear(result);
  }
}

void gmpBinaryInPlace(benchmark::State &state, GmpBinary op,
                      std::intptr_t first, std::intptr_t second) {
  Integer x(lhsWords), a(first), b(second);
  for (auto _ : state) {
    op(x.value, x.value, a.value);
    op(x.value, x.value, b.value);
    benchmark::DoNotOptimize(x.value);
  }
}

// lhs - rhs when positive, and zero otherwise, as GMP code computes it.
void gmpMonus(mpz_ptr result, mpz_srcptr lhs, mpz_srcptr rhs) {
  if (mpz_cmp(lhs, rhs) > 0)
    mpz_sub(result, lhs, rhs);
  else
    mpz_set_ui(result, 0);
}

struct BinaryCase {
  const char *name;
  std::array<Binary, 2> compiled;
  GmpBinary gmp;
  // Immediates applied in turn to a unique heap operand, restoring its value.
  std::intptr_t first, second;
};

const std::array<BinaryCase, 7> binaries{{
    {"addi", {addi, addi_atomic}, mpz_add, smallRhs, -smallRhs},
    {"subi", {subi, subi_atomic}, mpz_sub, smallRhs, -smallRhs},
    {"monusi", {monusi, monusi_atomic}, gmpMonus, smallRhs, -smallRhs},
    {"muli", {muli, muli_atomic}, mpz_mul, -1, -1},
    {"andi", {andi, andi_atomic}, mpz_and, -2, -2},
    {"ori", {ori, ori_atomic}, mpz_ior, 1, 1},
    {"xori", {xori, xori_atomic}, mpz_xor, smallRhs, smallRhs},
}};

const std::array<Counted, 2> countings{{
    {nullptr, retain, release},
    {nullptr, retain_atomic, release_atomic},
}};

const std::array<const char *, 2> suffixes{"", "_atomic"};

void registerBinaries() {
  static Integer smallX(smallLhs), smallY(smallRhs), heapX(lhsWords),
      heapY(rhsWords);
  for (const BinaryCase &op : binaries) {
    for (int counting = 0; counting < 2; ++counting) {
      std::string name = std::string(op.name) + suffixes[counting];
      Counted counted = countings[counting];
      counted.binary = op.compiled[counting];
      benchmark::RegisterBenchmark(name + "/immediates", binary, counted.binary,
                                   immediate(smallLhs), immediate(smallRhs));
      benchmark::RegisterBenchmark(
          name + "/shared heap, immediate", [=](benchmark::State &state) {
            binaryShared(state, counted, heap(lhsWords), immediate(smallRhs));
          });
      benchmark::RegisterBenchmark(
          name + "/unique heap, immediate x2", [=](benchmark::State &state) {
            binaryUnique(state, counted, heap(lhsWords), op.first, op.second);
          });
      benchmark::RegisterBenchmark(
          name + "/shared heaps", [=](benchmark::State &state) {
            binaryShared(state, counted, heap(lhsWords), heap(rhsWords));
          });
    }
    std::string gmp = std::string("gmp ") + op.name;
    benchmark::RegisterBenchmark(gmp + "/immediates", gmpBinary, op.gmp,
                                 smallX.value, smallY.value);
    benchmark::RegisterBenchmark(gmp + "/shared heap, immediate", gmpBinary,
                                 op.gmp, heapX.value, smallY.value);
    benchmark::RegisterBenchmark(gmp + "/unique heap, immediate x2",
                                 gmpBinaryInPlace, op.gmp, op.first, op.second);
    benchmark::RegisterBenchmark(gmp + "/shared heaps", gmpBinary, op.gmp,
                                 heapX.value, heapY.value);
  }
}

struct UnaryCase {
  const char *name;
  std::array<Unary, 2> compiled;
  GmpUnary gmp;
};

const std::array<UnaryCase, 3> unaries{{
    {"noti", {noti, noti_atomic}, mpz_com},
    {"negi", {negi, negi_atomic}, mpz_neg},
    {"absi", {absi, absi_atomic}, mpz_abs},
}};

void registerUnaries() {
  static Integer smallX(smallLhs), heapX(lhsWords);
  for (const UnaryCase &op : unaries) {
    for (int counting = 0; counting < 2; ++counting) {
      std::string name = std::string(op.name) + suffixes[counting];
      Unary apply = op.compiled[counting];
      Counted counted = countings[counting];
      benchmark::RegisterBenchmark(name + "/immediate",
                                   [=](benchmark::State &state) {
                                     void *x = immediate(smallLhs);
                                     for (auto _ : state) {
                                       benchmark::DoNotOptimize(x);
                                       benchmark::DoNotOptimize(apply(x));
                                     }
                                   });
      benchmark::RegisterBenchmark(name + "/shared heap",
                                   [=](benchmark::State &state) {
                                     void *x = heap(lhsWords);
                                     for (auto _ : state) {
                                       counted.retain(x);
                                       counted.release(apply(x));
                                     }
                                     counted.release(x);
                                   });
      // Applying any of these twice to a nonnegative integer restores it.
      benchmark::RegisterBenchmark(name + "/unique heap x2",
                                   [=](benchmark::State &state) {
                                     void *x = heap(lhsWords);
                                     for (auto _ : state) {
                                       x = apply(apply(x));
                                       benchmark::DoNotOptimize(x);
                                     }
                                     counted.release(x);
                                   });
    }
    std::string gmp = std::string("gmp ") + op.name;
    GmpUnary apply = op.gmp;
    for (auto [label, operand] :
         {std::pair{"/immediate", &smallX}, {"/shared heap", &heapX}})
      benchmark::RegisterBenchmark(gmp + label, [=](benchmark::State &state) {
        for (auto _ : state) {
          mpz_t result;
          mpz_init(result);
          apply(result, operand->value);
          benchmark::DoNotOptimize(result);
          mpz_clear(result);
        }
      });
    benchmark::RegisterBenchmark(gmp + "/unique heap x2",
                                 [=](benchmark::State &state) {
                                   Integer x(lhsWords);
                                   for (auto _ : state) {
                                     apply(x.value, x.value);
                                     apply(x.value, x.value);
                                     benchmark::DoNotOptimize(x.value);
                                   }
                                 });
  }
}

// Monus whose result is zero, which the binary benchmarks do not reach, and
// the comparison and exact subtraction that compute monus otherwise, with
// nonatomic counting only. Immediates are not retained, as in the binary
// benchmarks.
void registerMonus() {
  static Integer smallX(smallLhs), smallY(smallRhs), heapX(lhsWords),
      heapY(rhsWords);
  struct MonusCase {
    const char *label;
    // Whether the result is zero, and the binary benchmarks lack the case.
    bool zero;
    void *(*lhs)(), *(*rhs)();
    mpz_srcptr gmpLhs, gmpRhs;
  };
  const std::array<MonusCase, 5> cases{{
      {"/immediates", false, [] { return immediate(smallLhs); },
       [] { return immediate(smallRhs); }, smallX.value, smallY.value},
      {"/immediates, zero", true, [] { return immediate(smallRhs); },
       [] { return immediate(smallLhs); }, smallY.value, smallX.value},
      {"/shared heap, immediate", false, [] { return heap(lhsWords); },
       [] { return immediate(smallRhs); }, heapX.value, smallY.value},
      {"/shared heaps", false, [] { return heap(lhsWords); },
       [] { return heap(rhsWords); }, heapX.value, heapY.value},
      {"/shared heaps, zero", true, [] { return heap(rhsWords); },
       [] { return heap(lhsWords); }, heapY.value, heapX.value},
  }};
  for (const MonusCase &test : cases) {
    bool immediates = std::string_view(test.label).starts_with("/immediates");
    for (auto [name, op] :
         {std::pair{"monusi", monusi}, {"guarded subi", guarded_subi}}) {
      if (op == monusi && !test.zero)
        continue;
      Counted counted = countings[0];
      counted.binary = op;
      benchmark::RegisterBenchmark(
          std::string(name) + test.label, [=](benchmark::State &state) {
            if (immediates)
              binary(state, op, test.lhs(), test.rhs());
            else
              binaryShared(state, counted, test.lhs(), test.rhs());
          });
    }
    if (test.zero)
      benchmark::RegisterBenchmark(std::string("gmp monusi") + test.label,
                                   gmpBinary, gmpMonus, test.gmpLhs,
                                   test.gmpRhs);
  }
}

struct ShiftCase {
  const char *name;
  std::array<void *(*)(void *, std::int64_t), 2> compiled;
  void (*gmp)(mpz_ptr, mpz_srcptr, mp_bitcnt_t);
  // Whether shifting an immediate can leave the immediate range.
  bool overflows;
};

const std::array<ShiftCase, 2> shiftCases{{
    {"shli", {shli, shli_atomic}, mpz_mul_2exp, true},
    {"shrsi", {shrsi, shrsi_atomic}, mpz_fdiv_q_2exp, false},
}};

// A few bits, and enough to move smallLhs out of the immediate range.
constexpr std::int64_t smallShift = 5, overflowingShift = 50;

void registerShifts() {
  static Integer smallX(smallLhs), heapX(lhsWords);
  for (const ShiftCase &op : shiftCases) {
    for (int counting = 0; counting < 2; ++counting) {
      std::string name = std::string(op.name) + suffixes[counting];
      auto apply = op.compiled[counting];
      Counted counted = countings[counting];
      for (auto [label, amount] :
           {std::pair{"/immediate", smallShift},
            {"/immediate overflowing", overflowingShift}}) {
        if (amount == overflowingShift && !op.overflows)
          continue;
        benchmark::RegisterBenchmark(name + label,
                                     [=](benchmark::State &state) {
                                       void *x = immediate(smallLhs);
                                       for (auto _ : state) {
                                         benchmark::DoNotOptimize(x);
                                         counted.release(apply(x, amount));
                                       }
                                     });
      }
      benchmark::RegisterBenchmark(name + "/shared heap",
                                   [=](benchmark::State &state) {
                                     void *x = heap(lhsWords);
                                     for (auto _ : state) {
                                       counted.retain(x);
                                       counted.release(apply(x, smallShift));
                                     }
                                     counted.release(x);
                                   });
    }
    std::string gmp = std::string("gmp ") + op.name;
    auto apply = op.gmp;
    for (auto [label, operand, amount] :
         {std::tuple{"/immediate", &smallX, smallShift},
          {"/immediate overflowing", &smallX, overflowingShift},
          {"/shared heap", &heapX, smallShift}}) {
      if (amount == overflowingShift && !op.overflows)
        continue;
      benchmark::RegisterBenchmark(gmp + label, [=](benchmark::State &state) {
        for (auto _ : state) {
          mpz_t result;
          mpz_init(result);
          apply(result, operand->value, amount);
          benchmark::DoNotOptimize(result);
          mpz_clear(result);
        }
      });
    }
  }
  // Shifting left and back restores a unique heap operand in place.
  for (int counting = 0; counting < 2; ++counting) {
    auto left = shiftCases[0].compiled[counting];
    auto right = shiftCases[1].compiled[counting];
    Counted counted = countings[counting];
    benchmark::RegisterBenchmark(std::string("shli, shrsi") +
                                     suffixes[counting] + "/unique heap",
                                 [=](benchmark::State &state) {
                                   void *x = heap(lhsWords);
                                   for (auto _ : state) {
                                     x = right(left(x, smallShift), smallShift);
                                     benchmark::DoNotOptimize(x);
                                   }
                                   counted.release(x);
                                 });
  }
  benchmark::RegisterBenchmark(
      "gmp mul_2exp, fdiv_q_2exp/unique heap", [](benchmark::State &state) {
        Integer x(lhsWords);
        for (auto _ : state) {
          mpz_mul_2exp(x.value, x.value, smallShift);
          mpz_fdiv_q_2exp(x.value, x.value, smallShift);
          benchmark::DoNotOptimize(x.value);
        }
      });
}

// An eight-limb magnitude, least significant limb first.
constexpr std::array<std::uintptr_t, 8> longWords{
    0x9e3779b97f4a7c15, 0xbf58476d1ce4e5b9, 0x94d049bb133111eb,
    0x2545f4914f6cdd1d, 0xd6e8feb86659fd93, 0xa0761d6478bd642f,
    0xe7037ed1a0b428db, 0x1234567};

// An immediate operand, or a heap operand of the given limbs and sign.
struct RemainderOperand {
  std::intptr_t small;
  std::span<const std::uintptr_t> words = {};
  bool negative = false;

  // The compiled operand, which owns one reference.
  void *compiled() const {
    return words.empty()
               ? immediate(small)
               : rcgmp_mpz_from_words(words.data(), words.size(), negative, 0);
  }
  void gmp(mpz_ptr value) const {
    if (words.empty()) {
      mpz_set_si(value, small);
      return;
    }
    mpz_import(value, words.size(), -1, sizeof(std::uintptr_t), 0, 0,
               words.data());
    if (negative)
      mpz_neg(value, value);
  }
};

// Applies a remainder to an operand, retaining a heap operand first so that
// the result needs storage of its own unless it is the operand, and releases
// the result, as the shift benchmarks do.
template <typename Apply>
void remainder(benchmark::State &state, Counted counted, Apply apply,
               const RemainderOperand &operand) {
  void *x = operand.compiled();
  bool shared = !operand.words.empty();
  for (auto _ : state) {
    benchmark::DoNotOptimize(x);
    if (shared)
      counted.retain(x);
    counted.release(apply(x));
  }
  counted.release(x);
}

// Remainders by powers of two of immediates and of shared heap integers of
// either sign, with powers below and above them, next to GMP and to the forms
// that compute them without floormod2expi: a mask of 2^n - 1, and the
// Euclidean remainder by 2^n. Constant amounts compare with constant masks and
// divisors.
void registerRemainders() {
  struct RemainderCase {
    const char *label;
    RemainderOperand operand;
    std::int64_t amount;
  };
  const std::array<RemainderCase, 14> cases{{
      {"/immediate, 20 bits", {smallLhs}, 20},
      {"/negative immediate, 20 bits", {-smallLhs}, 20},
      {"/immediate, 64 bits", {smallLhs}, 64},
      {"/negative immediate, 64 bits", {-smallLhs}, 64},
      {"/shared heap, 20 bits", {0, lhsWords}, 20},
      {"/shared heap, 80 bits", {0, lhsWords}, 80},
      {"/shared heap, 200 bits", {0, lhsWords}, 200},
      {"/shared negative heap, 20 bits", {0, lhsWords, true}, 20},
      {"/shared negative heap, 80 bits", {0, lhsWords, true}, 80},
      {"/shared negative heap, 200 bits", {0, lhsWords, true}, 200},
      {"/shared heap of 8 limbs, 300 bits", {0, longWords}, 300},
      {"/shared heap of 8 limbs, 600 bits", {0, longWords}, 600},
      {"/shared negative heap of 8 limbs, 300 bits", {0, longWords, true}, 300},
      {"/shared negative heap of 8 limbs, 600 bits", {0, longWords, true}, 600},
  }};
  using Remainder = void *(*)(void *, std::int64_t);
  for (const RemainderCase &test : cases) {
    auto [label, operand, amount] = test;
    for (int counting = 0; counting < 2; ++counting)
      benchmark::RegisterBenchmark(
          std::string("floormod2expi") + suffixes[counting] + label,
          [=](benchmark::State &state) {
            Remainder apply = counting ? floormod2expi_atomic : floormod2expi;
            remainder(
                state, countings[counting],
                [=](void *x) { return apply(x, amount); }, operand);
          });
    for (auto [name, apply] :
         {std::pair{"andi mask", andi_mask}, {"emodsi power", emodsi_power}})
      benchmark::RegisterBenchmark(
          name + std::string(label), [=](benchmark::State &state) {
            remainder(
                state, countings[0], [=](void *x) { return apply(x, amount); },
                operand);
          });
    benchmark::RegisterBenchmark(std::string("gmp fdiv_r_2exp") + label,
                                 [=](benchmark::State &state) {
                                   mpz_t x;
                                   mpz_init(x);
                                   operand.gmp(x);
                                   for (auto _ : state) {
                                     mpz_t result;
                                     mpz_init(result);
                                     mpz_fdiv_r_2exp(result, x, amount);
                                     benchmark::DoNotOptimize(result);
                                     mpz_clear(result);
                                   }
                                   mpz_clear(x);
                                 });
  }
  for (auto [label, operand] :
       {std::pair{"/immediate", RemainderOperand{smallLhs}},
        {"/negative immediate", {-smallLhs}},
        {"/shared heap", {0, lhsWords}},
        {"/shared negative heap", {0, lhsWords, true}}})
    for (auto [name, apply] : {std::pair{"floormod2expi 32", floormod2expi_32},
                               {"andi 2^32-1", andi_32},
                               {"emodsi 2^32", emodsi_32},
                               {"floormod2expi 64", floormod2expi_64},
                               {"andi 2^64-1", andi_64},
                               {"emodsi 2^64", emodsi_64}})
      benchmark::RegisterBenchmark(
          name + std::string(label), [=](benchmark::State &state) {
            remainder(state, countings[0], apply, operand);
          });
}

// Powers of immediates on the inline path and in the runtime, and of a shared
// heap base, next to GMP and to the forms that compute them without ipowi:
// products for a fixed exponent and a loop that squares and multiplies. Powers
// of two compare with a shift of one. Results are released, as in the shift
// benchmarks.
void registerPowers() {
  static Integer heapX(lhsWords);
  struct PowerCase {
    const char *label;
    // The immediate base, or zero for the heap base.
    std::intptr_t base;
    std::int64_t exponent;
  };
  const std::array<PowerCase, 7> cases{{
      {"/immediate, exponent 2", smallLhs, 2},
      {"/immediate, exponent 5", 7, 5},
      {"/immediate, exponent 20", 7, 20},
      {"/immediate, exponent 50", 3, 50},
      {"/shared heap, exponent 2", 0, 2},
      {"/shared heap, exponent 5", 0, 5},
      {"/shared heap, exponent 20", 0, 20},
  }};
  using Power = void *(*)(void *, std::int64_t);
  auto run = [](benchmark::State &state, Counted counted, Power apply,
                std::intptr_t base, std::int64_t exponent) {
    if (base) {
      void *x = immediate(base);
      for (auto _ : state) {
        benchmark::DoNotOptimize(x);
        counted.release(apply(x, exponent));
      }
      return;
    }
    void *x = heap(lhsWords);
    for (auto _ : state) {
      counted.retain(x);
      counted.release(apply(x, exponent));
    }
    counted.release(x);
  };
  for (const PowerCase &test : cases) {
    for (int counting = 0; counting < 2; ++counting)
      benchmark::RegisterBenchmark(
          std::string("ipowi") + suffixes[counting] + test.label,
          [=](benchmark::State &state) {
            run(state, countings[counting], counting ? ipowi_atomic : ipowi,
                test.base, test.exponent);
          });
    Power fixed = nullptr;
    if (test.exponent == 2)
      fixed = [](void *x, std::int64_t) { return square_muli(x); };
    else if (test.exponent == 5)
      fixed = [](void *x, std::int64_t) { return fifth_muli(x); };
    if (fixed)
      benchmark::RegisterBenchmark(
          std::string("muli") + test.label, [=](benchmark::State &state) {
            run(state, countings[0], fixed, test.base, test.exponent);
          });
    // The constant exponent unrolls into products on the inline path.
    if (test.exponent == 5)
      benchmark::RegisterBenchmark(
          std::string("ipowi by constant") + test.label,
          [=](benchmark::State &state) {
            run(
                state, countings[0],
                [](void *x, std::int64_t) { return fifth_ipowi(x); }, test.base,
                test.exponent);
          });
    benchmark::RegisterBenchmark(
        std::string("muli loop") + test.label, [=](benchmark::State &state) {
          run(state, countings[0], power_loop, test.base, test.exponent);
        });
    benchmark::RegisterBenchmark(
        std::string("gmp pow_ui") + test.label, [=](benchmark::State &state) {
          Integer small(test.base);
          mpz_srcptr x = test.base ? small.value : heapX.value;
          for (auto _ : state) {
            mpz_t result;
            mpz_init(result);
            mpz_pow_ui(result, x, test.exponent);
            benchmark::DoNotOptimize(result);
            mpz_clear(result);
          }
        });
  }
  for (std::int64_t exponent : {40, 100}) {
    std::string label = "/exponent " + std::to_string(exponent);
    for (auto [name, apply] :
         {std::pair{"ipowi 2", ipowi_two}, {"shli 1", shli_one}})
      benchmark::RegisterBenchmark(name + label, [=](benchmark::State &state) {
        for (auto _ : state) {
          std::int64_t n = exponent;
          benchmark::DoNotOptimize(n);
          release(apply(n));
        }
      });
    benchmark::RegisterBenchmark("gmp ui_pow_ui 2" + label,
                                 [=](benchmark::State &state) {
                                   for (auto _ : state) {
                                     mpz_t result;
                                     mpz_init(result);
                                     mpz_ui_pow_ui(result, 2, exponent);
                                     benchmark::DoNotOptimize(result);
                                     mpz_clear(result);
                                   }
                                 });
  }
}

struct DivisionCase {
  const char *name;
  Binary compiled;
  GmpBinary gmp;
};

const std::array<DivisionCase, 4> divisions{{
    {"divsi", divsi, mpz_tdiv_q},
    {"remsi", remsi, mpz_tdiv_r},
    {"floordivsi", floordivsi, mpz_fdiv_q},
    {"floormodsi", floormodsi, mpz_fdiv_r},
}};

// Divisions by a negative divisor, so that floor division adjusts inexact
// results, with nonatomic counting only, as atomic counting costs what it
// does for the other operations.
void registerDivisions() {
  static Integer smallX(smallLhs), smallY(-smallRhs), heapX(lhsWords),
      heapY(rhsWords);
  mpz_neg(heapY.value, heapY.value);
  Counted counted = countings[0];
  // Each run makes its operands, which binaryShared releases at its end.
  using Operands = std::pair<void *, void *> (*)();
  const std::array<std::pair<const char *, Operands>, 3> cases{{
      {"/immediates",
       [] { return std::pair{immediate(smallLhs), immediate(-smallRhs)}; }},
      {"/shared heap, immediate",
       [] { return std::pair{heap(lhsWords), immediate(-smallRhs)}; }},
      {"/shared heaps",
       [] {
         return std::pair{heap(lhsWords), rcgmp_mpz_neg(heap(rhsWords), 0)};
       }},
  }};
  for (const DivisionCase &op : divisions) {
    counted.binary = op.compiled;
    for (auto [label, operands] : cases)
      benchmark::RegisterBenchmark(std::string(op.name) + label,
                                   [=](benchmark::State &state) {
                                     auto [x, y] = operands();
                                     binaryShared(state, counted, x, y);
                                   });
    for (auto [label, x, y] : {std::tuple{"/immediates", &smallX, &smallY},
                               {"/shared heap, immediate", &heapX, &smallY},
                               {"/shared heaps", &heapX, &heapY}})
      benchmark::RegisterBenchmark(std::string("gmp ") + op.name + label,
                                   gmpBinary, op.gmp, x->value, y->value);
  }
  for (auto [label, operands] : cases)
    benchmark::RegisterBenchmark(std::string("divremsi") + label,
                                 [=](benchmark::State &state) {
                                   auto [x, y] = operands();
                                   for (auto _ : state) {
                                     retain(x);
                                     retain(y);
                                     auto [q, r] = divremsi(x, y);
                                     release(q);
                                     release(r);
                                   }
                                   release(x);
                                   release(y);
                                 });
  for (auto [label, x, y] : {std::tuple{"/immediates", &smallX, &smallY},
                             {"/shared heap, immediate", &heapX, &smallY},
                             {"/shared heaps", &heapX, &heapY}})
    benchmark::RegisterBenchmark(std::string("gmp tdiv_qr") + label,
                                 [=](benchmark::State &state) {
                                   for (auto _ : state) {
                                     mpz_t q, r;
                                     mpz_init(q);
                                     mpz_init(r);
                                     mpz_tdiv_qr(q, r, x->value, y->value);
                                     benchmark::DoNotOptimize(q);
                                     benchmark::DoNotOptimize(r);
                                     mpz_clear(q);
                                     mpz_clear(r);
                                   }
                                 });
}

void gmpEuclideanQuotient(mpz_ptr result, mpz_srcptr lhs, mpz_srcptr rhs) {
  (mpz_sgn(rhs) > 0 ? mpz_fdiv_q : mpz_cdiv_q)(result, lhs, rhs);
}

// Euclidean division, mostly of a negative dividend, next to the same result
// composed from floored division, floored division itself, and GMP. Floored
// division is the Euclidean one for a positive divisor.
void registerEuclidean() {
  Counted counted = countings[0];
  using Operands = std::pair<void *, void *> (*)();
  const std::array<std::pair<const char *, Operands>, 7> cases{{
      {"/immediates, nonnegative",
       [] { return std::pair{immediate(smallLhs), immediate(smallRhs)}; }},
      {"/immediates, positive",
       [] { return std::pair{immediate(-smallLhs), immediate(smallRhs)}; }},
      {"/immediates, negative",
       [] { return std::pair{immediate(-smallLhs), immediate(-smallRhs)}; }},
      {"/shared heap, immediate, positive",
       [] {
         return std::pair{rcgmp_mpz_neg(heap(lhsWords), 0),
                          immediate(smallRhs)};
       }},
      {"/shared heap, immediate, negative",
       [] {
         return std::pair{rcgmp_mpz_neg(heap(lhsWords), 0),
                          immediate(-smallRhs)};
       }},
      {"/shared heaps, positive",
       [] {
         return std::pair{rcgmp_mpz_neg(heap(lhsWords), 0), heap(rhsWords)};
       }},
      {"/shared heaps, negative",
       [] {
         return std::pair{rcgmp_mpz_neg(heap(lhsWords), 0),
                          rcgmp_mpz_neg(heap(rhsWords), 0)};
       }},
  }};
  for (auto [name, op] : {std::pair{"edivsi", edivsi},
                          {"composed edivsi", composed_edivsi},
                          {"floordivsi", floordivsi},
                          {"emodsi", emodsi},
                          {"composed emodsi", composed_emodsi},
                          {"floormodsi", floormodsi}}) {
    counted.binary = op;
    for (auto [label, operands] : cases)
      benchmark::RegisterBenchmark(std::string(name) + label,
                                   [=](benchmark::State &state) {
                                     auto [x, y] = operands();
                                     binaryShared(state, counted, x, y);
                                   });
  }
  static Integer x(smallLhs), y(smallRhs), negativeX(-smallLhs),
      negativeY(-smallRhs), heapX(lhsWords), heapY(rhsWords),
      negativeHeapY(rhsWords);
  mpz_neg(heapX.value, heapX.value);
  mpz_neg(negativeHeapY.value, negativeHeapY.value);
  const std::array<std::tuple<const char *, Integer *, Integer *>, 7> values{{
      {"/immediates, nonnegative", &x, &y},
      {"/immediates, positive", &negativeX, &y},
      {"/immediates, negative", &negativeX, &negativeY},
      {"/shared heap, immediate, positive", &heapX, &y},
      {"/shared heap, immediate, negative", &heapX, &negativeY},
      {"/shared heaps, positive", &heapX, &heapY},
      {"/shared heaps, negative", &heapX, &negativeHeapY},
  }};
  for (auto [name, op] :
       {std::pair{"gmp edivsi", gmpEuclideanQuotient}, {"gmp emodsi", mpz_mod}})
    for (auto [label, lhs, rhs] : values)
      benchmark::RegisterBenchmark(std::string(name) + label, gmpBinary, op,
                                   lhs->value, rhs->value);
}

// Lean division, the same result from a comparison and a division whose zero
// divisor is undefined, and trapping division, which terminates on a zero
// divisor and so has no rows for one.
void registerLean() {
  Counted counted = countings[0];
  using Operands = std::pair<void *, void *> (*)();
  const std::array<std::tuple<const char *, Operands, bool>, 5> cases{{
      {"/immediates",
       [] { return std::pair{immediate(smallLhs), immediate(-smallRhs)}; },
       false},
      {"/immediates, zero",
       [] { return std::pair{immediate(smallLhs), immediate(0)}; }, true},
      {"/shared heap, immediate",
       [] { return std::pair{heap(lhsWords), immediate(-smallRhs)}; }, false},
      {"/shared heap, zero",
       [] { return std::pair{heap(lhsWords), immediate(0)}; }, true},
      {"/shared heaps",
       [] {
         return std::pair{heap(lhsWords), rcgmp_mpz_neg(heap(rhsWords), 0)};
       },
       false},
  }};
  struct LeanCase {
    const char *name;
    Binary trap, lean, guarded;
  };
  for (auto [name, trap, lean, guarded] :
       {LeanCase{"divsi", divsi, divsi_lean, guarded_divsi},
        {"remsi", remsi, remsi_lean, guarded_remsi},
        {"edivsi", edivsi, edivsi_lean, guarded_edivsi},
        {"emodsi", emodsi, emodsi_lean, guarded_emodsi}})
    for (auto [label, operands, zero] : cases)
      for (auto [mode, op] :
           {std::pair{" trap", trap}, {" lean", lean}, {" guarded", guarded}}) {
        if (zero && op == trap)
          continue;
        counted.binary = op;
        benchmark::RegisterBenchmark(std::string(name) + mode + label,
                                     [=](benchmark::State &state) {
                                       auto [x, y] = operands();
                                       binaryShared(state, counted, x, y);
                                     });
      }
}

// Truncations to i64 of borrowed integers, and GMP reading the low word with
// and without a range check.
void registerTruncations() {
  static Integer smallX(smallLhs), heapX(lhsWords);
  for (auto [label, x, gmp] :
       {std::tuple{"/immediate", immediate(smallLhs), &smallX},
        {"/heap", heap(lhsWords), &heapX}}) {
    // Direct calls, as GMP's are.
    benchmark::RegisterBenchmark(std::string("trunci") + label,
                                 [=](benchmark::State &state) mutable {
                                   for (auto _ : state) {
                                     benchmark::DoNotOptimize(x);
                                     benchmark::DoNotOptimize(trunci(x));
                                   }
                                 });
    benchmark::RegisterBenchmark(std::string("trunci_overflow") + label,
                                 [=](benchmark::State &state) mutable {
                                   for (auto _ : state) {
                                     benchmark::DoNotOptimize(x);
                                     benchmark::DoNotOptimize(
                                         trunci_overflow(x));
                                   }
                                 });
    benchmark::RegisterBenchmark(std::string("gmp get_si") + label,
                                 [=](benchmark::State &state) {
                                   mpz_srcptr y = gmp->value;
                                   for (auto _ : state) {
                                     benchmark::DoNotOptimize(y);
                                     benchmark::DoNotOptimize(mpz_get_si(y));
                                   }
                                 });
    benchmark::RegisterBenchmark(
        std::string("gmp fits_slong_p, get_si") + label,
        [=](benchmark::State &state) {
          mpz_srcptr y = gmp->value;
          for (auto _ : state) {
            benchmark::DoNotOptimize(y);
            benchmark::DoNotOptimize(mpz_fits_slong_p(y));
            benchmark::DoNotOptimize(mpz_get_si(y));
          }
        });
  }
}

// Logarithms of borrowed integers, the same logarithms found by shifting and
// truncating, and GMP counting bits, which it counts as one for zero.
void registerLogarithms() {
  static Integer smallX(smallLhs), heapX(lhsWords);
  for (auto [label, x, gmp] :
       {std::tuple{"/immediate", immediate(smallLhs), &smallX},
        {"/heap", heap(lhsWords), &heapX}}) {
    for (auto [name, op] : {std::pair{"log2i", log2i},
                            {"shifted trunci log2", shifted_trunci_log2}})
      benchmark::RegisterBenchmark(std::string(name) + label,
                                   [=](benchmark::State &state) mutable {
                                     for (auto _ : state) {
                                       benchmark::DoNotOptimize(x);
                                       benchmark::DoNotOptimize(op(x));
                                     }
                                   });
    benchmark::RegisterBenchmark(
        std::string("gmp sizeinbase") + label, [=](benchmark::State &state) {
          mpz_srcptr y = gmp->value;
          for (auto _ : state) {
            benchmark::DoNotOptimize(y);
            benchmark::DoNotOptimize(mpz_sgn(y) ? mpz_sizeinbase(y, 2) - 1 : 0);
          }
        });
  }
}

void registerOthers() {
  static Integer smallX(smallLhs), smallY(smallRhs), heapX(lhsWords),
      heapY(rhsWords), five(5);
  for (int counting = 0; counting < 2; ++counting) {
    std::string suffix = suffixes[counting];
    auto compare = counting ? cmpi_atomic : cmpi;
    for (auto [label, x, y] :
         {std::tuple{"/immediates", immediate(smallLhs), immediate(smallRhs)},
          {"/heap, immediate", heap(lhsWords), immediate(smallRhs)},
          {"/heaps", heap(lhsWords), heap(rhsWords)}})
      benchmark::RegisterBenchmark("cmpi" + suffix + label,
                                   [=](benchmark::State &state) mutable {
                                     for (auto _ : state) {
                                       benchmark::DoNotOptimize(x);
                                       benchmark::DoNotOptimize(compare(x, y));
                                     }
                                   });
    auto make = counting ? extsi_atomic : extsi;
    Counted counted = countings[counting];
    for (auto [label, value] : {std::pair{"/immediate", std::int64_t(12345)},
                                {"/heap", std::int64_t(1) << 62}})
      benchmark::RegisterBenchmark("extsi" + suffix + label,
                                   [=](benchmark::State &state) {
                                     for (auto _ : state) {
                                       std::int64_t v = value;
                                       benchmark::DoNotOptimize(v);
                                       counted.release(make(v));
                                     }
                                   });
    benchmark::RegisterBenchmark("retain and release" + suffix + "/heap",
                                 [=](benchmark::State &state) {
                                   void *x = heap(lhsWords);
                                   for (auto _ : state) {
                                     benchmark::DoNotOptimize(x);
                                     counted.release(counted.retain(x));
                                   }
                                   counted.release(x);
                                 });
  }
  for (auto [label, x, y] : {std::tuple{"/immediates", &smallX, &smallY},
                             {"/heap, immediate", &heapX, &smallY},
                             {"/heaps", &heapX, &heapY}})
    benchmark::RegisterBenchmark(
        std::string("gmp cmp") + label, [=](benchmark::State &state) {
          for (auto _ : state)
            benchmark::DoNotOptimize(mpz_cmp(x->value, y->value));
        });
  for (auto [label, value] :
       {std::pair{"/immediate", long(12345)}, {"/heap", long(1) << 62}})
    benchmark::RegisterBenchmark(std::string("gmp init_set_si") + label,
                                 [=](benchmark::State &state) {
                                   for (auto _ : state) {
                                     long v = value;
                                     benchmark::DoNotOptimize(v);
                                     mpz_t result;
                                     mpz_init_set_si(result, v);
                                     benchmark::DoNotOptimize(result);
                                     mpz_clear(result);
                                   }
                                 });

  // match n { 0, 1 => 1; at_least 2 (k) => k > 1000 ? 7 : 3 } on a shared
  // scrutinee, and GMP computing the remainder as a new integer.
  for (auto [label, operand] : {std::pair{"/immediate", immediate(5)},
                                {"/shared heap", heap(lhsWords)}})
    benchmark::RegisterBenchmark(
        std::string("successor match") + label, [=](benchmark::State &state) {
          for (auto _ : state) {
            retain(operand);
            benchmark::DoNotOptimize(successor(operand));
          }
        });
  for (auto [label, operand] :
       {std::pair{"/immediate", &five}, {"/shared heap", &heapX}})
    benchmark::RegisterBenchmark(
        std::string("gmp successor") + label, [=](benchmark::State &state) {
          for (auto _ : state) {
            std::int64_t result = 1;
            if (mpz_cmp_ui(operand->value, 2) >= 0) {
              mpz_t remainder;
              mpz_init(remainder);
              mpz_sub_ui(remainder, operand->value, 2);
              result = mpz_cmp_ui(remainder, 1000) > 0 ? 7 : 3;
              mpz_clear(remainder);
            }
            benchmark::DoNotOptimize(result);
          }
        });

  // match n { at_least 0 (k) => ...; at_most -1 (k) => ... } on a shared
  // negative scrutinee, and the same with a default arm that complements.
  for (auto [label, operand] :
       {std::pair{"/immediate", immediate(-5)},
        {"/shared heap", rcgmp_mpz_from_words(lhsWords.data(), 2, 1, 0)}})
    for (auto [name, match] : {std::pair{"int match", int_cases},
                               {"int match noti", int_cases_noti}})
      benchmark::RegisterBenchmark(std::string(name) + label,
                                   [=](benchmark::State &state) {
                                     for (auto _ : state) {
                                       retain(operand);
                                       benchmark::DoNotOptimize(match(operand));
                                     }
                                   });
}

// The runtime integer for a GMP value, with one reference.
void *fromGmp(mpz_srcptr value) {
  std::vector<std::uintptr_t> words(mpz_size(value));
  mpz_export(words.data(), nullptr, -1, sizeof(std::uintptr_t), 0, 0, value);
  return rcgmp_mpz_from_words(words.data(), words.size(), mpz_sgn(value) < 0,
                              0);
}

// Greatest common divisors of operands without and with a large common factor,
// compiled, by GMP, and by Euclid's algorithm from remainders, as code without
// gcdi computes them, with nonatomic counting only.
void registerGcd() {
  constexpr std::intptr_t factor = 1000003, first = 1234567, second = 7654321;
  static Integer smallX(smallLhs), smallY(smallRhs), commonX(factor * first),
      commonY(-factor * second), heapX(lhsWords), heapY(rhsWords),
      heapCommonX(lhsWords), heapCommonY(lhsWords), longX(0);
  mpz_mul_si(heapCommonX.value, heapCommonX.value, first);
  mpz_mul_si(heapCommonY.value, heapCommonY.value, -second);
  // Sixteen limbs against two, which the runtime divides first.
  for (std::uintptr_t word : {0x9e3779b97f4a7c15u, 0xbf58476d1ce4e5b9u,
                              0x94d049bb133111ebu, 0x2545f4914f6cdd1du})
    for (int repeat = 0; repeat < 4; ++repeat) {
      mpz_mul_2exp(longX.value, longX.value, 64);
      mpz_add_ui(longX.value, longX.value, word + repeat);
    }
  struct GcdCase {
    const char *label;
    mpz_srcptr x, y;
  };
  const std::array<GcdCase, 6> cases{{
      {"/immediates", smallX.value, smallY.value},
      {"/immediates, common factor", commonX.value, commonY.value},
      {"/shared heap, immediate", heapX.value, smallY.value},
      {"/shared heaps", heapX.value, heapY.value},
      {"/shared heaps, common factor", heapCommonX.value, heapCommonY.value},
      {"/shared heaps, 16 and 2 limbs", longX.value, heapY.value},
  }};
  for (const GcdCase &test : cases) {
    for (auto [name, op] : {std::pair{"gcdi", gcdi}, {"euclid", euclid}}) {
      Counted counted = countings[0];
      counted.binary = op;
      benchmark::RegisterBenchmark(
          std::string(name) + test.label, [=](benchmark::State &state) {
            binaryShared(state, counted, fromGmp(test.x), fromGmp(test.y));
          });
    }
    benchmark::RegisterBenchmark(std::string("gmp gcd") + test.label, gmpBinary,
                                 mpz_gcd, test.x, test.y);
  }
  // The greatest common divisor of x and a multiple of x is x, which a unique
  // x holds in place.
  static Integer multiple(lhsWords);
  mpz_mul_ui(multiple.value, multiple.value, 3);
  benchmark::RegisterBenchmark(
      "gcdi/unique heap, shared heap", [](benchmark::State &state) {
        void *x = heap(lhsWords), *y = fromGmp(multiple.value);
        for (auto _ : state) {
          retain(y);
          x = gcdi(x, y);
          benchmark::DoNotOptimize(x);
        }
        release(x);
        release(y);
      });
  benchmark::RegisterBenchmark("gmp gcd/unique heap, shared heap",
                               [](benchmark::State &state) {
                                 Integer x(lhsWords);
                                 for (auto _ : state) {
                                   mpz_gcd(x.value, x.value, multiple.value);
                                   benchmark::DoNotOptimize(x.value);
                                 }
                               });
}

// Sets value to count limbs of a fixed sequence, the highest with its top bit
// set.
void setLimbs(mpz_ptr value, std::size_t count, std::uint64_t seed) {
  mpz_set_ui(value, 0);
  for (std::size_t i = 0; i < count; ++i) {
    seed = seed * 6364136223846793005u + 1442695040888963407u;
    mpz_mul_2exp(value, value, 64);
    mpz_add_ui(value, value, i ? seed : seed | 1ul << 63);
  }
}

// Exact division of multiples, compiled with and without exact and by GMP's
// mpz_divexact and mpz_tdiv_q, with nonatomic counting only. Heap divisors
// are as long as their quotients, or longer, as in the least common multiple
// of operands with a large common factor.
void registerExactDivisions() {
  struct ExactCase {
    std::string label;
    mpz_srcptr x, y;
  };
  std::vector<ExactCase> cases;
  static std::deque<Integer> operands;
  auto multiple = [&](std::string label, mpz_srcptr quotient,
                      mpz_srcptr divisor) {
    Integer &x = operands.emplace_back(0), &y = operands.emplace_back(0);
    mpz_mul(x.value, quotient, divisor);
    mpz_set(y.value, divisor);
    cases.push_back({std::move(label), x.value, y.value});
  };
  Integer smallQ(smallLhs), smallY(-smallRhs), heapQ(lhsWords);
  multiple("/multiples, immediates", smallQ.value, smallY.value);
  multiple("/multiples, shared heap, immediate", heapQ.value, smallY.value);
  for (auto [quotientLimbs, divisorLimbs] :
       {std::pair{2u, 2u}, {4u, 4u}, {16u, 16u}, {2u, 16u}}) {
    Integer quotient(0), divisor(0);
    setLimbs(quotient.value, quotientLimbs, 1);
    setLimbs(divisor.value, divisorLimbs, 2);
    multiple("/multiples, shared heaps, " +
                 std::to_string(quotientLimbs + divisorLimbs) + " and " +
                 std::to_string(divisorLimbs) + " limbs",
             quotient.value, divisor.value);
  }
  for (const ExactCase &test : cases) {
    for (auto [name, op] :
         {std::pair{"divsi exact", divsi_exact}, {"divsi", divsi}}) {
      Counted counted = countings[0];
      counted.binary = op;
      benchmark::RegisterBenchmark(
          name + test.label, [=](benchmark::State &state) {
            binaryShared(state, counted, fromGmp(test.x), fromGmp(test.y));
          });
    }
    for (auto [name, op] :
         {std::pair{"gmp divexact", mpz_divexact}, {"gmp tdiv_q", mpz_tdiv_q}})
      benchmark::RegisterBenchmark(name + test.label, gmpBinary, op, test.x,
                                   test.y);
  }
  // Constant divisors, of which LLVM divides exact multiples by a shift and a
  // multiplication.
  static Integer smallMultiple(smallLhs * 24);
  for (auto [name, op] :
       {std::pair{"divsi exact/immediate by 12", divsi_exact_by_12},
        {"divsi/immediate by 12", divsi_by_12},
        {"divsi exact/immediate by 8", divsi_exact_by_8},
        {"divsi/immediate by 8", divsi_by_8}})
    benchmark::RegisterBenchmark(name, [=](benchmark::State &state) {
      for (auto _ : state) {
        void *x = immediate(smallLhs * 24);
        benchmark::DoNotOptimize(x);
        benchmark::DoNotOptimize(op(x));
      }
    });
  benchmark::RegisterBenchmark(
      "gmp divexact_ui/immediate by 12", [](benchmark::State &state) {
        for (auto _ : state) {
          mpz_t result;
          mpz_init(result);
          mpz_divexact_ui(result, smallMultiple.value, 12);
          benchmark::DoNotOptimize(result);
          mpz_clear(result);
        }
      });
  benchmark::RegisterBenchmark(
      "gmp tdiv_q_ui/immediate by 12", [](benchmark::State &state) {
        for (auto _ : state) {
          mpz_t result;
          mpz_init(result);
          mpz_tdiv_q_ui(result, smallMultiple.value, 12);
          benchmark::DoNotOptimize(result);
          mpz_clear(result);
        }
      });
}

[[maybe_unused]] const int registered =
    (registerBinaries(), registerMonus(), registerUnaries(), registerShifts(),
     registerRemainders(), registerPowers(), registerDivisions(),
     registerEuclidean(), registerLean(), registerTruncations(),
     registerLogarithms(), registerOthers(), registerGcd(),
     registerExactDivisions(), 0);

} // namespace
