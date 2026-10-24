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
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

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
RCGMP_ENTRY(std::int32_t, cmpi, void *, void *)
RCGMP_ENTRY(void *, extsi, std::int64_t)
RCGMP_ENTRY(void *, retain, void *)
RCGMP_ENTRY(void, release, void *)
#undef RCGMP_ENTRY
void *guarded_subi(void *, void *);
std::int64_t successor(void *);
std::int64_t int_cases(void *);
std::int64_t int_cases_noti(void *);
void *divsi(void *, void *);
void *remsi(void *, void *);
void *floordivsi(void *, void *);
void *floormodsi(void *, void *);
rcgmp_mpz_qr divremsi(void *, void *);
std::int64_t trunci(void *);
std::int64_t trunci_overflow(void *);
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

// Truncations to i64 of borrowed integers, and GMP reading the low word with
// and without a range check.
void registerTruncations() {
  static Integer smallX(smallLhs), heapX(lhsWords);
  for (auto [label, x, gmp] :
       {std::tuple{"/immediate", immediate(smallLhs), &smallX},
        {"/heap", heap(lhsWords), &heapX}}) {
    // Direct calls, as GMP's are.
    benchmark::RegisterBenchmark(std::string("trunci") + label,
                                 [=](benchmark::State &state) {
                                   for (auto _ : state) {
                                     benchmark::DoNotOptimize(x);
                                     benchmark::DoNotOptimize(trunci(x));
                                   }
                                 });
    benchmark::RegisterBenchmark(
        std::string("trunci_overflow") + label, [=](benchmark::State &state) {
          for (auto _ : state) {
            benchmark::DoNotOptimize(x);
            benchmark::DoNotOptimize(trunci_overflow(x));
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
                                   [=](benchmark::State &state) {
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

[[maybe_unused]] const int registered =
    (registerBinaries(), registerMonus(), registerUnaries(), registerShifts(),
     registerDivisions(), registerTruncations(), registerOthers(), 0);

} // namespace
