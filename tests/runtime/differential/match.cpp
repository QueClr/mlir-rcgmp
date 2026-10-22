// Compares compiled matches with the first arm that selects each integer,
// over the ownership of the scrutinee.
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <ranges>
#include <string_view>
#include <utility>
#include <vector>
import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.allocator;
import rcgmp.testing.differential;

// The index of the selected arm and its region argument, or zero.
struct MatchResult {
  std::int64_t arm;
  void *bound;
};

extern "C" {
MatchResult match_nat_local(void *);
MatchResult match_nat_atomic(void *);
MatchResult match_window_local(void *);
MatchResult match_window_atomic(void *);
MatchResult match_sparse_local(void *);
MatchResult match_sparse_atomic(void *);
MatchResult match_points_local(void *);
MatchResult match_points_atomic(void *);
MatchResult match_negative_local(void *);
MatchResult match_negative_atomic(void *);
MatchResult match_edges_local(void *);
MatchResult match_edges_atomic(void *);
MatchResult match_limbs_local(void *);
MatchResult match_limbs_atomic(void *);
MatchResult match_run_local(void *);
MatchResult match_run_atomic(void *);
MatchResult match_shadow_local(void *);
MatchResult match_shadow_atomic(void *);
std::int64_t index_table_local(void *);
std::int64_t index_table_atomic(void *);
std::int64_t index_hashed_local(void *);
std::int64_t index_hashed_atomic(void *);
std::int64_t index_hashed_wide_local(void *);
std::int64_t index_hashed_wide_atomic(void *);
std::int64_t index_mixed_local(void *);
std::int64_t index_mixed_atomic(void *);
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

enum class Kind { exact, atLeast, otherwise };

struct Arm {
  Kind kind;
  std::vector<std::string_view> values;
  // Whether the region takes an argument.
  bool argument;
};

// The entry points of matches whose arms yield only their index, so that
// selection may load it from a table, return no region argument.
template <std::int64_t (*entry)(void *)> MatchResult indexOnly(void *n) {
  return {entry(n), Handle::immediate(0).pointer()};
}

// The arms of each match in match.mlir, and its entry points indexed by
// counting.
struct Shape {
  std::string_view name;
  std::vector<Arm> arms;
  std::array<MatchResult (*)(void *), 2> compiled;
};

const std::vector<Shape> shapes{
    {"nat",
     {
         {Kind::exact, {"0"}, false},
         {Kind::exact, {"1"}, false},
         {Kind::atLeast, {"2"}, true},
     },
     {match_nat_local, match_nat_atomic}},
    {"window",
     {
         {Kind::exact, {"0"}, false},
         {Kind::exact, {"1", "3"}, false},
         {Kind::exact, {"2"}, false},
         {Kind::exact, {"5"}, false},
         {Kind::atLeast, {"7"}, true},
         {Kind::otherwise, {}, true},
     },
     {match_window_local, match_window_atomic}},
    {"sparse",
     {
         {Kind::exact, {"-1000"}, false},
         {Kind::exact, {"0"}, false},
         {Kind::exact, {"100"}, false},
         {Kind::exact, {"1000000"}, false},
         {Kind::atLeast, {"1099511627776"}, true},
         {Kind::otherwise, {}, true},
     },
     {match_sparse_local, match_sparse_atomic}},
    {"points",
     {
         {Kind::exact, {"3"}, false},
         {Kind::exact, {"9"}, false},
         {Kind::otherwise, {}, true},
     },
     {match_points_local, match_points_atomic}},
    {"negative",
     {
         {Kind::exact, {"-7"}, false},
         {Kind::atLeast, {"-5"}, true},
         {Kind::otherwise, {}, true},
     },
     {match_negative_local, match_negative_atomic}},
    {"edges",
     {
         {Kind::exact, {"2305843009213693951"}, false},
         {Kind::exact, {"-2305843009213693952"}, false},
         {Kind::atLeast, {"2305843009213693952"}, true},
         {Kind::exact, {"-2305843009213693953"}, false},
         {Kind::otherwise, {}, true},
     },
     {match_edges_local, match_edges_atomic}},
    {"limbs",
     {
         {Kind::exact, {"18446744073709551616", "18446744073709551617"}, false},
         {Kind::exact, {"36893488147419103232"}, false},
         {Kind::exact, {"55340232221128654853"}, false},
         {Kind::exact, {"-340282366920938463463374607431768211461"}, false},
         {Kind::atLeast, {"340282366920938463463374607431768211457"}, true},
         {Kind::atLeast, {"18446744073709551619"}, false},
         {Kind::otherwise, {}, true},
     },
     {match_limbs_local, match_limbs_atomic}},
    {"run",
     {
         {Kind::exact,
          {"2582249878086908589655919172003011874329705792829223512830659356540"
           "647622016841194629645353280137831435903171972747493383"},
          false},
         {Kind::exact,
          {"2582249878086908589655919172003011874329705792829223512830659356540"
           "647622016841194629645353280137831435903171972747493385"},
          false},
         {Kind::exact,
          {"-258224987808690858965591917200301187432970579282922351283065935654"
           "0647622016841194629645353280137831435903171972747493383"},
          false},
         {Kind::atLeast,
          {"4149515568880992958512407863691161151012446232242436899995657329690"
           "6528114129081463997070489471037942881978866113007891823951510754117"
           "75307886874834113963687061181803401509523685376"},
          true},
         {Kind::otherwise, {}, false},
     },
     {match_run_local, match_run_atomic}},
    {"shadow",
     {
         {Kind::atLeast, {"10"}, true},
         {Kind::exact, {"12", "3"}, false},
         {Kind::atLeast, {"5"}, true},
         {Kind::exact, {"-3"}, false},
     },
     {match_shadow_local, match_shadow_atomic}},
    {"table",
     {
         {Kind::exact, {"0"}, false},
         {Kind::exact, {"1"}, false},
         {Kind::exact, {"2"}, false},
         {Kind::exact, {"3"}, false},
         {Kind::exact, {"4"}, false},
         {Kind::exact, {"5"}, false},
         {Kind::exact, {"6"}, false},
         {Kind::exact, {"7"}, false},
         {Kind::exact, {"8"}, false},
         {Kind::exact, {"9"}, false},
         {Kind::atLeast, {"10"}, false},
         {Kind::otherwise, {}, false},
     },
     {indexOnly<index_table_local>, indexOnly<index_table_atomic>}},
    {"hashed",
     {
         {Kind::exact, {"-1000"}, false},
         {Kind::exact, {"0"}, false},
         {Kind::exact, {"7"}, false},
         {Kind::exact, {"100"}, false},
         {Kind::exact, {"1000"}, false},
         {Kind::exact, {"4096"}, false},
         {Kind::exact, {"20000"}, false},
         {Kind::exact, {"32768"}, false},
         {Kind::otherwise, {}, false},
     },
     {indexOnly<index_hashed_local>, indexOnly<index_hashed_atomic>}},
    {"hashed_wide",
     {
         {Kind::exact, {"-1000"}, false},
         {Kind::exact, {"0"}, false},
         {Kind::exact, {"7"}, false},
         {Kind::exact, {"65536"}, false},
         {Kind::exact, {"2147483648"}, false},
         {Kind::exact, {"1099511627776"}, false},
         {Kind::exact, {"-1125899906842624"}, false},
         {Kind::otherwise, {}, false},
     },
     {indexOnly<index_hashed_wide_local>, indexOnly<index_hashed_wide_atomic>}},
    {"mixed",
     {
         {Kind::exact, {"0"}, false},
         {Kind::exact, {"1", "3"}, false},
         {Kind::atLeast, {"5"}, false},
         {Kind::exact, {"2"}, false},
         {Kind::otherwise, {}, false},
     },
     {indexOnly<index_mixed_local>, indexOnly<index_mixed_atomic>}},
};

Integer parse(std::string_view digits) {
  Integer value;
  mpz_set_str(value, std::string(digits).c_str(), 10);
  return value;
}

// The first arm selecting value, if any.
std::optional<std::size_t> select(const Shape &shape, const Integer &value) {
  for (auto [index, arm] : std::views::enumerate(shape.arms)) {
    bool selected = arm.kind == Kind::otherwise ||
                    (arm.kind == Kind::atLeast &&
                     mpz_cmp(value, parse(arm.values[0])) >= 0) ||
                    std::ranges::any_of(arm.values, [&](std::string_view k) {
                      return arm.kind == Kind::exact && parse(k) == value;
                    });
    if (selected)
      return index;
  }
  return std::nullopt;
}

// A shared scrutinee has an extra reference held by the caller, which must
// survive with its value. An immortal heap scrutinee has a zero count.
void checkMatch(const Shape &shape, Counting counting, const Operand &operand,
                bool shared, bool immortal) {
  SCOPED_TRACE(std::format("{} ({}{}{}): {}", shape.name, rcgmp::name(counting),
                           shared ? ", shared" : "",
                           immortal ? ", immortal" : "", operand.describe()));
  const Integer &value = operand.value;
  // No arm selecting the scrutinee makes the behavior undefined.
  std::optional<std::size_t> expected = select(shape, value);
  if (!expected)
    return;
  const Arm &arm = shape.arms[*expected];
  std::size_t index = std::to_underlying(counting);
  AllocationCounter balance;
  void *x = makeOperand(operand, counting, immortal);
  bool heap = !Handle(x).isImmediate();
  if (shared)
    EXPECT_EQ(retains[index](x), x);
  AllocationCounter counter;
  MatchResult result = shape.compiled[index](x);
  EXPECT_EQ(result.arm, std::int64_t(*expected));
  Integer bound(0);
  if (arm.argument && arm.kind == Kind::atLeast)
    mpz_sub(bound, value, parse(arm.values[0]));
  else if (arm.argument)
    bound = value;
  EXPECT_THAT(result.bound, Represents(bound));
  // Subtracting a nonnegative threshold from a nonnegative, uniquely owned
  // heap scrutinee shrinks its magnitude in place.
  if (arm.argument && arm.kind == Kind::atLeast && heap && !immortal &&
      !shared && value.sign() >= 0 && parse(arm.values[0]).sign() >= 0)
    EXPECT_EQ(counter.allocations(), 0u)
        << "the remainder did not reuse the scrutinee";
  if (shared)
    releaseShared(counting, x, value, result.bound, immortal && heap);
  releases[index](result.bound);
  if (immortal && heap)
    dropImmortal(counting, x, value, false);
  EXPECT_TRUE(balance.balanced())
      << "allocations and deallocations do not balance";
}

// Integers near every pattern value and the immediate bounds, or anywhere.
fuzztest::Domain<Integer> AnyScrutinee(const Shape &shape) {
  std::vector<Integer> centers{Integer(0), Integer(Handle::immediateMax),
                               Integer(Handle::immediateMin)};
  for (const Arm &arm : shape.arms)
    for (std::string_view value : arm.values)
      centers.push_back(parse(value));
  return fuzztest::OneOf(
      fuzztest::Map(
          [](const Integer &center, long delta) { return center + delta; },
          fuzztest::ElementOf(centers), fuzztest::InRange(-2L, 2L)),
      AnyInteger());
}

void MatchesSelectFirstArm(const std::pair<std::size_t, Integer> &test,
                           Counting counting, std::size_t spare, bool shared,
                           bool immortal) {
  checkMatch(shapes[test.first], counting, {test.second, spare}, shared,
             immortal);
}
FUZZ_TEST(Differential, MatchesSelectFirstArm)
    .WithDomains(fuzztest::FlatMap(
                     [](std::size_t index) {
                       return fuzztest::PairOf(fuzztest::Just(index),
                                               AnyScrutinee(shapes[index]));
                     },
                     fuzztest::InRange<std::size_t>(0, shapes.size() - 1)),
                 AnyCounting(), AnySpare(), fuzztest::Arbitrary<bool>(),
                 fuzztest::Arbitrary<bool>());

// Every pattern value and its neighbors, in every ownership.
TEST(Differential, MatchesOnPatternBoundaries) {
  for (const Shape &shape : shapes) {
    std::vector<Integer> values{Integer(Handle::immediateMax),
                                Integer(Handle::immediateMin)};
    for (const Arm &arm : shape.arms)
      for (std::string_view value : arm.values)
        values.push_back(parse(value));
    for (const Integer &center : values)
      for (long delta : {-2L, -1L, 0L, 1L, 2L})
        for (Counting counting : countings)
          for (auto [shared, immortal] : std::views::cartesian_product(
                   std::array{false, true}, std::array{false, true}))
            checkMatch(shape, counting, {center + delta, 1}, shared, immortal);
  }
}

} // namespace
