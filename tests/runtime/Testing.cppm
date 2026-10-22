module;
#include "RCGMP/Runtime/Mpz.h"
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gmp.h>

#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <ostream>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
export module rcgmp.testing;
import rcgmp.runtime;

export namespace rcgmp {

constexpr std::string_view name(Counting counting) {
  return counting == Counting::atomic ? "atomic" : "nonatomic";
}

template <typename Sink> void AbslStringify(Sink &sink, Counting counting) {
  sink.Append(name(counting));
}

} // namespace rcgmp

export namespace rcgmp::test {

constexpr std::array countings{Counting::nonatomic, Counting::atomic};

std::string countingName(const ::testing::TestParamInfo<Counting> &info) {
  return std::string(name(info.param));
}

// An owned GMP integer, the reference that runtime results are compared with.
class Integer {
public:
  Integer() { mpz_init(value); }
  explicit Integer(long integer) { mpz_init_set_si(value, integer); }
  Integer(const Integer &other) { mpz_init_set(value, other); }
  Integer(Integer &&other) noexcept : Integer() { mpz_swap(value, other); }
  Integer &operator=(Integer other) noexcept {
    mpz_swap(value, other);
    return *this;
  }
  ~Integer() { mpz_clear(value); }

  static Integer powerOfTwo(unsigned exponent) {
    Integer power;
    mpz_setbit(power, exponent);
    return power;
  }
  // words is a magnitude, least significant word first.
  static Integer fromWords(std::span<const std::uintptr_t> words,
                           bool negative) {
    Integer integer;
    mpz_import(integer, words.size(), -1, sizeof(std::uintptr_t), 0, 0,
               words.data());
    if (negative)
      mpz_neg(integer, integer);
    return integer;
  }

  operator mpz_ptr() { return value; }
  operator mpz_srcptr() const { return value; }

  int sign() const { return mpz_sgn(value); }
  bool negative() const { return sign() < 0; }
  // The magnitude, least significant word first.
  std::vector<std::uintptr_t> words() const {
    std::vector<std::uintptr_t> words(mpz_size(value));
    mpz_export(words.data(), nullptr, -1, sizeof(std::uintptr_t), 0, 0, value);
    return words;
  }
  bool fitsImmediate() const {
    return mpz_size(value) <= 1 &&
           Handle::immediate(mpz_getlimbn(value, 0), negative());
  }

  Integer operator-() const {
    Integer negation;
    mpz_neg(negation, value);
    return negation;
  }
  Integer operator+(long delta) const {
    Integer sum;
    if (delta < 0)
      mpz_sub_ui(sum, value, -static_cast<unsigned long>(delta));
    else
      mpz_add_ui(sum, value, delta);
    return sum;
  }
  Integer operator-(long delta) const {
    Integer difference;
    if (delta < 0)
      mpz_add_ui(difference, value, -static_cast<unsigned long>(delta));
    else
      mpz_sub_ui(difference, value, delta);
    return difference;
  }
  friend bool operator==(const Integer &lhs, const Integer &rhs) {
    return mpz_cmp(lhs, rhs) == 0;
  }

  // Hexadecimal, as -0x1f.
  std::string hex() const {
    std::string digits(mpz_sizeinbase(value, 16) + 2, '\0');
    mpz_get_str(digits.data(), 16, value);
    digits.resize(digits.find('\0'));
    return negative() ? "-0x" + digits.substr(1) : "0x" + digits;
  }
  friend std::ostream &operator<<(std::ostream &stream, const Integer &value) {
    return stream << value.hex();
  }
  template <typename Sink>
  friend void AbslStringify(Sink &sink, const Integer &value) {
    sink.Append(value.hex());
  }

private:
  mpz_t value;
};

using Oracle = void (*)(mpz_ptr, mpz_srcptr, mpz_srcptr);

// The canonical runtime integer for value. The result owns one reference.
void *toRuntime(const Integer &value) {
  return fromWords(value.words(), value.negative()).pointer();
}

// The value of a runtime integer, if its representation is canonical: an
// immediate 4*n + 1 exactly when n fits the immediate range, and otherwise a
// heap integer without leading zero limbs.
std::optional<Integer> decode(void *pointer) {
  Handle handle(pointer);
  if (handle.isImmediate()) {
    if ((std::bit_cast<std::uintptr_t>(pointer) & 3) != 1)
      return std::nullopt;
    return Integer(handle.value());
  }
  Mpz &heap = handle.heap();
  std::size_t count = heap.count();
  if (!count || count > heap.capacity)
    return std::nullopt;
  std::span<const Limb> magnitude = heap.magnitude();
  if (!magnitude.back() ||
      (count == 1 && Handle::immediate(magnitude[0], heap.negative())))
    return std::nullopt;
  return Integer::fromWords(magnitude, heap.negative());
}

std::uintptr_t references(void *pointer, Counting counting) {
  std::uintptr_t &references = Handle(pointer).heap().references;
  return counting == Counting::atomic
             ? std::atomic_ref<std::uintptr_t>(references).load()
             : references;
}

class RepresentsMatcher {
public:
  using is_gtest_matcher = void;

  explicit RepresentsMatcher(Integer expected)
      : expected(std::move(expected)) {}

  bool MatchAndExplain(void *handle, std::ostream *listener) const {
    std::optional<Integer> actual = decode(handle);
    if (listener) {
      if (actual)
        *listener << "which represents " << *actual;
      else
        *listener << "which is not canonical";
    }
    return actual == expected;
  }
  void DescribeTo(std::ostream *stream) const {
    *stream << "canonically represents " << expected;
  }
  void DescribeNegationTo(std::ostream *stream) const {
    *stream << "does not canonically represent " << expected;
  }

private:
  Integer expected;
};

// Matches a runtime integer that canonically represents expected.
RepresentsMatcher Represents(Integer expected) {
  return RepresentsMatcher(std::move(expected));
}

// Which operands the caller keeps an extra reference to. A shared operand
// must keep its value.
struct Sharing {
  bool lhs = false;
  bool rhs = false;
};

constexpr std::array<Sharing, 4> sharings{
    {{false, false}, {true, false}, {false, true}, {true, true}}};

// Applies a compiled binary operation to operands with the given sharing and
// compares the result with oracle.
void checkBinary(void *(*apply)(void *, void *), Oracle oracle,
                 const Integer &lhs, const Integer &rhs, Counting counting,
                 Sharing sharing) {
  bool atomic = counting == Counting::atomic;
  void *x = toRuntime(lhs), *y = toRuntime(rhs);
  if (sharing.lhs)
    rcgmp_mpz_retain(x, atomic);
  if (sharing.rhs)
    rcgmp_mpz_retain(y, atomic);
  Integer expected;
  oracle(expected, lhs, rhs);
  void *result = apply(x, y);
  EXPECT_THAT(result, Represents(expected))
      << "lhs = " << lhs << ", rhs = " << rhs;
  if (sharing.lhs) {
    EXPECT_THAT(x, Represents(lhs));
    rcgmp_mpz_release(x, atomic);
  }
  if (sharing.rhs) {
    EXPECT_THAT(y, Represents(rhs));
    rcgmp_mpz_release(y, atomic);
  }
  rcgmp_mpz_release(result, atomic);
}

// A predicate of rcgmp.cmpi and whether it holds when mpz_cmp returns order.
struct Predicate {
  std::string_view name;
  bool (*holds)(int order);
};

constexpr std::array<Predicate, 6> predicates{{
    {"eq", [](int order) { return order == 0; }},
    {"ne", [](int order) { return order != 0; }},
    {"slt", [](int order) { return order < 0; }},
    {"sle", [](int order) { return order <= 0; }},
    {"sgt", [](int order) { return order > 0; }},
    {"sge", [](int order) { return order >= 0; }},
}};

// Compiled comparisons in the order of predicates, each returning 1 when its
// predicate holds and 0 otherwise.
using Comparisons =
    std::array<std::int32_t (*)(void *, void *), predicates.size()>;

// The reference count of a heap operand.
std::optional<std::uintptr_t> count(void *operand, Counting counting) {
  if (Handle(operand).isImmediate())
    return std::nullopt;
  return references(operand, counting);
}

// Compares x and y, holding lhs and rhs, with every compiled predicate and with
// the runtime entry point, which also accepts immediates. Both operands are
// borrowed, so each keeps its value and reference count.
void checkComparison(const Comparisons &compiled, void *x, void *y,
                     const Integer &lhs, const Integer &rhs,
                     Counting counting) {
  SCOPED_TRACE(std::format("cmpi ({}): lhs = {}, rhs = {}", name(counting),
                           lhs.hex(), rhs.hex()));
  auto sign = [](int order) { return (order > 0) - (order < 0); };
  int order = mpz_cmp(lhs, rhs);
  std::optional<std::uintptr_t> xCount = count(x, counting),
                                yCount = count(y, counting);
  for (auto [predicate, compare] : std::views::zip(predicates, compiled))
    EXPECT_EQ(compare(x, y), predicate.holds(order) ? 1 : 0) << predicate.name;
  auto runtime =
      counting == Counting::atomic ? rcgmp_mpz_cmp_atomic : rcgmp_mpz_cmp;
  EXPECT_EQ(sign(runtime(x, y)), sign(order));
  EXPECT_THAT(x, Represents(lhs));
  EXPECT_THAT(y, Represents(rhs));
  EXPECT_EQ(count(x, counting), xCount);
  EXPECT_EQ(count(y, counting), yCount);
}

// How a derived operand relates to another x: as ±x + delta for
// cancellation, as ~x for complementary bits, or as ±2^(W-3) / x + delta,
// whose product with x lands near the immediate bound.
struct Relation {
  enum class Kind { offset, complement, quotient };

  Kind kind;
  bool negate;
  long delta;

  Integer operator()(const Integer &x) const {
    Integer y;
    switch (kind) {
    case Kind::offset:
      y = x;
      break;
    case Kind::complement:
      mpz_com(y, x);
      return y;
    case Kind::quotient:
      if (x.sign())
        mpz_tdiv_q(y, Integer::powerOfTwo(GMP_NUMB_BITS - 3), x);
      break;
    }
    return (negate ? -y : y) + delta;
  }

  template <typename Sink>
  friend void AbslStringify(Sink &sink, const Relation &relation) {
    constexpr std::array<std::string_view, 3> kinds{"offset", "complement",
                                                    "quotient"};
    sink.Append(kinds[std::to_underlying(relation.kind)]);
    sink.Append(relation.negate ? "(negated, " : "(");
    sink.Append(std::to_string(relation.delta) + ")");
  }
};

// Integers 2^e + delta for each exponent e and delta in [-4, 4], of either
// sign.
fuzztest::Domain<Integer> NearPowersOfTwo(std::vector<unsigned> exponents) {
  return fuzztest::Map(
      [](unsigned exponent, long delta, bool negative) {
        Integer value = Integer::powerOfTwo(exponent) + delta;
        return negative ? -value : value;
      },
      fuzztest::ElementOf(std::move(exponents)), fuzztest::InRange(-4L, 4L),
      fuzztest::Arbitrary<bool>());
}

// Integers of up to count words, of either sign. Words of all zeros or all
// ones make long runs, which stress carries, borrows, and sign extension.
fuzztest::Domain<Integer> AnyWords(std::size_t count) {
  return fuzztest::Map(
      [](const std::vector<std::uintptr_t> &words, bool negative) {
        return Integer::fromWords(words, negative);
      },
      fuzztest::VectorOf(fuzztest::OneOf(fuzztest::Arbitrary<std::uintptr_t>(),
                                         fuzztest::ElementOf<std::uintptr_t>(
                                             {0, ~std::uintptr_t{0}})))
          .WithMaxSize(count),
      fuzztest::Arbitrary<bool>());
}

// Integers at the edges that decide representation and carries: near the
// square roots of the immediate bound, the immediate bound, the signed word
// bound, and one to four limbs; small values; and up to five words.
fuzztest::Domain<Integer> AnyInteger() {
  constexpr unsigned word = GMP_NUMB_BITS;
  return fuzztest::OneOf(
      NearPowersOfTwo({(word - 3) / 2, (word - 2) / 2, word - 3, word - 1, word,
                       2 * word, 3 * word, 4 * word}),
      fuzztest::Map([](long value) { return Integer(value); },
                    fuzztest::InRange(-4L, 4L)),
      AnyWords(5));
}

fuzztest::Domain<Relation> AnyRelation() {
  using enum Relation::Kind;
  return fuzztest::StructOf<Relation>(
      fuzztest::ElementOf({offset, complement, quotient}),
      fuzztest::Arbitrary<bool>(), fuzztest::InRange(-2L, 2L));
}

// Independent operands, or an operand and another related to it, which
// independent operands rarely reach.
fuzztest::Domain<std::pair<Integer, Integer>> AnyOperands() {
  return fuzztest::OneOf(fuzztest::PairOf(AnyInteger(), AnyInteger()),
                         fuzztest::Map(
                             [](const Integer &x, const Relation &relation) {
                               return std::pair{x, relation(x)};
                             },
                             AnyInteger(), AnyRelation()));
}

fuzztest::Domain<Sharing> AnySharing() {
  return fuzztest::StructOf<Sharing>(fuzztest::Arbitrary<bool>(),
                                     fuzztest::Arbitrary<bool>());
}

fuzztest::Domain<Counting> AnyCounting() {
  return fuzztest::ElementOf({Counting::nonatomic, Counting::atomic});
}

} // namespace rcgmp::test
