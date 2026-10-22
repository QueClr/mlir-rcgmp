// Compares each compiled binary operation and integer construction with GMP,
// on boundary operands and on operands that FuzzTest generates.
#include "RCGMP/Runtime/Mpz.h"
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gmp.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>
import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.allocator;

extern "C" {
void *addi_local(void *, void *);
void *addi_atomic(void *, void *);
void *subi_local(void *, void *);
void *subi_atomic(void *, void *);
void *muli_local(void *, void *);
void *muli_atomic(void *, void *);
void *andi_local(void *, void *);
void *andi_atomic(void *, void *);
void *xori_local(void *, void *);
void *xori_atomic(void *, void *);
void *retain_local(void *);
void *retain_atomic(void *);
void release_local(void *);
void release_atomic(void *);
void *from_signed_i61_local(std::uint64_t);
void *from_unsigned_i61_local(std::uint64_t);
void *from_signed_i62_local(std::uint64_t);
void *from_unsigned_i62_local(std::uint64_t);
void *from_signed_i63_local(std::uint64_t);
void *from_unsigned_i63_local(std::uint64_t);
void *from_signed_i64_local(std::uint64_t);
void *from_unsigned_i64_local(std::uint64_t);
void *from_signed_i65_local(std::uint64_t, std::uint64_t);
void *from_unsigned_i65_local(std::uint64_t, std::uint64_t);
void *from_signed_i128_local(std::uint64_t, std::uint64_t);
void *from_unsigned_i128_local(std::uint64_t, std::uint64_t);
void *from_signed_i129_local(std::uint64_t, std::uint64_t, std::uint64_t);
void *from_unsigned_i129_local(std::uint64_t, std::uint64_t, std::uint64_t);
void *from_signed_i64_atomic(std::uint64_t);
void *from_unsigned_i64_atomic(std::uint64_t);
void *from_signed_i129_atomic(std::uint64_t, std::uint64_t, std::uint64_t);
void *from_unsigned_i129_atomic(std::uint64_t, std::uint64_t, std::uint64_t);
void *constant_local(std::size_t);
void *constant_atomic(std::size_t);
}

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

constexpr unsigned word = GMP_NUMB_BITS;

// Compiled reference counting, indexed by counting. Operands are built by the
// runtime, so both implementations operate on the same objects.
constexpr std::array retains{retain_local, retain_atomic};
constexpr std::array releases{release_local, release_atomic};

// Compiled constants, selected by index, and their values in the order of
// differential.mlir.
constexpr std::array constants{constant_local, constant_atomic};
constexpr std::array constantValues{
    "0",
    "-1",
    "2305843009213693951",
    "-2305843009213693952",
    "2305843009213693952",
    "-2305843009213693953",
    "9223372036854775807",
    "-9223372036854775808",
    "18446744073709551615",
    "-18446744073709551616",
    "340282366920938463463374607431768211455",
    "-340282366920938463463374607431768211457",
};

Integer constantValue(std::size_t index) {
  Integer value;
  mpz_set_str(value, constantValues[index], 10);
  return value;
}

fuzztest::Domain<std::size_t> AnyConstant() {
  return fuzztest::InRange<std::size_t>(0, constantValues.size() - 1);
}

enum class Operator { addi, subi, muli, andi, xori };

// The compiled and runtime entry points of an operation, indexed by counting,
// and the GMP function it must agree with.
struct Operation {
  std::string_view name;
  std::array<void *(*)(void *, void *), 2> compiled;
  std::array<void *(*)(void *, void *, std::uint32_t), 2> runtime;
  Oracle oracle;
};

constexpr std::array<Operation, 5> operations{{
    {"addi",
     {addi_local, addi_atomic},
     {rcgmp_mpz_add, rcgmp_mpz_add_atomic},
     mpz_add},
    {"subi",
     {subi_local, subi_atomic},
     {rcgmp_mpz_sub, rcgmp_mpz_sub_atomic},
     mpz_sub},
    {"muli",
     {muli_local, muli_atomic},
     {rcgmp_mpz_mul, rcgmp_mpz_mul_atomic},
     mpz_mul},
    {"andi",
     {andi_local, andi_atomic},
     {rcgmp_mpz_and, rcgmp_mpz_and_atomic},
     mpz_and},
    {"xori",
     {xori_local, xori_atomic},
     {rcgmp_mpz_xor, rcgmp_mpz_xor_atomic},
     mpz_xor},
}};

constexpr std::array operators{Operator::addi, Operator::subi, Operator::muli,
                               Operator::andi, Operator::xori};

const Operation &operation(Operator op) {
  return operations[std::to_underlying(op)];
}

template <typename Sink> void AbslStringify(Sink &sink, Operator op) {
  sink.Append(operation(op).name);
}

fuzztest::Domain<Operator> AnyOperator() {
  return fuzztest::ElementOf(std::vector(operators.begin(), operators.end()));
}

// A heap operand is allocated with spare limbs beyond its magnitude, as results
// of earlier operations often are, so that an operation may reuse it in place.
// A constant operand is the object that compiled code produces for the value,
// which has no spare limbs.
struct Operand {
  Integer value;
  std::size_t spare = 0;

  template <typename Sink>
  friend void AbslStringify(Sink &sink, const Operand &operand) {
    sink.Append(operand.describe());
  }
  std::string describe(bool constant = false) const {
    return value.fitsImmediate() ? std::format("{} (immediate)", value.hex())
           : constant ? std::format("{} (constant)", value.hex())
                      : std::format("{} ({} spare limbs)", value.hex(), spare);
  }
};

// With a one-limb operand, a result needs at most one limb more than the other
// operand, so a one-limb heap operand can hold it in place when its spare limbs
// cover the other operand. Generated operands have at most five limbs and
// derived ones at most six, so spares range up to six.
fuzztest::Domain<std::size_t> AnySpare() {
  return fuzztest::InRange<std::size_t>(0, 6);
}

// An immortal heap operand keeps its spare limbs, which would let a counted one
// hold the result in place.
void *makeOperand(const Operand &operand, Counting counting,
                  bool immortal = false) {
  void *handle = toRuntime(operand.value);
  if (Handle(handle).isImmediate())
    return handle;
  rcgmp::Mpz &exact = Handle(handle).heap();
  rcgmp::Mpz &padded = rcgmp::allocate(exact.count() + operand.spare);
  std::ranges::copy(exact.magnitude(), padded.limbs().begin());
  // Like the stale limbs of such results, the spare limbs are nonzero, so
  // reading past the magnitude changes the value.
  for (auto [index, limb] :
       std::views::enumerate(padded.limbs().subspan(exact.count())))
    limb = index % 2 ? GMP_NUMB_MAX / 0xff * 0xa5 : GMP_NUMB_MAX;
  padded.size = exact.size;
  if (immortal)
    padded.references = 0;
  rcgmp_mpz_release(handle, counting == Counting::atomic);
  return Handle(padded).pointer();
}

// Which heap operands have a zero reference count, like compiled constants.
struct Immortality {
  bool lhs = false;
  bool rhs = false;
};

// The sharing and immortality of a pair of operands.
struct Ownership {
  Sharing sharing;
  Immortality immortal;
};

constexpr std::array<Ownership, 7> ownerships{{
    {{false, false}, {false, false}},
    {{true, false}, {false, false}},
    {{false, true}, {false, false}},
    {{true, true}, {false, false}},
    {{false, false}, {true, false}},
    {{true, false}, {false, true}},
    {{false, false}, {true, true}},
}};

// Aliased operands share their sharing and immortality.
constexpr std::array<Ownership, 3> aliasOwnerships{{
    {{false, false}, {false, false}},
    {{true, false}, {false, false}},
    {{false, false}, {true, false}},
}};

// A shared operand has an extra reference held by the caller, which must
// survive with its value. An aliased rhs is a retained reference to the lhs.
// An immortal heap operand has a zero reference count, like a compiled
// constant.
struct BinaryCase {
  Operator op;
  Counting counting;
  Ownership ownership;
  bool aliased;
  Operand lhs, rhs;
  // Each heap operand has room for the result, so a unique one must hold it.
  bool roomy;
  // The compiled constants, by index, that hold the lhs or the rhs.
  std::optional<std::size_t> lhsConstant = std::nullopt,
                             rhsConstant = std::nullopt;

  std::string describe() const {
    auto [sharing, immortal] = ownership;
    std::string_view counted =
        aliased ? (sharing.lhs ? "aliased shared operand" : "aliased operands")
        : sharing.lhs && sharing.rhs ? "shared operands"
        : sharing.lhs                ? "shared lhs"
        : sharing.rhs                ? "shared rhs"
                                     : "unique operands";
    std::string_view immortality = immortal.lhs && immortal.rhs
                                       ? ", immortal operands"
                                   : immortal.lhs ? ", immortal lhs"
                                   : immortal.rhs ? ", immortal rhs"
                                                  : "";
    return std::format("{} ({}, {}{}): lhs = {}, rhs = {}", operation(op).name,
                       rcgmp::name(counting), counted, immortality,
                       lhs.describe(lhsConstant.has_value()),
                       rhs.describe(rhsConstant.has_value()));
  }
};

// The compiled constant or the runtime integer holding operand.
void *makeOperand(const Operand &operand, std::optional<std::size_t> constant,
                  Counting counting, bool immortal) {
  return constant ? constants[std::to_underlying(counting)](*constant)
                  : makeOperand(operand, counting, immortal);
}

// Each operand position holding operand consumed one of its references, and the
// caller's reference kept its value. The result may be another reference to it.
// An immortal operand keeps its zero count.
void releaseShared(const BinaryCase &test, void *operand, const Integer &value,
                   void *result, bool immortal) {
  EXPECT_THAT(operand, Represents(value)) << "shared operand was modified";
  if (!Handle(operand).isImmediate())
    EXPECT_EQ(references(operand, test.counting),
              immortal ? 0u : 1u + (operand == result))
        << "operand positions did not each consume one reference";
  releases[std::to_underlying(test.counting)](operand);
}

// Immortal operands are neither counted nor written, even when the result is
// the same object. Those made here are then freed.
void dropImmortal(const BinaryCase &test, void *operand, const Integer &value,
                  bool constant) {
  EXPECT_THAT(operand, Represents(value)) << "immortal operand was modified";
  EXPECT_EQ(references(operand, test.counting), 0u)
      << "immortal operand was counted";
  if (!constant)
    rcgmp_mpz_free(operand, test.counting == Counting::atomic);
}

void checkBinary(const BinaryCase &test) {
  SCOPED_TRACE(test.describe());
  std::size_t index = std::to_underlying(test.counting);
  auto [sharing, immortal] = test.ownership;
  const Integer &lhs = test.lhs.value;
  const Integer &rhs = test.aliased ? lhs : test.rhs.value;
  AllocationCounter balance;
  void *x =
      makeOperand(test.lhs, test.lhsConstant, test.counting, immortal.lhs);
  void *y = x;
  if (test.aliased)
    EXPECT_EQ(retains[index](x), x);
  else
    y = makeOperand(test.rhs, test.rhsConstant, test.counting, immortal.rhs);
  bool immortalX =
      !Handle(x).isImmediate() && (test.lhsConstant || immortal.lhs);
  bool immortalY =
      y == x ? immortalX
             : !Handle(y).isImmediate() && (test.rhsConstant || immortal.rhs);
  if (sharing.lhs)
    EXPECT_EQ(retains[index](x), x);
  if (sharing.rhs)
    EXPECT_EQ(retains[index](y), y);
  Integer expected;
  operation(test.op).oracle(expected, lhs, rhs);
  AllocationCounter counter;
  void *result = operation(test.op).compiled[index](x, y);
  EXPECT_THAT(result, Represents(expected));
  // General multiplication cannot write over its operands, and immortal ones
  // are never written.
  bool reusable =
      test.op != Operator::muli || mpz_size(lhs) <= 1 || mpz_size(rhs) <= 1;
  bool unique = !sharing.lhs && !sharing.rhs && !test.aliased;
  bool counted = (!Handle(x).isImmediate() && !immortalX) ||
                 (!Handle(y).isImmediate() && !immortalY);
  if (test.roomy && unique && reusable && counted &&
      !Handle(result).isImmediate())
    EXPECT_EQ(counter.allocations(), 0u)
        << "result did not reuse a unique operand";
  if (sharing.lhs)
    releaseShared(test, x, lhs, result, immortalX);
  if (sharing.rhs)
    releaseShared(test, y, rhs, result, immortalY);
  releases[index](result);
  if (immortalX)
    dropImmortal(test, x, lhs, test.lhsConstant.has_value());
  if (immortalY && y != x)
    dropImmortal(test, y, rhs, test.rhsConstant.has_value());
  EXPECT_TRUE(balance.balanced())
      << "allocations and deallocations do not balance";
}

void PairsMatchGmp(Operator op, Counting counting,
                   const std::pair<Integer, Integer> &operands,
                   std::size_t lhsSpare, std::size_t rhsSpare) {
  for (Ownership ownership : ownerships)
    checkBinary({op,
                 counting,
                 ownership,
                 false,
                 {operands.first, lhsSpare},
                 {operands.second, rhsSpare},
                 false});
}
FUZZ_TEST(Differential, PairsMatchGmp)
    .WithDomains(AnyOperator(), AnyCounting(), AnyOperands(), AnySpare(),
                 AnySpare());

void AliasesMatchGmp(Operator op, Counting counting, const Integer &value,
                     std::size_t spare) {
  for (Ownership ownership : aliasOwnerships)
    checkBinary(
        {op, counting, ownership, true, {value, spare}, {value, spare}, false});
}
FUZZ_TEST(Differential, AliasesMatchGmp)
    .WithDomains(AnyOperator(), AnyCounting(), AnyInteger(), AnySpare());

// A compiled constant replaces either operand.
void ConstantsMatchGmp(Operator op, Counting counting, std::size_t constant,
                       const Integer &other, std::size_t spare, bool left) {
  Operand c{constantValue(constant)}, y{other, spare};
  for (Ownership ownership : ownerships) {
    BinaryCase test{op, counting, ownership, false, c, y, false};
    if (left) {
      test.lhsConstant = constant;
    } else {
      std::swap(test.lhs, test.rhs);
      test.rhsConstant = constant;
    }
    checkBinary(test);
  }
}
FUZZ_TEST(Differential, ConstantsMatchGmp)
    .WithDomains(AnyOperator(), AnyCounting(), AnyConstant(), AnyInteger(),
                 AnySpare(), fuzztest::Arbitrary<bool>());

// A compiled constant, by index.
struct Constant {
  std::size_t index;
};

// A step of a chain applies an operation to the result so far and either a
// fresh operand, one related to the result so far, or a compiled constant.
struct Step {
  Operator op;
  std::variant<Integer, Relation, Constant> operand;
  std::size_t spare;
};

// Each chain feeds every result into the next operation as its uniquely owned
// lhs, so operands also carry the growth headroom and the stale limbs above the
// magnitude that the runtime leaves behind, and come from earlier reuse. A
// constant rhs may also come back as the result and become the next lhs.
void ChainsMatchGmp(Counting counting, const Operand &start,
                    const std::vector<Step> &steps) {
  std::size_t index = std::to_underlying(counting);
  AllocationCounter balance;
  Integer accumulator = start.value;
  void *result = makeOperand(start, counting);
  for (const auto &[position, step] : std::views::enumerate(steps)) {
    // Keep repeated products from growing without bound.
    if (step.op == Operator::muli && mpz_sizeinbase(accumulator, 2) > 1000)
      continue;
    auto *relation = std::get_if<Relation>(&step.operand);
    auto *constant = std::get_if<Constant>(&step.operand);
    Integer operand = relation   ? (*relation)(accumulator)
                      : constant ? constantValue(constant->index)
                                 : std::get<Integer>(step.operand);
    void *rhs = constant ? constants[index](constant->index)
                         : makeOperand({operand, step.spare}, counting);
    Integer expected;
    operation(step.op).oracle(expected, accumulator, operand);
    result = operation(step.op).compiled[index](result, rhs);
    ASSERT_THAT(result, Represents(expected))
        << operation(step.op).name << " at step " << position + 1 << " of "
        << steps.size() << ": lhs = " << accumulator << ", rhs = " << operand;
    accumulator = std::move(expected);
  }
  releases[index](result);
  EXPECT_TRUE(balance.balanced())
      << "allocations and deallocations do not balance";
}
FUZZ_TEST(Differential, ChainsMatchGmp)
    .WithDomains(
        AnyCounting(), fuzztest::StructOf<Operand>(AnyInteger(), AnySpare()),
        fuzztest::VectorOf(fuzztest::StructOf<Step>(
                               AnyOperator(),
                               fuzztest::VariantOf(
                                   AnyInteger(), AnyRelation(),
                                   fuzztest::StructOf<Constant>(AnyConstant())),
                               AnySpare()))
            .WithMinSize(8)
            .WithMaxSize(16));

// Zero, one, both sides of the immediate bounds, and one or two full limbs;
// the square roots of the immediate bound, whose products land exactly on it;
// and the signed word bound.
std::vector<Integer> boundaries() {
  std::vector<Integer> values{Integer(0), Integer(1), Integer(-1)};
  auto addPair = [&](unsigned exponent, long delta) {
    Integer value = Integer::powerOfTwo(exponent) + delta;
    values.push_back(value);
    values.push_back(-value);
  };
  for (unsigned exponent : {word - 3, word})
    for (long delta : {-1L, 0L, 1L})
      addPair(exponent, delta);
  for (unsigned exponent : {(word - 3) / 2, (word - 2) / 2, word - 1})
    addPair(exponent, 0);
  return values;
}

// Each compiled constant holds its value, and one outside the immediate range
// is the same static object with a zero count on every call.
TEST(Differential, ConstantsHoldTheirValues) {
  for (Counting counting : countings) {
    for (std::size_t i = 0; i < constantValues.size(); ++i) {
      SCOPED_TRACE(std::format("constant {} ({})", constantValues[i],
                               rcgmp::name(counting)));
      void *value = constants[std::to_underlying(counting)](i);
      EXPECT_THAT(value, Represents(constantValue(i)));
      if (!Handle(value).isImmediate()) {
        EXPECT_EQ(references(value, counting), 0u);
        EXPECT_EQ(constants[std::to_underlying(counting)](i), value);
      }
    }
  }
}

class DifferentialTest
    : public ::testing::TestWithParam<std::tuple<Operator, Counting>> {
protected:
  Operator op = std::get<Operator>(GetParam());
  Counting counting = std::get<Counting>(GetParam());
};

std::string parameterName(
    const ::testing::TestParamInfo<std::tuple<Operator, Counting>> &info) {
  auto [op, counting] = info.param;
  return std::string(operation(op).name) + "_" +
         std::string(rcgmp::name(counting));
}

INSTANTIATE_TEST_SUITE_P(Operations, DifferentialTest,
                         ::testing::Combine(::testing::ValuesIn(operators),
                                            ::testing::ValuesIn(countings)),
                         parameterName);

// No spare limbs, so that every growing result needs a new allocation, and
// enough for any result of two boundary operands to fit in either one.
TEST_P(DifferentialTest, MatchesGmpOnBoundaries) {
  std::vector<Integer> values = boundaries();
  for (std::size_t spare : {0, 4}) {
    for (const Integer &lhs : values) {
      Operand x{lhs, spare};
      for (const auto &[rhs, ownership] :
           std::views::cartesian_product(values, ownerships))
        checkBinary(
            {op, counting, ownership, false, x, {rhs, spare}, spare > 0});
      for (Ownership ownership : aliasOwnerships)
        checkBinary({op, counting, ownership, true, x, x, false});
    }
  }
}

// Constants against boundary operands on either side and each other.
TEST_P(DifferentialTest, MatchesGmpOnConstants) {
  std::vector<Integer> values = boundaries();
  for (std::size_t i = 0; i < constantValues.size(); ++i) {
    Operand c{constantValue(i)};
    for (std::size_t spare : {0, 4}) {
      for (const auto &[other, ownership] :
           std::views::cartesian_product(values, ownerships)) {
        Operand y{other, spare};
        checkBinary({op, counting, ownership, false, c, y, spare > 0, i});
        checkBinary(
            {op, counting, ownership, false, y, c, spare > 0, std::nullopt, i});
      }
    }
    for (std::size_t j = 0; j < constantValues.size(); ++j)
      for (Ownership ownership : ownerships)
        checkBinary({op,
                     counting,
                     ownership,
                     false,
                     c,
                     {constantValue(j)},
                     false,
                     i,
                     j});
    for (Ownership ownership : aliasOwnerships)
      checkBinary({op, counting, ownership, true, c, c, false, i});
  }
}

// Compiled code calls the runtime only when an operand is on the heap or an
// immediate result overflows, but every runtime entry point accepts any
// canonical operands and must return a canonical result.
TEST_P(DifferentialTest, RuntimeAcceptsImmediates) {
  std::vector<Integer> values = boundaries();
  for (const auto &[lhs, rhs] : std::views::cartesian_product(values, values)) {
    if (!lhs.fitsImmediate() || !rhs.fitsImmediate())
      continue;
    AllocationCounter balance;
    Integer expected;
    operation(op).oracle(expected, lhs, rhs);
    std::size_t index = std::to_underlying(counting);
    void *result =
        operation(op).runtime[index](toRuntime(lhs), toRuntime(rhs), 0);
    EXPECT_THAT(result, Represents(expected))
        << "lhs = " << lhs << ", rhs = " << rhs;
    releases[index](result);
    EXPECT_TRUE(balance.balanced());
  }
}

using Words = std::array<std::uint64_t, 3>;

template <typename> constexpr std::size_t parameterCount = 0;
template <typename Result, typename... Parameters>
constexpr std::size_t parameterCount<Result (*)(Parameters...)> =
    sizeof...(Parameters);

// Passes as many leading words as the entry point takes.
template <auto entry> void *passWords(const Words &words) {
  return [&]<std::size_t... index>(std::index_sequence<index...>) {
    return entry(words[index]...);
  }(std::make_index_sequence<parameterCount<decltype(entry)>>());
}

// Each entry point converts an operand of the given width, passed as
// ceil(width / 64) 64-bit words, least significant first. The widths cover
// each way of deciding whether a value is immediate, and operands of one, two,
// and three words.
struct FromInt {
  std::string_view name;
  unsigned width;
  bool isUnsigned;
  Counting counting;
  void *(*apply)(const Words &);
};

constexpr std::array<FromInt, 18> fromInts{{
    {"from_signed_i61", 61, false, Counting::nonatomic,
     passWords<from_signed_i61_local>},
    {"from_unsigned_i61", 61, true, Counting::nonatomic,
     passWords<from_unsigned_i61_local>},
    {"from_signed_i62", 62, false, Counting::nonatomic,
     passWords<from_signed_i62_local>},
    {"from_unsigned_i62", 62, true, Counting::nonatomic,
     passWords<from_unsigned_i62_local>},
    {"from_signed_i63", 63, false, Counting::nonatomic,
     passWords<from_signed_i63_local>},
    {"from_unsigned_i63", 63, true, Counting::nonatomic,
     passWords<from_unsigned_i63_local>},
    {"from_signed_i64", 64, false, Counting::nonatomic,
     passWords<from_signed_i64_local>},
    {"from_unsigned_i64", 64, true, Counting::nonatomic,
     passWords<from_unsigned_i64_local>},
    {"from_signed_i65", 65, false, Counting::nonatomic,
     passWords<from_signed_i65_local>},
    {"from_unsigned_i65", 65, true, Counting::nonatomic,
     passWords<from_unsigned_i65_local>},
    {"from_signed_i128", 128, false, Counting::nonatomic,
     passWords<from_signed_i128_local>},
    {"from_unsigned_i128", 128, true, Counting::nonatomic,
     passWords<from_unsigned_i128_local>},
    {"from_signed_i129", 129, false, Counting::nonatomic,
     passWords<from_signed_i129_local>},
    {"from_unsigned_i129", 129, true, Counting::nonatomic,
     passWords<from_unsigned_i129_local>},
    {"from_signed_i64", 64, false, Counting::atomic,
     passWords<from_signed_i64_atomic>},
    {"from_unsigned_i64", 64, true, Counting::atomic,
     passWords<from_unsigned_i64_atomic>},
    {"from_signed_i129", 129, false, Counting::atomic,
     passWords<from_signed_i129_atomic>},
    {"from_unsigned_i129", 129, true, Counting::atomic,
     passWords<from_unsigned_i129_atomic>},
}};

// Passes the low words of input, which the entry point truncates further to
// the operand width.
void checkFromInt(const FromInt &from, const Integer &input) {
  unsigned count = (from.width + 63) / 64;
  Integer expected;
  mpz_fdiv_r_2exp(expected, input, 64 * count);
  Words words{};
  mpz_export(words.data(), nullptr, -1, sizeof(std::uint64_t), 0, 0, expected);
  mpz_fdiv_r_2exp(expected, expected, from.width);
  // A set sign bit makes the ceiling remainder the value minus 2^width.
  if (!from.isUnsigned && mpz_tstbit(expected, from.width - 1))
    mpz_cdiv_r_2exp(expected, expected, from.width);
  AllocationCounter balance;
  void *result = from.apply(words);
  EXPECT_THAT(result, Represents(expected))
      << std::format("{} ({}): words = {:#x} {:#x} {:#x}", from.name,
                     rcgmp::name(from.counting), words[0], words[1], words[2]);
  releases[std::to_underlying(from.counting)](result);
  EXPECT_TRUE(balance.balanced());
}

// Values near the immediate bounds, word boundaries, and the operand width,
// with long runs of ones and zeros, and bits above the operand width.
fuzztest::Domain<Integer> AnyFromIntInput(unsigned width) {
  return fuzztest::OneOf(
      NearPowersOfTwo({word - 3, word - 2, word - 1, word, width - 1, width}),
      AnyWords((width + 63) / 64 + 1));
}

void FromIntMatchesGmp(const std::pair<std::size_t, Integer> &test) {
  checkFromInt(fromInts[test.first], test.second);
}
FUZZ_TEST(Differential, FromIntMatchesGmp)
    .WithDomains(fuzztest::FlatMap(
        [](std::size_t index) {
          return fuzztest::PairOf(fuzztest::Just(index),
                                  AnyFromIntInput(fromInts[index].width));
        },
        fuzztest::InRange<std::size_t>(0, fromInts.size() - 1)));

class FromIntTest : public ::testing::TestWithParam<FromInt> {};

INSTANTIATE_TEST_SUITE_P(Widths, FromIntTest, ::testing::ValuesIn(fromInts),
                         [](const ::testing::TestParamInfo<FromInt> &info) {
                           return std::string(info.param.name) + "_" +
                                  std::string(rcgmp::name(info.param.counting));
                         });

// Zero and one with either sign; both sides of the immediate bounds and of
// 2^(word-2), 2^(word-1), and 2^word; powers of two at two-word boundaries;
// and the operand width's signed minimum and maximum and unsigned maximum.
// Inputs wider than the operand are truncated.
TEST_P(FromIntTest, MatchesGmpOnBoundaries) {
  unsigned width = GetParam().width;
  struct Point {
    unsigned exponent;
    long delta;
    bool negate;
  };
  std::vector<Point> grid{
      {0, -1, false},          {0, 0, false},          {0, 0, true},
      {word - 3, -1, false},   {word - 3, 0, false},   {word - 3, 0, true},
      {word - 3, 1, true},     {word - 2, -1, false},  {word - 2, 0, false},
      {word - 1, -1, false},   {word - 1, 0, false},   {word, -1, false},
      {word, 0, false},        {word, 0, true},        {2 * word - 1, 0, false},
      {2 * word - 1, 0, true}, {2 * word, 0, false},   {2 * word, 0, true},
      {width - 1, 0, true},    {width - 1, -1, false}, {width, -1, false},
  };
  for (auto [exponent, delta, negate] : grid) {
    Integer input = Integer::powerOfTwo(exponent) + delta;
    checkFromInt(GetParam(), negate ? -input : input);
  }
}

} // namespace
