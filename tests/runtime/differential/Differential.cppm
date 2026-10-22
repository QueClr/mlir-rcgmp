// The compiled entry points that the differential tests share, and the checks
// that compare binary operations and comparisons on them with GMP.
module;
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
#include <utility>
#include <vector>
export module rcgmp.testing.differential;
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
void *ori_local(void *, void *);
void *ori_atomic(void *, void *);
void *noti_local(void *);
void *noti_atomic(void *);
void *negi_local(void *);
void *negi_atomic(void *);
void *absi_local(void *);
void *absi_atomic(void *);
void *retain_local(void *);
void *retain_atomic(void *);
void release_local(void *);
void release_atomic(void *);
void *constant_local(std::size_t);
void *constant_atomic(std::size_t);
std::int32_t cmpi_eq_local(void *, void *);
std::int32_t cmpi_eq_atomic(void *, void *);
std::int32_t cmpi_ne_local(void *, void *);
std::int32_t cmpi_ne_atomic(void *, void *);
std::int32_t cmpi_slt_local(void *, void *);
std::int32_t cmpi_slt_atomic(void *, void *);
std::int32_t cmpi_sle_local(void *, void *);
std::int32_t cmpi_sle_atomic(void *, void *);
std::int32_t cmpi_sgt_local(void *, void *);
std::int32_t cmpi_sgt_atomic(void *, void *);
std::int32_t cmpi_sge_local(void *, void *);
std::int32_t cmpi_sge_atomic(void *, void *);
}

export namespace rcgmp::test {

constexpr unsigned word = GMP_NUMB_BITS;

// Compiled reference counting, indexed by counting. Operands are built by the
// runtime, so both implementations operate on the same objects.
constexpr std::array retains{retain_local, retain_atomic};
constexpr std::array releases{release_local, release_atomic};

// Compiled constants, selected by index, and their values in the order of
// constant.mlir.
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

// A compiled constant, by index.
struct Constant {
  std::size_t index;
};

fuzztest::Domain<std::size_t> AnyConstant() {
  return fuzztest::InRange<std::size_t>(0, constantValues.size() - 1);
}

enum class Operator { addi, subi, muli, andi, xori, ori };

// The compiled and runtime entry points of an operation, indexed by counting,
// and the GMP function it must agree with.
struct Operation {
  std::string_view name;
  std::array<void *(*)(void *, void *), 2> compiled;
  std::array<void *(*)(void *, void *, std::uint32_t), 2> runtime;
  Oracle oracle;
};

constexpr std::array<Operation, 6> operations{{
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
    {"ori",
     {ori_local, ori_atomic},
     {rcgmp_mpz_or, rcgmp_mpz_or_atomic},
     mpz_ior},
}};

constexpr std::array operators{Operator::addi, Operator::subi, Operator::muli,
                               Operator::andi, Operator::xori, Operator::ori};

const Operation &operation(Operator op) {
  return operations[std::to_underlying(op)];
}

template <typename Sink> void AbslStringify(Sink &sink, Operator op) {
  sink.Append(operation(op).name);
}

fuzztest::Domain<Operator> AnyOperator() {
  return fuzztest::ElementOf(std::vector(operators.begin(), operators.end()));
}

enum class UnaryOperator { noti, negi, absi };

// The compiled and runtime entry points of a unary operation, indexed by
// counting, and the GMP function it must agree with.
struct UnaryOperation {
  std::string_view name;
  std::array<void *(*)(void *), 2> compiled;
  std::array<void *(*)(void *, std::uint32_t), 2> runtime;
  UnaryOracle oracle;
};

constexpr std::array<UnaryOperation, 3> unaryOperations{{
    {"noti",
     {noti_local, noti_atomic},
     {rcgmp_mpz_not, rcgmp_mpz_not_atomic},
     mpz_com},
    {"negi",
     {negi_local, negi_atomic},
     {rcgmp_mpz_neg, rcgmp_mpz_neg_atomic},
     mpz_neg},
    {"absi",
     {absi_local, absi_atomic},
     {rcgmp_mpz_abs, rcgmp_mpz_abs_atomic},
     mpz_abs},
}};

constexpr std::array unaryOperators{UnaryOperator::noti, UnaryOperator::negi,
                                    UnaryOperator::absi};

const UnaryOperation &operation(UnaryOperator op) {
  return unaryOperations[std::to_underlying(op)];
}

template <typename Sink> void AbslStringify(Sink &sink, UnaryOperator op) {
  sink.Append(operation(op).name);
}

fuzztest::Domain<UnaryOperator> AnyUnaryOperator() {
  return fuzztest::ElementOf(
      std::vector(unaryOperators.begin(), unaryOperators.end()));
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

fuzztest::Domain<Operand> AnyOperand() {
  return fuzztest::StructOf<Operand>(AnyInteger(), AnySpare());
}

// An immortal heap operand keeps its spare limbs, which would let a counted one
// hold the result in place.
void *makeOperand(const Operand &operand, Counting counting,
                  bool immortal = false) {
  void *handle = toRuntime(operand.value);
  if (Handle(handle).isImmediate())
    return handle;
  Mpz &exact = Handle(handle).heap();
  Mpz &padded = allocate(exact.count() + operand.spare);
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

// The compiled constant or the runtime integer holding operand.
void *makeOperand(const Operand &operand, std::optional<std::size_t> constant,
                  Counting counting, bool immortal) {
  return constant ? constants[std::to_underlying(counting)](*constant)
                  : makeOperand(operand, counting, immortal);
}

// Drops the caller's reference to an operand made by makeOperand, or frees it
// if it is immortal. Compiled constants are static.
void dropOperand(void *operand, Counting counting, bool constant,
                 bool immortal) {
  if (constant || Handle(operand).isImmediate())
    return;
  if (immortal)
    rcgmp_mpz_free(operand, counting == Counting::atomic);
  else
    releases[std::to_underlying(counting)](operand);
}

// Which heap operands have a zero reference count, like compiled constants.
struct Immortality {
  bool lhs = false;
  bool rhs = false;
};

fuzztest::Domain<Immortality> AnyImmortality() {
  return fuzztest::StructOf<Immortality>(fuzztest::Arbitrary<bool>(),
                                         fuzztest::Arbitrary<bool>());
}

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
  // Calls the runtime entry point instead of compiled code, with this reuse
  // mask limited to the operands that compiled code could pass.
  std::optional<std::uint32_t> reuse = std::nullopt;

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
    std::string entry =
        reuse ? std::format(", runtime with reuse mask {}", *reuse) : "";
    return std::format("{} ({}, {}{}{}): lhs = {}, rhs = {}",
                       operation(op).name, name(counting), counted, immortality,
                       entry, lhs.describe(lhsConstant.has_value()),
                       rhs.describe(rhsConstant.has_value()));
  }
};

// Each operand position holding operand consumed one of its references, and the
// caller's reference kept its value. The result may be another reference to it.
// An immortal operand keeps its zero count.
void releaseShared(Counting counting, void *operand, const Integer &value,
                   void *result, bool immortal) {
  EXPECT_THAT(operand, Represents(value)) << "shared operand was modified";
  if (!Handle(operand).isImmediate())
    EXPECT_EQ(references(operand, counting),
              immortal ? 0u : 1u + (operand == result))
        << "operand positions did not each consume one reference";
  releases[std::to_underlying(counting)](operand);
}

// Immortal operands are neither counted nor written, even when the result is
// the same object. Those made here are then freed.
void dropImmortal(Counting counting, void *operand, const Integer &value,
                  bool constant) {
  EXPECT_THAT(operand, Represents(value)) << "immortal operand was modified";
  EXPECT_EQ(references(operand, counting), 0u)
      << "immortal operand was counted";
  if (!constant)
    rcgmp_mpz_free(operand, counting == Counting::atomic);
}

} // namespace rcgmp::test

namespace rcgmp::test {

// Complements the spare limbs of a heap operand made by makeOperand. Operands
// otherwise share one pattern of stale limbs, which would hide a comparison
// that reads past equally long magnitudes.
void complementSpare(void *operand) {
  if (Handle(operand).isImmediate())
    return;
  Mpz &heap = Handle(operand).heap();
  for (Limb &limb : heap.limbs().subspan(heap.count()))
    limb = ~limb;
}

} // namespace rcgmp::test

export namespace rcgmp::test {

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
  // Compiled code permits reusing each operand that is a uniquely owned,
  // counted heap integer.
  bool unique = !sharing.lhs && !sharing.rhs && !test.aliased;
  std::uint32_t uniqueness =
      (unique && !Handle(x).isImmediate() && !immortalX) |
      (unique && !Handle(y).isImmediate() && !immortalY) << 1;
  std::uint32_t reuse = test.reuse.value_or(3) & uniqueness;
  Integer expected;
  operation(test.op).oracle(expected, lhs, rhs);
  AllocationCounter counter;
  void *result = test.reuse ? operation(test.op).runtime[index](x, y, reuse)
                            : operation(test.op).compiled[index](x, y);
  EXPECT_THAT(result, Represents(expected));
  // General multiplication cannot write over its operands.
  bool reusable =
      test.op != Operator::muli || mpz_size(lhs) <= 1 || mpz_size(rhs) <= 1;
  if (test.roomy && reusable && reuse && !Handle(result).isImmediate())
    EXPECT_EQ(counter.allocations(), 0u)
        << "result did not reuse a unique operand";
  if (sharing.lhs)
    releaseShared(test.counting, x, lhs, result, immortalX);
  if (sharing.rhs)
    releaseShared(test.counting, y, rhs, result, immortalY);
  releases[index](result);
  if (immortalX)
    dropImmortal(test.counting, x, lhs, test.lhsConstant.has_value());
  if (immortalY && y != x)
    dropImmortal(test.counting, y, rhs, test.rhsConstant.has_value());
  EXPECT_TRUE(balance.balanced())
      << "allocations and deallocations do not balance";
}

// Compiled comparisons from compare.mlir, indexed by counting.
constexpr std::array<Comparisons, 2> comparisons{{
    {cmpi_eq_local, cmpi_ne_local, cmpi_slt_local, cmpi_sle_local,
     cmpi_sgt_local, cmpi_sge_local},
    {cmpi_eq_atomic, cmpi_ne_atomic, cmpi_slt_atomic, cmpi_sle_atomic,
     cmpi_sgt_atomic, cmpi_sge_atomic},
}};

// Comparisons borrow their operands, so the caller's ownership needs no
// variation beyond whether a heap operand is counted. An aliased rhs is the
// lhs itself, with no extra reference.
struct ComparisonCase {
  Counting counting;
  Immortality immortal;
  bool aliased;
  Operand lhs, rhs;
  // The compiled constants, by index, that hold the lhs or the rhs.
  std::optional<std::size_t> lhsConstant = std::nullopt,
                             rhsConstant = std::nullopt;

  std::string describe() const {
    std::string_view immortality = immortal.lhs && immortal.rhs
                                       ? ", immortal operands"
                                   : immortal.lhs ? ", immortal lhs"
                                   : immortal.rhs ? ", immortal rhs"
                                                  : "";
    return std::format("cmpi ({}{}{}): lhs = {}, rhs = {}", name(counting),
                       aliased ? ", aliased" : "", immortality,
                       lhs.describe(lhsConstant.has_value()),
                       rhs.describe(rhsConstant.has_value()));
  }
};

void checkComparison(const ComparisonCase &test) {
  SCOPED_TRACE(test.describe());
  AllocationCounter balance;
  void *x =
      makeOperand(test.lhs, test.lhsConstant, test.counting, test.immortal.lhs);
  void *y = test.aliased ? x
                         : makeOperand(test.rhs, test.rhsConstant,
                                       test.counting, test.immortal.rhs);
  if (!test.aliased && !test.rhsConstant)
    complementSpare(y);
  const Integer &rhs = test.aliased ? test.lhs.value : test.rhs.value;
  AllocationCounter counter;
  checkComparison(comparisons[std::to_underlying(test.counting)], x, y,
                  test.lhs.value, rhs, test.counting);
  EXPECT_EQ(counter.allocations(), 0u) << "comparison allocated";
  EXPECT_EQ(counter.deallocations(), 0u) << "comparison freed an operand";
  dropOperand(x, test.counting, test.lhsConstant.has_value(),
              test.immortal.lhs);
  if (!test.aliased)
    dropOperand(y, test.counting, test.rhsConstant.has_value(),
                test.immortal.rhs);
  EXPECT_TRUE(balance.balanced())
      << "allocations and deallocations do not balance";
}

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

} // namespace rcgmp::test
