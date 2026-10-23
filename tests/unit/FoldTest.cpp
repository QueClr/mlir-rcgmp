// Folds operations on constants and compares the folded constants with GMP.
#include "RCGMP/IR/RCGMPDialect.h"
#include "RCGMP/IR/RCGMPOps.h"
#include <fuzztest/fuzztest.h>
#include <gmp.h>
#include <gtest/gtest.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>
import rcgmp.testing;

namespace {

using namespace mlir;
using ::rcgmp::test::Integer;

APInt toAPInt(const Integer &value) {
  std::vector<std::uintptr_t> words = value.words();
  unsigned width = 64 * words.size() + 1;
  APInt magnitude(width, words.empty() ? std::vector<std::uint64_t>{0}
                                       : std::vector<std::uint64_t>(
                                             words.begin(), words.end()));
  return value.negative() ? -magnitude : magnitude;
}

Integer fromAPInt(const APInt &value, bool isUnsigned = false) {
  APInt wide = isUnsigned ? value.zext(value.getBitWidth() + 1)
                          : value.sext(value.getBitWidth() + 1);
  APInt magnitude = wide.abs();
  std::vector<std::uintptr_t> words(
      magnitude.getRawData(), magnitude.getRawData() + magnitude.getNumWords());
  return Integer::fromWords(words, wide.isNegative());
}

// The integer modulo 2^width, read as a signed or unsigned integer.
Integer wrapped(const Integer &value, unsigned width, bool isUnsigned) {
  Integer result;
  mpz_fdiv_r_2exp(result, value, width);
  if (!isUnsigned && mpz_tstbit(result, width - 1))
    mpz_sub(result, result, Integer::powerOfTwo(width));
  return result;
}

// Builds operations on constants in a fresh module, folding them as built.
class Folder {
public:
  Folder() : builder(&context()), module(ModuleOp::create(location())) {
    builder.setInsertionPointToStart(module->getBody());
  }

  Value mpz(const Integer &value) {
    return mlir::rcgmp::ConstantOp::create(builder, location(), mpzType(),
                                           toAPInt(value));
  }
  Value integer(const APInt &value) {
    return arith::ConstantOp::create(
        builder, location(),
        builder.getIntegerAttr(builder.getIntegerType(value.getBitWidth()),
                               value));
  }

  template <typename Op, typename... Arguments>
  SmallVector<Value> fold(Arguments &&...arguments) {
    SmallVector<Value> results;
    builder.createOrFold<Op>(results, location(),
                             std::forward<Arguments>(arguments)...);
    return results;
  }

  Type mpzType() { return mlir::rcgmp::MpzType::get(&context(), false); }
  IntegerType integerType(unsigned width) {
    return builder.getIntegerType(width);
  }

  static MLIRContext &context() {
    static MLIRContext *context = [] {
      auto *context = new MLIRContext;
      context->loadDialect<mlir::rcgmp::RCGMPDialect, arith::ArithDialect>();
      return context;
    }();
    return *context;
  }

private:
  static Location location() { return UnknownLoc::get(&context()); }

  OpBuilder builder;
  OwningOpRef<ModuleOp> module;
};

// The value of a folded mpz constant, or none if it was not folded.
std::optional<Integer> mpzValue(Value value) {
  if (auto constant = value.getDefiningOp<mlir::rcgmp::ConstantOp>())
    return fromAPInt(constant.getValue());
  return std::nullopt;
}

// The value of a folded integer constant, or none if it was not folded.
std::optional<APInt> integerValue(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    return cast<IntegerAttr>(constant.getValue()).getValue();
  return std::nullopt;
}

using Oracle = void (*)(mpz_ptr, mpz_srcptr, mpz_srcptr);
using UnaryOracle = void (*)(mpz_ptr, mpz_srcptr);

template <typename Op>
void checkBinary(const Integer &x, const Integer &y, Oracle oracle) {
  Folder folder;
  Integer expected;
  oracle(expected, x, y);
  EXPECT_EQ(mpzValue(folder.fold<Op>(folder.mpzType(), folder.mpz(x),
                                     folder.mpz(y))[0]),
            expected)
      << Op::getOperationName().str() << " " << x << ", " << y;
}

template <typename Op> void checkUnary(const Integer &x, UnaryOracle oracle) {
  Folder folder;
  Integer expected;
  oracle(expected, x);
  EXPECT_EQ(mpzValue(folder.fold<Op>(folder.mpzType(), folder.mpz(x))[0]),
            expected)
      << Op::getOperationName().str() << " " << x;
}

void checkShifts(const Integer &x, unsigned amount) {
  Folder folder;
  Value bits = folder.integer(APInt(64, amount));
  Integer left, right;
  mpz_mul_2exp(left, x, amount);
  mpz_fdiv_q_2exp(right, x, amount);
  EXPECT_EQ(mpzValue(folder.fold<mlir::rcgmp::ShLIOp>(folder.mpzType(),
                                                      folder.mpz(x), bits)[0]),
            left)
      << x << " << " << amount;
  EXPECT_EQ(mpzValue(folder.fold<mlir::rcgmp::ShRSIOp>(folder.mpzType(),
                                                       folder.mpz(x), bits)[0]),
            right)
      << x << " >> " << amount;
}

// Every division, which is not folded by zero.
void checkDivisions(const Integer &x, const Integer &y) {
  Folder folder;
  Integer tq, tr, fq, fr;
  bool zero = !y.sign();
  if (!zero) {
    mpz_tdiv_qr(tq, tr, x, y);
    mpz_fdiv_qr(fq, fr, x, y);
  }
  auto expect = [&](Value value, const Integer &expected, const char *name) {
    std::optional<Integer> folded = mpzValue(value);
    if (zero)
      EXPECT_FALSE(folded) << name << " by zero was folded";
    else
      EXPECT_EQ(folded, expected) << name << " " << x << ", " << y;
  };
  Type type = folder.mpzType();
  expect(
      folder.fold<mlir::rcgmp::DivSIOp>(type, folder.mpz(x), folder.mpz(y))[0],
      tq, "divsi");
  expect(
      folder.fold<mlir::rcgmp::RemSIOp>(type, folder.mpz(x), folder.mpz(y))[0],
      tr, "remsi");
  expect(folder.fold<mlir::rcgmp::FloorDivSIOp>(type, folder.mpz(x),
                                                folder.mpz(y))[0],
         fq, "floordivsi");
  expect(folder.fold<mlir::rcgmp::FloorModSIOp>(type, folder.mpz(x),
                                                folder.mpz(y))[0],
         fr, "floormodsi");
  auto both = folder.fold<mlir::rcgmp::DivRemSIOp>(type, type, folder.mpz(x),
                                                   folder.mpz(y));
  expect(both[0], tq, "divremsi quotient");
  expect(both[1], tr, "divremsi remainder");
  auto floored = folder.fold<mlir::rcgmp::FloorDivModSIOp>(
      type, type, folder.mpz(x), folder.mpz(y));
  expect(floored[0], fq, "floordivmodsi quotient");
  expect(floored[1], fr, "floordivmodsi remainder");
}

void checkComparisons(const Integer &x, const Integer &y) {
  Folder folder;
  int order = mpz_cmp(x, y);
  for (auto [predicate, expected] :
       {std::pair{mlir::rcgmp::CmpIPredicate::eq, order == 0},
        {mlir::rcgmp::CmpIPredicate::ne, order != 0},
        {mlir::rcgmp::CmpIPredicate::slt, order < 0},
        {mlir::rcgmp::CmpIPredicate::sle, order <= 0},
        {mlir::rcgmp::CmpIPredicate::sgt, order > 0},
        {mlir::rcgmp::CmpIPredicate::sge, order >= 0}}) {
    std::optional<APInt> folded = integerValue(folder.fold<mlir::rcgmp::CmpIOp>(
        predicate, folder.mpz(x), folder.mpz(y))[0]);
    ASSERT_TRUE(folded);
    EXPECT_EQ(folded->getBoolValue(), expected)
        << mlir::rcgmp::stringifyCmpIPredicate(predicate).str() << " " << x
        << ", " << y;
  }
}

// Truncations to width, and extensions from it of the truncated bits.
void checkConversions(const Integer &x, unsigned width) {
  Folder folder;
  Integer low = wrapped(x, width, false);
  std::optional<APInt> truncated =
      integerValue(folder.fold<mlir::rcgmp::TruncIOp>(folder.integerType(width),
                                                      folder.mpz(x))[0]);
  ASSERT_TRUE(truncated);
  EXPECT_EQ(fromAPInt(*truncated), low) << "trunci i" << width << " " << x;
  for (bool isUnsigned : {false, true}) {
    auto results = folder.fold<mlir::rcgmp::TruncIOverflowOp>(
        folder.integerType(width), folder.integerType(1), folder.mpz(x),
        isUnsigned);
    std::optional<APInt> value = integerValue(results[0]),
                         overflow = integerValue(results[1]);
    ASSERT_TRUE(value && overflow);
    EXPECT_EQ(fromAPInt(*value), low);
    EXPECT_EQ(overflow->getBoolValue(), wrapped(x, width, isUnsigned) != x)
        << "trunci_overflow i" << width << (isUnsigned ? " unsigned " : " ")
        << x;
  }
  Value bits = folder.integer(toAPInt(x).sextOrTrunc(width));
  EXPECT_EQ(
      mpzValue(folder.fold<mlir::rcgmp::ExtSIOp>(folder.mpzType(), bits)[0]),
      low)
      << "extsi i" << width << " of " << x;
  EXPECT_EQ(
      mpzValue(folder.fold<mlir::rcgmp::ExtUIOp>(folder.mpzType(), bits)[0]),
      wrapped(x, width, true))
      << "extui i" << width << " of " << x;
}

void FoldsMatchGmp(const std::pair<Integer, Integer> &operands, unsigned amount,
                   unsigned width) {
  auto &[x, y] = operands;
  checkBinary<mlir::rcgmp::AddIOp>(x, y, mpz_add);
  checkBinary<mlir::rcgmp::SubIOp>(x, y, mpz_sub);
  checkBinary<mlir::rcgmp::MulIOp>(x, y, mpz_mul);
  checkBinary<mlir::rcgmp::AndIOp>(x, y, mpz_and);
  checkBinary<mlir::rcgmp::OrIOp>(x, y, mpz_ior);
  checkBinary<mlir::rcgmp::XOrIOp>(x, y, mpz_xor);
  checkUnary<mlir::rcgmp::NotIOp>(x, mpz_com);
  checkUnary<mlir::rcgmp::NegIOp>(x, mpz_neg);
  checkUnary<mlir::rcgmp::AbsIOp>(x, mpz_abs);
  checkShifts(x, amount);
  checkDivisions(x, y);
  checkComparisons(x, y);
  checkConversions(x, width);
}
FUZZ_TEST(Fold, FoldsMatchGmp)
    .WithDomains(::rcgmp::test::AnyOperands(), fuzztest::InRange(0u, 400u),
                 fuzztest::InRange(1u, 200u));

} // namespace
