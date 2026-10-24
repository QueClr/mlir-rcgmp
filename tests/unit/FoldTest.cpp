// Folds operations on constants and compares the folded constants with GMP.
#include "RCGMP/IR/RCGMPDialect.h"
#include "RCGMP/IR/RCGMPOps.h"
#include "RCGMP/Runtime/Mpz.h"
#include <fuzztest/fuzztest.h>
#include <gmp.h>
#include <gtest/gtest.h>
#include <llvm/ADT/StringExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
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
  Value index(std::uint64_t value) {
    return arith::ConstantIndexOp::create(builder, location(), value);
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
  Type indexType() { return builder.getIndexType(); }

  FloatType floatType(unsigned width) {
    return width == 32 ? FloatType(builder.getF32Type()) : builder.getF64Type();
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

// A remainder by a power of two has at most 400 bits here, which folds.
void checkFloorMod2Exp(const Integer &x, unsigned amount) {
  Folder folder;
  Integer expected;
  mpz_fdiv_r_2exp(expected, x, amount);
  EXPECT_EQ(mpzValue(folder.fold<mlir::rcgmp::FloorMod2ExpIOp>(
                folder.mpzType(), folder.mpz(x),
                folder.integer(APInt(64, amount)))[0]),
            expected)
      << x << " modulo 2^" << amount;
}

// A power folds exactly when it has at most 4096 significant bits.
void checkPower(const Integer &x, unsigned exponent) {
  Folder folder;
  Integer expected;
  mpz_pow_ui(expected, x, exponent);
  std::optional<Integer> folded = mpzValue(folder.fold<mlir::rcgmp::IPowIOp>(
      folder.mpzType(), folder.mpz(x), folder.integer(APInt(64, exponent)))[0]);
  if (toAPInt(expected).getSignificantBits() <= 4096)
    EXPECT_EQ(folded, expected) << x << " to the power " << exponent;
  else
    EXPECT_FALSE(folded) << x << " to the power " << exponent << " was folded";
}

// Every division in each mode. A zero divisor folds only in lean mode, where
// the quotient is zero and the remainder x.
void checkDivisions(const Integer &x, const Integer &y) {
  using mlir::rcgmp::DivZero;
  Integer tq, tr = x, fq, fr = x, eq, er = x;
  bool zero = !y.sign();
  if (!zero) {
    mpz_tdiv_qr(tq, tr, x, y);
    mpz_fdiv_qr(fq, fr, x, y);
    ::rcgmp::test::gmpEuclidean(eq, er, x, y);
  }
  for (DivZero mode : {DivZero::trap, DivZero::undefined, DivZero::lean}) {
    Folder folder;
    auto expect = [&](Value value, const Integer &expected, const char *name) {
      std::optional<Integer> folded = mpzValue(value);
      std::string operation =
          std::string(name) + " " + mlir::rcgmp::stringifyDivZero(mode).str();
      if (zero && mode != DivZero::lean)
        EXPECT_FALSE(folded) << operation << " by zero was folded";
      else
        EXPECT_EQ(folded, expected) << operation << " " << x << ", " << y;
    };
    Type type = folder.mpzType();
    expect(folder.fold<mlir::rcgmp::DivSIOp>(type, folder.mpz(x), folder.mpz(y),
                                             mode)[0],
           tq, "divsi");
    expect(folder.fold<mlir::rcgmp::RemSIOp>(type, folder.mpz(x), folder.mpz(y),
                                             mode)[0],
           tr, "remsi");
    expect(folder.fold<mlir::rcgmp::FloorDivSIOp>(type, folder.mpz(x),
                                                  folder.mpz(y), mode)[0],
           fq, "floordivsi");
    expect(folder.fold<mlir::rcgmp::FloorModSIOp>(type, folder.mpz(x),
                                                  folder.mpz(y), mode)[0],
           fr, "floormodsi");
    auto both = folder.fold<mlir::rcgmp::DivRemSIOp>(type, type, folder.mpz(x),
                                                     folder.mpz(y), mode);
    expect(both[0], tq, "divremsi quotient");
    expect(both[1], tr, "divremsi remainder");
    auto floored = folder.fold<mlir::rcgmp::FloorDivModSIOp>(
        type, type, folder.mpz(x), folder.mpz(y), mode);
    expect(floored[0], fq, "floordivmodsi quotient");
    expect(floored[1], fr, "floordivmodsi remainder");
    expect(folder.fold<mlir::rcgmp::EDivSIOp>(type, folder.mpz(x),
                                              folder.mpz(y), mode)[0],
           eq, "edivsi");
    expect(folder.fold<mlir::rcgmp::EModSIOp>(type, folder.mpz(x),
                                              folder.mpz(y), mode)[0],
           er, "emodsi");
    auto euclidean = folder.fold<mlir::rcgmp::EDivModSIOp>(
        type, type, folder.mpz(x), folder.mpz(y), mode);
    expect(euclidean[0], eq, "edivmodsi quotient");
    expect(euclidean[1], er, "edivmodsi remainder");
  }
}

// Exact quotients in each mode fold when x is a multiple of y, and are
// otherwise undefined and left unfolded, as are zero divisors except in lean
// mode, where zero by zero folds to zero.
void checkExactQuotients(const Integer &x, const Integer &y) {
  using mlir::rcgmp::DivZero;
  Integer quotient;
  bool multiple = mpz_divisible_p(x, y);
  if (multiple && y.sign())
    mpz_divexact(quotient, x, y);
  for (DivZero mode : {DivZero::trap, DivZero::undefined, DivZero::lean}) {
    Folder folder;
    bool folds = multiple && (y.sign() || mode == DivZero::lean);
    auto expect = [&](Value value, const char *name) {
      std::string operation = std::string(name) + " exact " +
                              mlir::rcgmp::stringifyDivZero(mode).str() + " " +
                              x.hex() + ", " + y.hex();
      std::optional<Integer> folded = mpzValue(value);
      if (folds)
        EXPECT_EQ(folded, quotient) << operation;
      else
        EXPECT_FALSE(folded) << operation << " was folded";
    };
    Type type = folder.mpzType();
    expect(folder.fold<mlir::rcgmp::DivSIOp>(type, folder.mpz(x), folder.mpz(y),
                                             mode, true)[0],
           "divsi");
    expect(folder.fold<mlir::rcgmp::FloorDivSIOp>(type, folder.mpz(x),
                                                  folder.mpz(y), mode, true)[0],
           "floordivsi");
    expect(folder.fold<mlir::rcgmp::EDivSIOp>(type, folder.mpz(x),
                                              folder.mpz(y), mode, true)[0],
           "edivsi");
  }
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

// Logarithms of the magnitude, zero for zero, as a word and reduced modulo
// 2^width.
void checkLogs(const Integer &x, unsigned width) {
  Folder folder;
  Integer log(x.sign() ? static_cast<long>(mpz_sizeinbase(x, 2) - 1) : 0L);
  for (unsigned bits : {64u, width}) {
    std::optional<APInt> folded =
        integerValue(folder.fold<mlir::rcgmp::Log2IOp>(folder.integerType(bits),
                                                       folder.mpz(x))[0]);
    ASSERT_TRUE(folded);
    EXPECT_EQ(fromAPInt(*folded, true), wrapped(log, bits, true))
        << "log2i i" << bits << " " << x;
  }
}

// Bits of the two's complement, including those beyond the magnitude and at
// an index beyond 64 bits, which is the sign.
void checkBits(const Integer &x, unsigned index) {
  Folder folder;
  auto bit = [&](const APInt &index) {
    std::optional<APInt> folded =
        integerValue(folder.fold<mlir::rcgmp::TestBitIOp>(
            folder.mpz(x), folder.integer(index))[0]);
    return folded ? std::optional(folded->getBoolValue()) : std::nullopt;
  };
  EXPECT_EQ(bit(APInt(64, index)), mpz_tstbit(x, index) != 0)
      << x << " bit " << index;
  EXPECT_EQ(bit(APInt::getOneBitSet(128, 64) + index), x.negative())
      << x << " bit 2^64 + " << index;
}

// Limbs of the magnitude, as GMP counts and reads them, up to one beyond the
// count, which is zero.
void checkLimbs(const Integer &x) {
  Folder folder;
  Value value = folder.mpz(x);
  std::optional<APInt> count = integerValue(
      folder.fold<mlir::rcgmp::LimbCountOp>(folder.indexType(), value)[0]);
  ASSERT_TRUE(count) << x;
  EXPECT_EQ(count->getZExtValue(), mpz_size(x)) << x;
  for (std::size_t i = 0; i <= mpz_size(x); ++i) {
    std::optional<APInt> limb = integerValue(folder.fold<mlir::rcgmp::LimbOp>(
        folder.integerType(GMP_NUMB_BITS), value, folder.index(i))[0]);
    ASSERT_TRUE(limb) << x << " limb " << i;
    EXPECT_EQ(limb->getZExtValue(), mpz_getlimbn(x, i)) << x << " limb " << i;
  }
}

// Folded rounding matches the runtime.
void checkFloating(const Integer &x) {
  Folder folder;
  void *runtime = ::rcgmp::test::toRuntime(x);
  for (FloatType type : {folder.floatType(32), folder.floatType(64)}) {
    Value folded = folder.fold<mlir::rcgmp::SIToFPOp>(type, folder.mpz(x))[0];
    auto constant = folded.getDefiningOp<arith::ConstantOp>();
    ASSERT_TRUE(constant);
    APFloat value = cast<FloatAttr>(constant.getValue()).getValue();
    double expected =
        type.isF32() ? rcgmp_mpz_to_f32(runtime) : rcgmp_mpz_to_f64(runtime);
    EXPECT_EQ(type.isF32() ? value.convertToFloat() : value.convertToDouble(),
              expected)
        << "sitofp " << x;
  }
  rcgmp_mpz_release(runtime, 0);
}

void FoldsMatchGmp(const std::pair<Integer, Integer> &operands, unsigned amount,
                   unsigned width) {
  auto &[x, y] = operands;
  checkBinary<mlir::rcgmp::AddIOp>(x, y, mpz_add);
  checkBinary<mlir::rcgmp::SubIOp>(x, y, mpz_sub);
  checkBinary<mlir::rcgmp::MonusIOp>(x, y, ::rcgmp::test::gmpMonus);
  checkBinary<mlir::rcgmp::MulIOp>(x, y, mpz_mul);
  checkBinary<mlir::rcgmp::GcdIOp>(x, y, mpz_gcd);
  checkBinary<mlir::rcgmp::AndIOp>(x, y, mpz_and);
  checkBinary<mlir::rcgmp::OrIOp>(x, y, mpz_ior);
  checkBinary<mlir::rcgmp::XOrIOp>(x, y, mpz_xor);
  checkUnary<mlir::rcgmp::NotIOp>(x, mpz_com);
  checkUnary<mlir::rcgmp::NegIOp>(x, mpz_neg);
  checkUnary<mlir::rcgmp::AbsIOp>(x, mpz_abs);
  checkShifts(x, amount);
  checkFloorMod2Exp(x, amount);
  checkPower(x, amount);
  checkDivisions(x, y);
  checkComparisons(x, y);
  checkConversions(x, width);
  checkLogs(x, width);
  checkBits(x, amount);
  checkLimbs(x);
  checkFloating(x);
}
FUZZ_TEST(Fold, FoldsMatchGmp)
    .WithDomains(::rcgmp::test::AnyOperands(), fuzztest::InRange(0u, 400u),
                 fuzztest::InRange(1u, 200u));

// Operands with common factors, whose greatest common divisor is not one.
void GcdFoldsMatchGmp(const std::pair<Integer, Integer> &operands) {
  checkBinary<mlir::rcgmp::GcdIOp>(operands.first, operands.second, mpz_gcd);
}
FUZZ_TEST(Fold, GcdFoldsMatchGmp).WithDomains(::rcgmp::test::AnyGcdOperands());

// Multiples x = q y, of which arbitrary operands are rarely any, and the same
// pairs with x off by one, which are not multiples unless y is one.
void ExactFoldsMatchGmp(const std::pair<Integer, Integer> &operands) {
  auto &[q, y] = operands;
  Integer x;
  mpz_mul(x, q, y);
  checkExactQuotients(x, y);
  checkExactQuotients(x + 1, y);
}
FUZZ_TEST(Fold, ExactFoldsMatchGmp).WithDomains(::rcgmp::test::AnyOperands());

// Operands past 4096 bits stay computed at run time, even when the result is
// small.
TEST(Fold, KeepsGcdOfWideOperands) {
  Folder folder;
  Value wide = folder.mpz(Integer::powerOfTwo(5000));
  EXPECT_EQ(mpzValue(folder.fold<mlir::rcgmp::GcdIOp>(
                folder.mpzType(), wide, folder.mpz(Integer(6)))[0]),
            std::nullopt);
}

// A nonnegative integer below the power is its own remainder, however wide
// either is. That of a negative integer has as many bits as the amount, and
// folds only up to 4096 of them.
TEST(Fold, FoldsRemaindersByWideAmounts) {
  Folder folder;
  Integer wide = Integer::powerOfTwo(5000) + 1, expected;
  for (auto [value, amount, folds] :
       {std::tuple{wide, APInt::getOneBitSet(128, 64), true},
        {wide, APInt(64, 5001), true},
        {Integer(-1), APInt::getOneBitSet(128, 64), false},
        {Integer(-1), APInt(64, 4095), true},
        {Integer(-1), APInt(64, 4096), false},
        {-wide, APInt(64, 5000), false}}) {
    std::optional<Integer> folded =
        mpzValue(folder.fold<mlir::rcgmp::FloorMod2ExpIOp>(
            folder.mpzType(), folder.mpz(value), folder.integer(amount))[0]);
    std::string power = "2^" + llvm::toString(amount, 10, false);
    if (!folds) {
      EXPECT_FALSE(folded) << value << " modulo " << power << " was folded";
      continue;
    }
    // An amount beyond 64 bits saturates for GMP, which leaves the
    // nonnegative integers here unchanged as well.
    mpz_fdiv_r_2exp(expected, value, amount.getLimitedValue());
    EXPECT_EQ(folded, expected) << value << " modulo " << power;
  }
}

// The parity of an exponent wider than 64 bits decides the sign of a power of
// -1, and other bases of magnitude at least two have powers too large to fold.
TEST(Fold, FoldsPowersByWideExponents) {
  Folder folder;
  APInt even = APInt::getOneBitSet(128, 64);
  for (auto [base, exponent, expected] : {std::tuple{-1L, even, 1L},
                                          {-1L, even + 1, -1L},
                                          {0L, even, 0L},
                                          {1L, even + 1, 1L}})
    EXPECT_EQ(mpzValue(folder.fold<mlir::rcgmp::IPowIOp>(
                  folder.mpzType(), folder.mpz(Integer(base)),
                  folder.integer(exponent))[0]),
              Integer(expected));
  EXPECT_FALSE(mpzValue(folder.fold<mlir::rcgmp::IPowIOp>(
      folder.mpzType(), folder.mpz(Integer(2)), folder.integer(even))[0]));
}

} // namespace
