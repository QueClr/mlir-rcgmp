#include "RCGMP/IR/RCGMPOps.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/StringExtras.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>

namespace mlir::rcgmp {

static IntegerAttr getNarrowestAttr(Builder &builder, const APInt &value) {
  unsigned width = value.getSignificantBits();
  return builder.getIntegerAttr(builder.getIntegerType(width),
                                value.sextOrTrunc(width));
}

// Every value, zero included, needs at least one bit.
static bool isNarrowest(IntegerAttr attr) {
  auto type = dyn_cast<IntegerType>(attr.getType());
  return type && type.isSignless() && type.getWidth() != 0 &&
         type.getWidth() == attr.getValue().getSignificantBits();
}

static ParseResult parseSignedInteger(OpAsmParser &parser, IntegerAttr &attr) {
  auto loc = parser.getCurrentLocation();
  // The generic integer parser also accepts the boolean keywords.
  StringRef keyword;
  if (succeeded(parser.parseOptionalKeyword(&keyword, {"true", "false"})))
    return parser.emitError(loc, "expected integer value");
  APInt value;
  OptionalParseResult result = parser.parseOptionalInteger(value);
  if (!result.has_value())
    return parser.emitError(loc, "expected integer value");
  if (failed(*result))
    return failure();
  if (value.getSignificantBits() > IntegerType::kMaxWidth)
    return parser.emitError(loc, "integer value is too wide");
  attr = getNarrowestAttr(parser.getBuilder(), value);
  return success();
}

static void printSignedInteger(OpAsmPrinter &printer, Operation *,
                               IntegerAttr attr) {
  attr.getValue().print(printer.getStream(), /*isSigned=*/true);
}

} // namespace mlir::rcgmp

#include "RCGMP/IR/RCGMPEnums.cpp.inc"

#define GET_OP_CLASSES
#include "RCGMP/IR/RCGMPOps.cpp.inc"

namespace mlir::rcgmp {

void ConstantOp::build(OpBuilder &builder, OperationState &state, Type type,
                       const APInt &value) {
  build(builder, state, type, getNarrowestAttr(builder, value));
}

//===----------------------------------------------------------------------===//
// Folding
//===----------------------------------------------------------------------===//

// A result of more bits stays computed at run time: its constant would be a
// large static object, and computing it could take long.
static constexpr unsigned maxFoldedBits = 4096;

// The value of a constant mpz operand, which is signed.
static std::optional<APInt> mpzOperand(Attribute attribute) {
  if (auto value = dyn_cast_or_null<MpzValueAttr>(attribute))
    return value.getValue();
  return std::nullopt;
}

// The bits of a constant fixed-width integer operand.
static std::optional<APInt> integerOperand(Attribute attribute) {
  if (auto integer = dyn_cast_or_null<IntegerAttr>(attribute))
    return integer.getValue();
  return std::nullopt;
}

// The constant mpz value, in its narrowest width, unless it is too wide to
// fold.
static OpFoldResult foldedMpz(MLIRContext *context, const APInt &value) {
  unsigned width = value.getSignificantBits();
  if (width > maxFoldedBits)
    return {};
  return MpzValueAttr::get(context, value.sextOrTrunc(width));
}

// Sign-extends both values to one width with extra bits beyond the wider.
static std::pair<APInt, APInt> widened(const APInt &a, const APInt &b,
                                       unsigned extra) {
  unsigned width = std::max(a.getBitWidth(), b.getBitWidth()) + extra;
  return {a.sext(width), b.sext(width)};
}

// Folds an operation on two constant mpz operands.
template <typename Function>
static OpFoldResult foldBinary(Operation *op, ArrayRef<Attribute> operands,
                               Function function) {
  std::optional<APInt> lhs = mpzOperand(operands[0]),
                       rhs = mpzOperand(operands[1]);
  if (!lhs || !rhs)
    return {};
  std::optional<APInt> result = function(*lhs, *rhs);
  return result ? foldedMpz(op->getContext(), *result) : OpFoldResult();
}

OpFoldResult ConstantOp::fold(FoldAdaptor) {
  return MpzValueAttr::get(getContext(), getValue());
}

OpFoldResult ExtSIOp::fold(FoldAdaptor adaptor) {
  std::optional<APInt> value = integerOperand(adaptor.getValue());
  return value ? foldedMpz(getContext(), *value) : OpFoldResult();
}

OpFoldResult ExtUIOp::fold(FoldAdaptor adaptor) {
  std::optional<APInt> value = integerOperand(adaptor.getValue());
  return value ? foldedMpz(getContext(), value->zext(value->getBitWidth() + 1))
               : OpFoldResult();
}

OpFoldResult AddIOp::fold(FoldAdaptor adaptor) {
  return foldBinary(*this, adaptor.getOperands(),
                    [](const APInt &lhs, const APInt &rhs) {
                      auto [a, b] = widened(lhs, rhs, 1);
                      return std::optional(a + b);
                    });
}

OpFoldResult SubIOp::fold(FoldAdaptor adaptor) {
  return foldBinary(*this, adaptor.getOperands(),
                    [](const APInt &lhs, const APInt &rhs) {
                      auto [a, b] = widened(lhs, rhs, 1);
                      return std::optional(a - b);
                    });
}

OpFoldResult MonusIOp::fold(FoldAdaptor adaptor) {
  return foldBinary(*this, adaptor.getOperands(),
                    [](const APInt &lhs, const APInt &rhs) {
                      auto [a, b] = widened(lhs, rhs, 1);
                      APInt difference = a - b;
                      if (difference.isNegative())
                        difference.clearAllBits();
                      return std::optional(difference);
                    });
}

OpFoldResult MulIOp::fold(FoldAdaptor adaptor) {
  return foldBinary(
      *this, adaptor.getOperands(),
      [](const APInt &lhs, const APInt &rhs) -> std::optional<APInt> {
        // A product of nonzero integers has at least one bit less than its
        // operands together.
        unsigned bits = lhs.getSignificantBits() + rhs.getSignificantBits();
        if (!lhs.isZero() && !rhs.isZero() && bits > maxFoldedBits + 1)
          return std::nullopt;
        unsigned width = lhs.getBitWidth() + rhs.getBitWidth();
        return lhs.sext(width) * rhs.sext(width);
      });
}

OpFoldResult GcdIOp::fold(FoldAdaptor adaptor) {
  return foldBinary(
      *this, adaptor.getOperands(),
      [](const APInt &lhs, const APInt &rhs) -> std::optional<APInt> {
        // The time to compute grows with the square of the width.
        if (lhs.getSignificantBits() > maxFoldedBits ||
            rhs.getSignificantBits() > maxFoldedBits)
          return std::nullopt;
        // One extra bit holds the absolute value of either minimum, which
        // bounds the nonnegative result.
        auto [a, b] = widened(lhs, rhs, 1);
        return llvm::APIntOps::GreatestCommonDivisor(a.abs(), b.abs());
      });
}

// Constant operands of a commutative operation sort to the right. With zero,
// the greatest common divisor is the absolute value of the other operand, and
// with one or minus one it is one, so neither computes it.
static LogicalResult simplifyGcdWithConstant(GcdIOp op,
                                             PatternRewriter &rewriter) {
  auto constant = op.getRhs().getDefiningOp<ConstantOp>();
  if (!constant)
    return failure();
  APInt value = constant.getValue();
  if (value.isZero()) {
    rewriter.replaceOpWithNewOp<AbsIOp>(op, op.getType(), op.getLhs());
    return success();
  }
  if (!value.sext(value.getBitWidth() + 1).abs().isOne())
    return failure();
  ReleaseOp::create(rewriter, op.getLoc(), op.getLhs());
  rewriter.replaceOpWithNewOp<ConstantOp>(op, op.getType(), APInt(2, 1));
  return success();
}

void GcdIOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *) {
  results.add(simplifyGcdWithConstant);
}

OpFoldResult AndIOp::fold(FoldAdaptor adaptor) {
  return foldBinary(*this, adaptor.getOperands(),
                    [](const APInt &lhs, const APInt &rhs) {
                      auto [a, b] = widened(lhs, rhs, 0);
                      return std::optional(a & b);
                    });
}

OpFoldResult OrIOp::fold(FoldAdaptor adaptor) {
  return foldBinary(*this, adaptor.getOperands(),
                    [](const APInt &lhs, const APInt &rhs) {
                      auto [a, b] = widened(lhs, rhs, 0);
                      return std::optional(a | b);
                    });
}

OpFoldResult XOrIOp::fold(FoldAdaptor adaptor) {
  return foldBinary(*this, adaptor.getOperands(),
                    [](const APInt &lhs, const APInt &rhs) {
                      auto [a, b] = widened(lhs, rhs, 0);
                      return std::optional(a ^ b);
                    });
}

OpFoldResult NotIOp::fold(FoldAdaptor adaptor) {
  std::optional<APInt> value = mpzOperand(adaptor.getOperand());
  return value ? foldedMpz(getContext(), ~*value) : OpFoldResult();
}

OpFoldResult NegIOp::fold(FoldAdaptor adaptor) {
  std::optional<APInt> value = mpzOperand(adaptor.getOperand());
  return value ? foldedMpz(getContext(), -value->sext(value->getBitWidth() + 1))
               : OpFoldResult();
}

OpFoldResult AbsIOp::fold(FoldAdaptor adaptor) {
  std::optional<APInt> value = mpzOperand(adaptor.getOperand());
  return value ? foldedMpz(getContext(),
                           value->sext(value->getBitWidth() + 1).abs())
               : OpFoldResult();
}

// The shift amount is unsigned, and one beyond 64 bits exceeds any limit.
static std::optional<std::uint64_t> shiftAmount(Attribute attribute) {
  std::optional<APInt> amount = integerOperand(attribute);
  if (!amount)
    return std::nullopt;
  return amount->getActiveBits() > 64 ? UINT64_MAX : amount->getZExtValue();
}

OpFoldResult ShLIOp::fold(FoldAdaptor adaptor) {
  std::optional<APInt> value = mpzOperand(adaptor.getValue());
  std::optional<std::uint64_t> amount = shiftAmount(adaptor.getAmount());
  if (!value || !amount)
    return {};
  if (value->isZero())
    return foldedMpz(getContext(), *value);
  if (*amount > maxFoldedBits)
    return {};
  unsigned width = value->getBitWidth() + *amount;
  return foldedMpz(getContext(), value->sext(width).shl(*amount));
}

OpFoldResult ShRSIOp::fold(FoldAdaptor adaptor) {
  std::optional<APInt> value = mpzOperand(adaptor.getValue());
  std::optional<std::uint64_t> amount = shiftAmount(adaptor.getAmount());
  if (!value || !amount)
    return {};
  // Shifting by the width or more leaves the sign.
  if (*amount >= value->getBitWidth())
    return foldedMpz(getContext(),
                     APInt(1, value->isNegative() ? 1 : 0, /*isSigned=*/false));
  return foldedMpz(getContext(), value->ashr(*amount));
}

OpFoldResult FloorMod2ExpIOp::fold(FoldAdaptor adaptor) {
  std::optional<APInt> value = mpzOperand(adaptor.getValue());
  if (!value)
    return {};
  // A nonnegative integer below the power, as zero is for any amount, is its
  // own remainder, and its reference becomes the result's.
  std::optional<std::uint64_t> amount = shiftAmount(adaptor.getAmount());
  if (!value->isNegative() &&
      (value->isZero() || (amount && *amount >= value->getActiveBits())))
    return getValue();
  // The remainder of a negative integer may have as many bits as the amount.
  if (!amount || *amount > maxFoldedBits)
    return {};
  return foldedMpz(getContext(), value->sextOrTrunc(*amount).zext(*amount + 1));
}

namespace {

// The remainder by two to the power zero is zero, which needs no reference, so
// the integer is released.
struct ZeroAmount : OpRewritePattern<FloorMod2ExpIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(FloorMod2ExpIOp op,
                                PatternRewriter &rewriter) const override {
    if (!matchPattern(op.getAmount(), m_Zero()))
      return failure();
    ReleaseOp::create(rewriter, op.getLoc(), op.getValue());
    rewriter.replaceOpWithNewOp<ConstantOp>(op, op.getType(), APInt(1, 0));
    return success();
  }
};

} // namespace

void FloorMod2ExpIOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  patterns.add<ZeroAmount>(context);
}

// A remainder by 2^k that is nonnegative, as a Euclidean one is and a floored
// one by a positive divisor, is computed by rcgmp.floormod2expi without
// division. A Euclidean remainder depends only on the magnitude of the
// divisor. The divisor is a constant, which needs no release, or a shift of
// one used only here, which is then dropped.
template <typename Op>
static LogicalResult remainderByPowerOfTwo(Op op, PatternRewriter &rewriter) {
  Operation *divisor = op.getRhs().getDefiningOp();
  Value amount;
  if (auto constant = dyn_cast_or_null<ConstantOp>(divisor)) {
    APInt value = constant.getValue();
    if (std::is_same_v<Op, EModSIOp>)
      value = value.sext(value.getBitWidth() + 1).abs();
    if (!value.isStrictlyPositive() || !value.isPowerOf2())
      return failure();
    amount = arith::ConstantIntOp::create(rewriter, op.getLoc(),
                                          value.exactLogBase2(), 64);
  } else if (auto shift = dyn_cast_or_null<ShLIOp>(divisor)) {
    auto one = shift.getValue().getDefiningOp<ConstantOp>();
    if (!one || !one.getValue().isOne() || !shift->hasOneUse())
      return failure();
    amount = shift.getAmount();
  } else {
    return failure();
  }
  rewriter.replaceOpWithNewOp<FloorMod2ExpIOp>(op, op.getType(), op.getLhs(),
                                               amount);
  if (isa<ShLIOp>(divisor))
    rewriter.eraseOp(divisor);
  return success();
}

void EModSIOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *) {
  patterns.add(remainderByPowerOfTwo<EModSIOp>);
}

void FloorModSIOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                               MLIRContext *) {
  patterns.add(remainderByPowerOfTwo<FloorModSIOp>);
}

// Masking with 2^k - 1 keeps the low k bits, which rcgmp.floormod2expi
// computes from k alone: inline for heap integers when the mask is immediate,
// and for nonnegative immediates when it is not, where rcgmp.andi calls the
// runtime. Constant operands of a commutative operation sort to the right.
static LogicalResult maskToRemainder(AndIOp op, PatternRewriter &rewriter) {
  auto mask = op.getRhs().getDefiningOp<ConstantOp>();
  if (!mask)
    return failure();
  APInt value = mask.getValue();
  if (value.isNegative() || value.countr_one() != value.getActiveBits())
    return failure();
  auto amount = arith::ConstantIntOp::create(rewriter, op.getLoc(),
                                             value.getActiveBits(), 64);
  rewriter.replaceOpWithNewOp<FloorMod2ExpIOp>(op, op.getType(), op.getLhs(),
                                               amount);
  return success();
}

void AndIOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                         MLIRContext *) {
  patterns.add(maskToRemainder);
}

OpFoldResult IPowIOp::fold(FoldAdaptor adaptor) {
  std::optional<APInt> base = mpzOperand(adaptor.getBase()),
                       exponent = integerOperand(adaptor.getExponent());
  // The first power is the base, whose reference becomes the result's.
  if (exponent && exponent->isOne())
    return getBase();
  if (!base)
    return {};
  // Every power of one is one. Powers of zero and -1 are themselves, except
  // for exponent zero and the even powers of -1, which are one.
  std::optional<std::int64_t> small = base->trySExtValue();
  APInt one(2, 1);
  if (small == 1)
    return foldedMpz(getContext(), one);
  if (!exponent)
    return {};
  if (exponent->isZero() || (small == -1 && !(*exponent)[0]))
    return foldedMpz(getContext(), one);
  if (small == 0 || small == -1)
    return foldedMpz(getContext(), *base);
  // A magnitude of bits bits, at least two, raised to the power e has more
  // than (bits - 1) * e bits.
  std::uint64_t bits =
      base->sext(base->getBitWidth() + 1).abs().getActiveBits();
  if (exponent->getActiveBits() > 32 ||
      (bits - 1) * exponent->getZExtValue() >= maxFoldedBits)
    return {};
  std::uint64_t remaining = exponent->getZExtValue();
  unsigned width = bits * remaining + 1;
  APInt result(width, 1), power = base->sext(width);
  for (;;) {
    if (remaining & 1)
      result *= power;
    remaining >>= 1;
    if (!remaining)
      break;
    power *= power;
  }
  return foldedMpz(getContext(), result);
}

namespace {

// Any power zero is one, which needs no reference, so the base is released.
struct ZeroExponent : OpRewritePattern<IPowIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IPowIOp op,
                                PatternRewriter &rewriter) const override {
    if (!matchPattern(op.getExponent(), m_Zero()))
      return failure();
    ReleaseOp::create(rewriter, op.getLoc(), op.getBase());
    rewriter.replaceOpWithNewOp<ConstantOp>(op, op.getType(), APInt(2, 1));
    return success();
  }
};

// Powers of two are shifts of one, whose inline path covers every immediate
// power.
struct PowerOfTwo : OpRewritePattern<IPowIOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IPowIOp op,
                                PatternRewriter &rewriter) const override {
    auto base = op.getBase().getDefiningOp<ConstantOp>();
    if (!base || base.getValue().trySExtValue() != 2)
      return failure();
    auto one =
        ConstantOp::create(rewriter, op.getLoc(), op.getType(), APInt(2, 1));
    rewriter.replaceOpWithNewOp<ShLIOp>(op, op.getType(), one,
                                        op.getExponent());
    return success();
  }
};

} // namespace

void IPowIOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                          MLIRContext *context) {
  patterns.add<ZeroExponent, PowerOfTwo>(context);
}

namespace {
enum class Rounding { truncate, floor, euclidean };
} // namespace

// The quotient and remainder of constant operands, rounded toward zero,
// toward negative infinity, or to a nonnegative remainder. A zero divisor
// folds only in lean mode, where the quotient is zero and the remainder lhs:
// in trap mode it terminates the program, and in undefined mode it is
// undefined.
static std::optional<std::pair<APInt, APInt>>
divideConstants(ArrayRef<Attribute> operands, Rounding rounding,
                DivZero divZero) {
  std::optional<APInt> lhs = mpzOperand(operands[0]),
                       rhs = mpzOperand(operands[1]);
  if (!lhs || !rhs)
    return std::nullopt;
  if (rhs->isZero()) {
    if (divZero != DivZero::lean)
      return std::nullopt;
    return std::pair(APInt(1, 0), *lhs);
  }
  // One extra bit holds the quotient of the minimum by -1.
  auto [a, b] = widened(*lhs, *rhs, 1);
  APInt quotient, remainder;
  APInt::sdivrem(a, b, quotient, remainder);
  bool adjust = false;
  switch (rounding) {
  case Rounding::truncate:
    break;
  case Rounding::floor:
    adjust = !remainder.isZero() && remainder.isNegative() != b.isNegative();
    break;
  case Rounding::euclidean:
    adjust = remainder.isNegative();
    break;
  }
  // The quotient moves down by one and the remainder up by one divisor,
  // except that a Euclidean remainder rises by the divisor's magnitude.
  if (adjust && (rounding == Rounding::floor || !b.isNegative())) {
    quotient -= 1;
    remainder += b;
  } else if (adjust) {
    quotient += 1;
    remainder -= b;
  }
  return std::pair(quotient, remainder);
}

// Whether a division gives the quotient zero and the remainder lhs for any
// lhs: a lean division by a constant zero.
template <typename Op>
static bool dividesLeanByZero(Op op, ArrayRef<Attribute> operands) {
  std::optional<APInt> rhs = mpzOperand(operands[1]);
  return op.getDivZero() == DivZero::lean && rhs && rhs->isZero();
}

template <Rounding rounding, bool quotient, typename Op>
static OpFoldResult foldDivision(Op op, ArrayRef<Attribute> operands) {
  // The remainder takes over the reference to lhs, and the constant divisor
  // needs no release. A quotient by zero would leave lhs unreleased, so the
  // canonicalization releases it instead.
  if (!quotient && dividesLeanByZero(op, operands))
    return op.getLhs();
  auto results = divideConstants(operands, rounding, op.getDivZero());
  if (!results)
    return {};
  // An exact quotient of a lhs that is not a multiple of rhs is undefined. It
  // is not folded to an arbitrary value, so the program computes what it
  // would without folding.
  if constexpr (quotient)
    if (op.getExact() && !results->second.isZero())
      return {};
  return foldedMpz(op->getContext(),
                   quotient ? results->first : results->second);
}

OpFoldResult DivSIOp::fold(FoldAdaptor adaptor) {
  return foldDivision<Rounding::truncate, true>(*this, adaptor.getOperands());
}

OpFoldResult RemSIOp::fold(FoldAdaptor adaptor) {
  return foldDivision<Rounding::truncate, false>(*this, adaptor.getOperands());
}

OpFoldResult FloorDivSIOp::fold(FoldAdaptor adaptor) {
  return foldDivision<Rounding::floor, true>(*this, adaptor.getOperands());
}

OpFoldResult FloorModSIOp::fold(FoldAdaptor adaptor) {
  return foldDivision<Rounding::floor, false>(*this, adaptor.getOperands());
}

OpFoldResult EDivSIOp::fold(FoldAdaptor adaptor) {
  return foldDivision<Rounding::euclidean, true>(*this, adaptor.getOperands());
}

OpFoldResult EModSIOp::fold(FoldAdaptor adaptor) {
  return foldDivision<Rounding::euclidean, false>(*this, adaptor.getOperands());
}

template <Rounding rounding, typename Op>
static LogicalResult foldDivisionPair(Op op, ArrayRef<Attribute> operands,
                                      SmallVectorImpl<OpFoldResult> &results) {
  if (dividesLeanByZero(op, operands)) {
    results.push_back(MpzValueAttr::get(op->getContext(), APInt(1, 0)));
    results.push_back(op.getLhs());
    return success();
  }
  auto divided = divideConstants(operands, rounding, op.getDivZero());
  if (!divided)
    return failure();
  results.push_back(foldedMpz(op->getContext(), divided->first));
  results.push_back(foldedMpz(op->getContext(), divided->second));
  return success();
}

LogicalResult DivRemSIOp::fold(FoldAdaptor adaptor,
                               SmallVectorImpl<OpFoldResult> &results) {
  return foldDivisionPair<Rounding::truncate>(*this, adaptor.getOperands(),
                                              results);
}

LogicalResult FloorDivModSIOp::fold(FoldAdaptor adaptor,
                                    SmallVectorImpl<OpFoldResult> &results) {
  return foldDivisionPair<Rounding::floor>(*this, adaptor.getOperands(),
                                           results);
}

LogicalResult EDivModSIOp::fold(FoldAdaptor adaptor,
                                SmallVectorImpl<OpFoldResult> &results) {
  return foldDivisionPair<Rounding::euclidean>(*this, adaptor.getOperands(),
                                               results);
}

OpFoldResult CmpIOp::fold(FoldAdaptor adaptor) {
  std::optional<APInt> lhs = mpzOperand(adaptor.getLhs()),
                       rhs = mpzOperand(adaptor.getRhs());
  if (!lhs || !rhs)
    return {};
  auto [a, b] = widened(*lhs, *rhs, 0);
  bool result = false;
  switch (getPredicate()) {
  case CmpIPredicate::eq:
    result = a == b;
    break;
  case CmpIPredicate::ne:
    result = a != b;
    break;
  case CmpIPredicate::slt:
    result = a.slt(b);
    break;
  case CmpIPredicate::sle:
    result = a.sle(b);
    break;
  case CmpIPredicate::sgt:
    result = a.sgt(b);
    break;
  case CmpIPredicate::sge:
    result = a.sge(b);
    break;
  }
  return BoolAttr::get(getContext(), result);
}

// Whether a signed value fits the width as a signed or an unsigned integer.
static bool fitsWidth(const APInt &value, unsigned width, bool isUnsigned) {
  if (isUnsigned)
    return !value.isNegative() && value.getActiveBits() <= width;
  return value.getSignificantBits() <= width;
}

OpFoldResult TruncIOp::fold(FoldAdaptor adaptor) {
  std::optional<APInt> value = mpzOperand(adaptor.getValue());
  if (!value)
    return {};
  // A value outside the range a flag promises is poison, which is not folded.
  unsigned width = getType().getIntOrFloatBitWidth();
  auto flags = getOverflowFlags();
  if ((bitEnumContainsAll(flags, arith::IntegerOverflowFlags::nsw) &&
       !fitsWidth(*value, width, false)) ||
      (bitEnumContainsAll(flags, arith::IntegerOverflowFlags::nuw) &&
       !fitsWidth(*value, width, true)))
    return {};
  return IntegerAttr::get(getType(), value->sextOrTrunc(width));
}

LogicalResult TruncIOverflowOp::fold(FoldAdaptor adaptor,
                                     SmallVectorImpl<OpFoldResult> &results) {
  std::optional<APInt> value = mpzOperand(adaptor.getValue());
  if (!value)
    return failure();
  unsigned width = getResult().getType().getIntOrFloatBitWidth();
  results.push_back(
      IntegerAttr::get(getResult().getType(), value->sextOrTrunc(width)));
  results.push_back(
      BoolAttr::get(getContext(), !fitsWidth(*value, width, getIsUnsigned())));
  return success();
}

OpFoldResult Log2IOp::fold(FoldAdaptor adaptor) {
  std::optional<APInt> value = mpzOperand(adaptor.getValue());
  if (!value)
    return {};
  APInt magnitude = value->sext(value->getBitWidth() + 1).abs();
  std::uint64_t log = magnitude.isZero() ? 0 : magnitude.logBase2();
  unsigned width = getType().getIntOrFloatBitWidth();
  return IntegerAttr::get(getType(), APInt(64, log).zextOrTrunc(width));
}

OpFoldResult TestBitIOp::fold(FoldAdaptor adaptor) {
  std::optional<APInt> value = mpzOperand(adaptor.getValue());
  std::optional<std::uint64_t> index = shiftAmount(adaptor.getIndex());
  if (!value || !index)
    return {};
  // Bits beyond the width are copies of the sign.
  bool bit =
      *index < value->getBitWidth() ? (*value)[*index] : value->isNegative();
  return BoolAttr::get(getContext(), bit);
}

// A limb is as wide as the handle of an integer of the given type.
static unsigned limbWidth(Operation *op, Type type) {
  return DataLayout::closest(op).getTypeSizeInBits(type).getFixedValue();
}

// The absolute value of a signed value, one bit wider so that it is exact.
static APInt magnitude(const APInt &value) {
  return value.sext(value.getBitWidth() + 1).abs();
}

OpFoldResult LimbCountOp::fold(FoldAdaptor adaptor) {
  std::optional<APInt> value = mpzOperand(adaptor.getValue());
  if (!value)
    return {};
  unsigned width = limbWidth(*this, getValue().getType());
  return IntegerAttr::get(
      getType(), llvm::divideCeil(magnitude(*value).getActiveBits(), width));
}

OpFoldResult LimbOp::fold(FoldAdaptor adaptor) {
  std::optional<APInt> value = mpzOperand(adaptor.getValue());
  std::optional<APInt> index = integerOperand(adaptor.getIndex());
  if (!value || !index)
    return {};
  unsigned width = cast<IntegerType>(getType()).getWidth();
  APInt bits = magnitude(*value);
  // Limbs at or beyond the count are zero.
  if (index->uge(llvm::divideCeil(bits.getActiveBits(), width)))
    return IntegerAttr::get(getType(), 0);
  return IntegerAttr::get(
      getType(), bits.lshr(index->getZExtValue() * width).zextOrTrunc(width));
}

// Constants need no reference counting.
LogicalResult RetainOp::canonicalize(RetainOp op, PatternRewriter &rewriter) {
  if (!op.getValue().getDefiningOp<ConstantOp>())
    return failure();
  rewriter.eraseOp(op);
  return success();
}

LogicalResult ReleaseOp::canonicalize(ReleaseOp op, PatternRewriter &rewriter) {
  if (!op.getValue().getDefiningOp<ConstantOp>())
    return failure();
  rewriter.eraseOp(op);
  return success();
}

// A lean quotient by a constant zero is zero, and lhs, which no result takes
// over, is released.
template <typename Op>
static LogicalResult canonicalizeLeanQuotient(Op op,
                                              PatternRewriter &rewriter) {
  auto divisor = op.getRhs().template getDefiningOp<ConstantOp>();
  if (op.getDivZero() != DivZero::lean || !divisor ||
      !divisor.getValue().isZero())
    return failure();
  ReleaseOp::create(rewriter, op.getLoc(), op.getLhs());
  rewriter.replaceOpWithNewOp<ConstantOp>(op, op.getType(), APInt(1, 0));
  return success();
}

LogicalResult DivSIOp::canonicalize(DivSIOp op, PatternRewriter &rewriter) {
  return canonicalizeLeanQuotient(op, rewriter);
}

// An exact quotient does not depend on its rounding, so a floored or Euclidean
// one becomes the truncated one, and exact quotients have a single form.
template <typename Op>
static LogicalResult canonicalizeRoundedQuotient(Op op,
                                                 PatternRewriter &rewriter) {
  if (succeeded(canonicalizeLeanQuotient(op, rewriter)))
    return success();
  if (!op.getExact())
    return failure();
  rewriter.replaceOpWithNewOp<DivSIOp>(op, op.getType(), op.getLhs(),
                                       op.getRhs(), op.getDivZero(),
                                       /*exact=*/true);
  return success();
}

LogicalResult FloorDivSIOp::canonicalize(FloorDivSIOp op,
                                         PatternRewriter &rewriter) {
  return canonicalizeRoundedQuotient(op, rewriter);
}

LogicalResult EDivSIOp::canonicalize(EDivSIOp op, PatternRewriter &rewriter) {
  return canonicalizeRoundedQuotient(op, rewriter);
}

LogicalResult ConstantOp::verify() {
  if (!isNarrowest(getValueAttr()))
    return emitOpError("requires a signless integer attribute of the narrowest "
                       "width holding its value");
  return success();
}

LogicalResult ArmAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                              ArmKind kind, ArrayAttr values) {
  switch (kind) {
  case ArmKind::exact:
    if (values.empty())
      return emitError() << "a 'case' arm requires at least one value";
    break;
  case ArmKind::atLeast:
  case ArmKind::atMost:
    if (values.size() != 1)
      return emitError() << "an '" << stringifyArmKind(kind)
                         << "' arm requires exactly one value";
    break;
  case ArmKind::otherwise:
    if (!values.empty())
      return emitError() << "a 'default' arm takes no values";
    break;
  }
  // Narrowest attributes are equal exactly when their values are.
  llvm::SmallDenseSet<Attribute> seen;
  for (Attribute value : values) {
    auto integer = dyn_cast<IntegerAttr>(value);
    if (!integer || !isNarrowest(integer))
      return emitError() << "requires signless integer values of the narrowest "
                            "width holding them";
    if (!seen.insert(value).second)
      return emitError() << "lists " << toString(integer.getValue(), 10, true)
                         << " more than once";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// MatchOp
//===----------------------------------------------------------------------===//

ParseResult MatchOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand scrutinee;
  Type type;
  if (parser.parseOperand(scrutinee) || parser.parseColonType(type) ||
      parser.resolveOperand(scrutinee, type, result.operands) ||
      parser.parseOptionalArrowTypeList(result.types) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
      parser.parseLBrace())
    return failure();
  Builder &builder = parser.getBuilder();
  SmallVector<Attribute> arms;
  while (failed(parser.parseOptionalRBrace())) {
    auto loc = parser.getCurrentLocation();
    StringRef keyword;
    if (parser.parseKeyword(&keyword))
      return failure();
    std::optional<ArmKind> kind = symbolizeArmKind(keyword);
    if (!kind)
      return parser.emitError(
          loc, "expected 'case', 'at_least', 'at_most', or 'default'");
    SmallVector<Attribute> values;
    if (*kind != ArmKind::otherwise) {
      do {
        IntegerAttr value;
        if (parseSignedInteger(parser, value))
          return failure();
        values.push_back(value);
      } while (*kind == ArmKind::exact &&
               succeeded(parser.parseOptionalComma()));
    }
    SmallVector<OpAsmParser::Argument> arguments;
    if (succeeded(parser.parseOptionalLParen())) {
      OpAsmParser::Argument &argument = arguments.emplace_back();
      if (parser.parseArgument(argument) || parser.parseRParen())
        return failure();
      argument.type = type;
    }
    Region *body = result.addRegion();
    if (parser.parseRegion(*body, arguments))
      return failure();
    ensureTerminator(*body, builder, result.location);
    arms.push_back(ArmAttr::get(builder.getContext(), *kind,
                                builder.getArrayAttr(values)));
  }
  result.addAttribute(getArmsAttrName(result.name), builder.getArrayAttr(arms));
  return success();
}

void MatchOp::print(OpAsmPrinter &p) {
  p << ' ' << getScrutinee() << " : " << getScrutinee().getType();
  p.printOptionalArrowTypeList(getResultTypes());
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), {getArmsAttrName()});
  p << " {";
  p.increaseIndent();
  for (auto [arm, body] :
       llvm::zip_equal(getArms().getAsRange<ArmAttr>(), getBodies())) {
    p.printNewline();
    p << stringifyArmKind(arm.getKind());
    if (!arm.getValues().empty())
      p << ' ';
    llvm::interleaveComma(arm.getValues(), p, [&](Attribute value) {
      printSignedInteger(p, *this, cast<IntegerAttr>(value));
    });
    if (body.getNumArguments()) {
      p << " (";
      p.printOperand(body.getArgument(0));
      p << ')';
    }
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/!getResults().empty());
  }
  p.decreaseIndent();
  p.printNewline();
  p << '}';
}

LogicalResult LimbOp::verify() {
  if (cast<IntegerType>(getType()).getWidth() !=
      limbWidth(*this, getValue().getType()))
    return emitOpError("requires an integer matching the handle bit width");
  return success();
}

LogicalResult MatchOp::verify() {
  if (getBodies().empty())
    return emitOpError("requires at least one arm");
  if (getArms().size() != getBodies().size())
    return emitOpError("requires one region per arm");
  for (auto [index, body] : llvm::enumerate(getBodies())) {
    // The custom syntax builds arms without verifying them.
    ArmAttr arm = getArm(index);
    if (failed(ArmAttr::verify([&] { return emitOpError(); }, arm.getKind(),
                               arm.getValues())))
      return failure();
    ArmKind kind = arm.getKind();
    if (kind == ArmKind::otherwise && index + 1 != getBodies().size())
      return emitOpError("requires the 'default' arm to be last");
    Block &block = body.front();
    unsigned limit = kind == ArmKind::exact ? 0 : 1;
    if (block.getNumArguments() > limit)
      return emitOpError("arm ")
             << index << " takes at most " << limit << " arguments";
    if (limit && block.getNumArguments() &&
        block.getArgument(0).getType() != getScrutinee().getType())
      return emitOpError("arm ")
             << index << " argument must have the scrutinee type";
    auto yield = cast<YieldOp>(block.getTerminator());
    if (yield.getResults().getTypes() != getResultTypes())
      return emitOpError("arm ")
             << index << " must yield values of the result types";
  }
  return success();
}

void MatchOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &successors) {
  if (!point.isParent()) {
    successors.push_back(RegionSuccessor(getOperation()));
    return;
  }
  for (Region &body : getBodies())
    successors.emplace_back(&body);
}

// Region arguments are bound by the match itself, not forwarded operands.
ValueRange MatchOp::getSuccessorInputs(RegionSuccessor successor) {
  return successor.isOperation() ? ValueRange(getResults()) : ValueRange();
}

SmallVector<MatchInterval> MatchOp::getIntervals() {
  unsigned width = 2;
  for (ArmAttr arm : getArms().getAsRange<ArmAttr>())
    for (auto value : arm.getValues().getAsRange<IntegerAttr>())
      width = std::max(width, value.getValue().getBitWidth() + 2);
  auto widen = [&](Attribute value) {
    return cast<IntegerAttr>(value).getValue().sext(width);
  };

  // Each pattern changes the selected arm only at its own boundaries, so the
  // arm selecting an interval between consecutive boundaries is the arm
  // selecting its first integer.
  SmallVector<APInt> boundaries;
  llvm::DenseMap<APInt, unsigned> firstCase;
  SmallVector<std::pair<APInt, unsigned>> thresholds, bounds;
  std::optional<unsigned> fallback;
  for (auto [index, arm] : llvm::enumerate(getArms().getAsRange<ArmAttr>())) {
    switch (arm.getKind()) {
    case ArmKind::exact:
      for (Attribute value : arm.getValues()) {
        APInt k = widen(value);
        firstCase.try_emplace(k, index);
        boundaries.push_back(k);
        boundaries.push_back(k + 1);
      }
      break;
    case ArmKind::atLeast:
      thresholds.emplace_back(widen(arm.getValues()[0]), index);
      boundaries.push_back(thresholds.back().first);
      break;
    case ArmKind::atMost:
      bounds.emplace_back(widen(arm.getValues()[0]), index);
      boundaries.push_back(bounds.back().first + 1);
      break;
    case ArmKind::otherwise:
      fallback = index;
      break;
    }
  }
  auto less = [](const APInt &a, const APInt &b) { return a.slt(b); };
  llvm::sort(boundaries, less);
  boundaries.erase(llvm::unique(boundaries), boundaries.end());
  llvm::sort(thresholds, [&](const auto &a, const auto &b) {
    return less(a.first, b.first);
  });

  // Sweep upwards, tracking the earliest at_least arm whose threshold has been
  // passed. An at_most arm selects until its bound is passed.
  std::optional<unsigned> atLeast;
  auto earliest = [](std::optional<unsigned> a, std::optional<unsigned> b) {
    return !a || (b && *b < *a) ? b : a;
  };
  const auto *nextThreshold = thresholds.begin();
  auto select = [&](const APInt &value) {
    for (; nextThreshold != thresholds.end() && nextThreshold->first.sle(value);
         ++nextThreshold)
      atLeast = earliest(atLeast, nextThreshold->second);
    std::optional<unsigned> arm = earliest(atLeast, fallback);
    for (auto &[bound, index] : bounds)
      if (value.sle(bound))
        arm = earliest(arm, index);
    if (auto it = firstCase.find(value); it != firstCase.end())
      arm = earliest(arm, it->second);
    return arm;
  };
  APInt below = boundaries.empty() ? APInt(width, 0) : boundaries.front() - 1;
  SmallVector<MatchInterval> intervals{
      {APInt::getSignedMinValue(width), select(below)}};
  for (const APInt &start : boundaries)
    if (std::optional<unsigned> arm = select(start);
        arm != intervals.back().arm)
      intervals.push_back({start, arm});
  return intervals;
}

const MatchInterval &findInterval(ArrayRef<MatchInterval> intervals,
                                  const APInt &value) {
  unsigned width = intervals.front().start.getBitWidth();
  if (value.getSignificantBits() > width)
    return value.isNegative() ? intervals.front() : intervals.back();
  APInt key = value.sextOrTrunc(width);
  const auto *after =
      llvm::upper_bound(intervals.drop_front(), key,
                        [](const APInt &key, const MatchInterval &i) {
                          return key.slt(i.start);
                        });
  return *std::prev(after);
}

namespace {

// Rebuilds a match with the given arms and their regions, in order.
MatchOp rebuildMatch(PatternRewriter &rewriter, MatchOp op,
                     ArrayRef<Attribute> arms, ArrayRef<Region *> bodies) {
  auto match = MatchOp::create(rewriter, op.getLoc(), op.getResultTypes(),
                               op.getScrutinee(), rewriter.getArrayAttr(arms),
                               bodies.size());
  match->setDiscardableAttrs(op->getDiscardableAttrDictionary());
  for (auto [body, source] : llvm::zip_equal(match.getBodies(), bodies))
    body.takeBody(*source);
  return match;
}

// Drops arms that select no integer, and case values that an earlier arm
// selects.
struct RemoveUnreachableArms : OpRewritePattern<MatchOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MatchOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<MatchInterval> intervals = op.getIntervals();
    llvm::SmallDenseSet<unsigned> selected;
    for (const MatchInterval &interval : intervals)
      if (interval.arm)
        selected.insert(*interval.arm);
    SmallVector<Attribute> arms;
    SmallVector<Region *> bodies;
    bool changed = false;
    for (auto [index, body] : llvm::enumerate(op.getBodies())) {
      ArmAttr arm = op.getArm(index);
      if (!selected.contains(index)) {
        changed = true;
        continue;
      }
      if (arm.getKind() == ArmKind::exact) {
        SmallVector<Attribute> values;
        for (auto value : arm.getValues().getAsRange<IntegerAttr>())
          if (findInterval(intervals, value.getValue()).arm == index)
            values.push_back(value);
        if (values.size() != arm.getValues().size()) {
          changed = true;
          arm = ArmAttr::get(op.getContext(), ArmKind::exact,
                             rewriter.getArrayAttr(values));
        }
      }
      arms.push_back(arm);
      bodies.push_back(&body);
    }
    if (!changed)
      return failure();
    rewriter.replaceOp(op, rebuildMatch(rewriter, op, arms, bodies));
    return success();
  }
};

// An at_least c region that only matches its argument m = n - c and yields the
// results becomes the inner arms, shifted by c: case k selects c + k, and
// at_least d selects c + d with argument m - d = n - (c + d). The inner default
// becomes at_least c. Since m is nonnegative, inner arms selecting only
// negative integers are dropped. An inner at_least with a negative threshold
// would bind a different remainder, and an inner at_most would need a lower
// bound too, so they prevent fusion. An at_most c region matching m = c - n
// mirrors this: case k selects c - k, and at_least d becomes at_most c - d,
// with argument m - d = (c - d) - n.
struct FuseRemainderMatch : OpRewritePattern<MatchOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MatchOp op,
                                PatternRewriter &rewriter) const override {
    for (auto [index, body] : llvm::enumerate(op.getBodies())) {
      ArmAttr arm = op.getArm(index);
      if ((arm.getKind() != ArmKind::atLeast &&
           arm.getKind() != ArmKind::atMost) ||
          body.getNumArguments() != 1)
        continue;
      Block &block = body.front();
      BlockArgument remainder = block.getArgument(0);
      auto inner = dyn_cast<MatchOp>(block.front());
      if (!inner || inner.getScrutinee() != remainder ||
          !remainder.hasOneUse() ||
          inner->getNextNode() != block.getTerminator() ||
          block.getTerminator()->getOperands() != inner.getResults())
        continue;
      if (llvm::any_of(inner.getArms().getAsRange<ArmAttr>(), [](ArmAttr a) {
            return a.getKind() == ArmKind::atMost ||
                   (a.getKind() == ArmKind::atLeast &&
                    cast<IntegerAttr>(a.getValues()[0])
                        .getValue()
                        .isNegative());
          }))
        continue;
      return fuse(rewriter, op, index, inner);
    }
    return failure();
  }

private:
  static LogicalResult fuse(PatternRewriter &rewriter, MatchOp op,
                            unsigned index, MatchOp inner) {
    ArmKind outer = op.getArm(index).getKind();
    APInt offset =
        cast<IntegerAttr>(op.getArm(index).getValues()[0]).getValue();
    // The integer at distance k from the bound, inside the outer arm.
    auto shift = [&](Attribute value) -> Attribute {
      APInt k = cast<IntegerAttr>(value).getValue();
      unsigned width = std::max(k.getBitWidth(), offset.getBitWidth()) + 1;
      APInt distance = k.sext(width);
      if (outer == ArmKind::atMost)
        distance.negate();
      return getNarrowestAttr(rewriter, offset.sext(width) + distance);
    };
    SmallVector<Attribute> arms;
    SmallVector<Region *> bodies;
    auto keep = [&](Attribute arm, Region &body) {
      arms.push_back(arm);
      bodies.push_back(&body);
    };
    for (unsigned i = 0; i < index; ++i)
      keep(op.getArm(i), op.getBodies()[i]);
    for (auto [arm, body] : llvm::zip_equal(
             inner.getArms().getAsRange<ArmAttr>(), inner.getBodies())) {
      SmallVector<Attribute> values;
      for (auto value : arm.getValues().getAsRange<IntegerAttr>())
        if (!value.getValue().isNegative())
          values.push_back(shift(value));
      switch (arm.getKind()) {
      case ArmKind::exact:
        if (!values.empty())
          keep(ArmAttr::get(op.getContext(), ArmKind::exact,
                            rewriter.getArrayAttr(values)),
               body);
        break;
      case ArmKind::atLeast:
        keep(
            ArmAttr::get(op.getContext(), outer, rewriter.getArrayAttr(values)),
            body);
        break;
      case ArmKind::otherwise:
        keep(op.getArm(index), body);
        break;
      case ArmKind::atMost:
        llvm_unreachable("an inner at_most prevents fusion");
      }
    }
    for (unsigned i = index + 1; i < op.getBodies().size(); ++i)
      keep(op.getArm(i), op.getBodies()[i]);
    rewriter.replaceOp(op, rebuildMatch(rewriter, op, arms, bodies));
    return success();
  }
};

// A region argument that the region only releases is dropped, so that the
// match releases the scrutinee instead of computing the argument.
struct DropReleasedArgument : OpRewritePattern<MatchOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MatchOp op,
                                PatternRewriter &rewriter) const override {
    for (Region &body : op.getBodies()) {
      if (body.getNumArguments() != 1)
        continue;
      BlockArgument argument = body.getArgument(0);
      if (!argument.hasOneUse())
        continue;
      auto release = dyn_cast<ReleaseOp>(*argument.getUsers().begin());
      if (!release || release->getBlock() != &body.front())
        continue;
      rewriter.modifyOpInPlace(op, [&] {
        rewriter.eraseOp(release);
        body.eraseArgument(0);
      });
      return success();
    }
    return failure();
  }
};

// A constant scrutinee selects its arm statically. Constants need no reference
// counting, so the scrutinee is not released.
struct FoldConstantScrutinee : OpRewritePattern<MatchOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MatchOp op,
                                PatternRewriter &rewriter) const override {
    auto constant = op.getScrutinee().getDefiningOp<ConstantOp>();
    if (!constant)
      return failure();
    APInt value = constant.getValue();
    std::optional<unsigned> index = findInterval(op.getIntervals(), value).arm;
    if (!index)
      return failure();
    Block &block = op.getBodies()[*index].front();
    SmallVector<Value> arguments;
    if (block.getNumArguments()) {
      ArmAttr arm = op.getArm(*index);
      if (arm.getKind() == ArmKind::otherwise) {
        arguments.push_back(constant);
      } else {
        APInt bound = cast<IntegerAttr>(arm.getValues()[0]).getValue();
        unsigned width = std::max(value.getBitWidth(), bound.getBitWidth()) + 1;
        APInt distance = value.sext(width) - bound.sext(width);
        if (arm.getKind() == ArmKind::atMost)
          distance.negate();
        arguments.push_back(ConstantOp::create(rewriter, op.getLoc(),
                                               constant.getType(), distance));
      }
    }
    Operation *yield = block.getTerminator();
    SmallVector<Value> results(yield->getOperands());
    rewriter.eraseOp(yield);
    rewriter.inlineBlockBefore(&block, op, arguments);
    rewriter.replaceOp(op, results);
    return success();
  }
};

} // namespace

void MatchOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                          MLIRContext *context) {
  patterns.add<RemoveUnreachableArms, FuseRemainderMatch, DropReleasedArgument,
               FoldConstantScrutinee>(context);
}

} // namespace mlir::rcgmp
