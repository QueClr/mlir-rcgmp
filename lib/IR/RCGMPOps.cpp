#include "RCGMP/IR/RCGMPOps.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/StringExtras.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/PatternMatch.h>

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

// The quotient and remainder of constant operands, rounded toward zero or
// toward negative infinity. A zero divisor is not folded: without
// assume_nonzero it terminates the program, and with it it is undefined.
static std::optional<std::pair<APInt, APInt>>
divideConstants(ArrayRef<Attribute> operands, bool floor) {
  std::optional<APInt> lhs = mpzOperand(operands[0]),
                       rhs = mpzOperand(operands[1]);
  if (!lhs || !rhs || rhs->isZero())
    return std::nullopt;
  // One extra bit holds the quotient of the minimum by -1.
  auto [a, b] = widened(*lhs, *rhs, 1);
  APInt quotient, remainder;
  APInt::sdivrem(a, b, quotient, remainder);
  if (floor && !remainder.isZero() &&
      remainder.isNegative() != b.isNegative()) {
    quotient -= 1;
    remainder += b;
  }
  return std::pair(quotient, remainder);
}

template <bool floor, bool quotient>
static OpFoldResult foldDivision(Operation *op, ArrayRef<Attribute> operands) {
  auto results = divideConstants(operands, floor);
  if (!results)
    return {};
  return foldedMpz(op->getContext(),
                   quotient ? results->first : results->second);
}

OpFoldResult DivSIOp::fold(FoldAdaptor adaptor) {
  return foldDivision<false, true>(*this, adaptor.getOperands());
}

OpFoldResult RemSIOp::fold(FoldAdaptor adaptor) {
  return foldDivision<false, false>(*this, adaptor.getOperands());
}

OpFoldResult FloorDivSIOp::fold(FoldAdaptor adaptor) {
  return foldDivision<true, true>(*this, adaptor.getOperands());
}

OpFoldResult FloorModSIOp::fold(FoldAdaptor adaptor) {
  return foldDivision<true, false>(*this, adaptor.getOperands());
}

template <bool floor>
static LogicalResult foldDivisionPair(Operation *op,
                                      ArrayRef<Attribute> operands,
                                      SmallVectorImpl<OpFoldResult> &results) {
  auto divided = divideConstants(operands, floor);
  if (!divided)
    return failure();
  results.push_back(foldedMpz(op->getContext(), divided->first));
  results.push_back(foldedMpz(op->getContext(), divided->second));
  return success();
}

LogicalResult DivRemSIOp::fold(FoldAdaptor adaptor,
                               SmallVectorImpl<OpFoldResult> &results) {
  return foldDivisionPair<false>(*this, adaptor.getOperands(), results);
}

LogicalResult FloorDivModSIOp::fold(FoldAdaptor adaptor,
                                    SmallVectorImpl<OpFoldResult> &results) {
  return foldDivisionPair<true>(*this, adaptor.getOperands(), results);
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
