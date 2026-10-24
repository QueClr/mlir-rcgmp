#include "RCGMP/Conversion/ConvertRCGMPToSCF.h"
#include "FastPathLowering.h"
#include "MatchLowering.h"
#include "RCGMP/IR/RCGMPBridgeOps.h"
#include "RCGMP/IR/RCGMPOps.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::rcgmp {
namespace {

template <typename ExtendOp, bool isUnsigned>
struct ExtendLowering : OpRewritePattern<ExtendOp> {
  using OpRewritePattern<ExtendOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ExtendOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto fits = bridge::MpzFitsImmediateOp::create(rewriter, loc, op.getValue(),
                                                   isUnsigned);
    auto result =
        scf::IfOp::create(rewriter, loc, TypeRange{op.getType()}, fits,
                          /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(&result.getThenRegion().front());
    auto immediate = bridge::MpzFromImmediateOp::create(
        rewriter, loc, op.getType(), op.getValue(), isUnsigned);
    scf::YieldOp::create(rewriter, loc, immediate.getResult());
    rewriter.setInsertionPointToStart(&result.getElseRegion().front());
    auto heap = bridge::MpzPromoteOp::create(rewriter, loc, op.getType(),
                                             op.getValue(), isUnsigned);
    scf::YieldOp::create(rewriter, loc, heap.getResult());
    rewriter.replaceOp(op, result.getResults());
    return success();
  }
};
using ExtSILowering = ExtendLowering<ExtSIOp, false>;
using ExtUILowering = ExtendLowering<ExtUIOp, true>;

struct ConstantLowering : OpRewritePattern<ConstantOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto type = op.getType();
    unsigned width =
        DataLayout::closest(op).getTypeSizeInBits(type).getFixedValue();
    APInt value = op.getValue();
    if (!bridge::fitsImmediate(value, width)) {
      rewriter.replaceOpWithNewOp<bridge::MpzImmortalOp>(op, type,
                                                         op.getValueAttr());
      return success();
    }
    APInt tagged = value.sextOrTrunc(width).shl(2);
    tagged.setBit(0);
    auto bits = arith::ConstantOp::create(
        rewriter, op.getLoc(),
        rewriter.getIntegerAttr(rewriter.getIntegerType(width), tagged));
    rewriter.replaceOpWithNewOp<bridge::MpzFromBitsOp>(op, type, bits);
    return success();
  }
};

// Move the insertion point into a branch taken only for counted heap values.
// Immediate and immortal values have no reference count to update.
void enterCountedBranch(PatternRewriter &rewriter, Location loc, Value value) {
  auto immediate = bridge::MpzIsImmediateOp::create(rewriter, loc, value);
  auto heap = scf::IfOp::create(rewriter, loc, immediate, true);
  rewriter.setInsertionPointToStart(&heap.getElseRegion().front());
  auto immortal = bridge::MpzIsImmortalOp::create(rewriter, loc, value);
  auto counted = scf::IfOp::create(rewriter, loc, immortal, true);
  rewriter.setInsertionPointToStart(&counted.getElseRegion().front());
}

struct RetainLowering : OpRewritePattern<RetainOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RetainOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    enterCountedBranch(rewriter, loc, op.getValue());
    auto overflow = bridge::MpzIncRefOp::create(rewriter, loc, op.getValue());
    auto one = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
    auto valid = arith::XOrIOp::create(rewriter, loc, overflow, one);
    cf::AssertOp::create(rewriter, loc, valid,
                         "rcgmp reference count overflow");
    rewriter.eraseOp(op);
    return success();
  }
};

// The only reference to a heap value is released by freeing it, without
// updating its count: no other owner can observe the count, and the
// uniqueness test's acquire load orders the accesses of earlier owners before
// the free. Other heap values are released as counted.
struct ReleaseLowering : OpRewritePattern<ReleaseOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReleaseOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value value = op.getValue();
    auto immediate = bridge::MpzIsImmediateOp::create(rewriter, loc, value);
    auto heap = scf::IfOp::create(rewriter, loc, immediate, true);
    rewriter.setInsertionPointToStart(&heap.getElseRegion().front());
    auto unique = bridge::MpzIsUniqueOp::create(rewriter, loc, value);
    auto shared = scf::IfOp::create(rewriter, loc, unique, true);
    rewriter.setInsertionPointToStart(&shared.getThenRegion().front());
    bridge::MpzFreeOp::create(rewriter, loc, value);
    rewriter.setInsertionPointToStart(&shared.getElseRegion().front());
    auto immortal = bridge::MpzIsImmortalOp::create(rewriter, loc, value);
    auto counted = scf::IfOp::create(rewriter, loc, immortal, true);
    rewriter.setInsertionPointToStart(&counted.getElseRegion().front());
    auto last = bridge::MpzDecRefOp::create(rewriter, loc, value);
    auto destroy = scf::IfOp::create(rewriter, loc, last, false);
    rewriter.setInsertionPointToStart(&destroy.getThenRegion().front());
    bridge::MpzFreeOp::create(rewriter, loc, value);
    rewriter.eraseOp(op);
    return success();
  }
};

struct AddILowering : FastPathLowering<AddILowering> {
  using FastPathLowering::FastPathLowering;
  using Op = AddIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_add_atomic" : "rcgmp_mpz_add";
  }

  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *, ValueRange operands) {
    Value lhs = operands[0], rhs = operands[1];
    unsigned width = cast<IntegerType>(lhs.getType()).getWidth();
    auto checked = bridge::AddSIExtendedOp::create(
        rewriter, loc, lhs.getType(), rewriter.getI1Type(), lhs, rhs);
    Value sum = checked.getSum();
    auto one = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
    auto fits =
        arith::XOrIOp::create(rewriter, loc, checked.getOverflow(), one);
    // 4*x+1 + 4*y+1 has bit 1 set iff both handles are immediate.
    auto two = arith::ConstantIntOp::create(rewriter, loc, 2, width);
    auto tag = arith::AndIOp::create(rewriter, loc, sum, two);
    auto both = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                      tag, two);
    auto valid = arith::AndIOp::create(rewriter, loc, fits, both);
    auto three = arith::ConstantIntOp::create(rewriter, loc, 3, width);
    auto tagged = arith::XOrIOp::create(rewriter, loc, sum, three);
    return {{tagged}, valid};
  }
};

struct SubILowering : FastPathLowering<SubILowering> {
  using FastPathLowering::FastPathLowering;
  using Op = SubIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_sub_atomic" : "rcgmp_mpz_sub";
  }

  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *, ValueRange operands) {
    Value lhs = operands[0], rhs = operands[1];
    auto checked = bridge::SubSIExtendedOp::create(
        rewriter, loc, lhs.getType(), rewriter.getI1Type(), lhs, rhs);
    auto yes = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
    auto fits =
        arith::XOrIOp::create(rewriter, loc, checked.getOverflow(), yes);
    auto both = allImmediate(rewriter, loc, {lhs, rhs});
    auto valid = arith::AndIOp::create(rewriter, loc, fits, both);
    auto one = arith::ConstantIntOp::create(
        rewriter, loc, 1, cast<IntegerType>(lhs.getType()).getWidth());
    // Subtracting tagged immediates yields 4*(x-y); restore the low tag bit.
    auto tagged = arith::OrIOp::create(rewriter, loc, checked.getDiff(), one);
    return {{tagged}, valid};
  }
};

struct MonusILowering : FastPathLowering<MonusILowering> {
  using FastPathLowering::FastPathLowering;
  using Op = MonusIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_monus_atomic" : "rcgmp_mpz_monus";
  }

  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *, ValueRange operands) {
    Value lhs = operands[0], rhs = operands[1];
    unsigned width = cast<IntegerType>(lhs.getType()).getWidth();
    // Tagged immediates order as their values, so the larger minus rhs is
    // 4*max(x-y, 0). It lies in [0, 2^W), and is immediate iff its sign bit
    // is clear, which the bounds of the maximum prove for a nonnegative
    // constant rhs.
    auto larger = arith::MaxSIOp::create(rewriter, loc, lhs, rhs);
    auto difference = arith::SubIOp::create(rewriter, loc, larger, rhs);
    auto zero = arith::ConstantIntOp::create(rewriter, loc, 0, width);
    auto fits = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sge,
                                      difference, zero);
    auto valid = arith::AndIOp::create(rewriter, loc, fits,
                                       allImmediate(rewriter, loc, {lhs, rhs}));
    auto one = arith::ConstantIntOp::create(rewriter, loc, 1, width);
    auto tagged = arith::OrIOp::create(rewriter, loc, difference, one);
    return {{tagged}, valid};
  }
};

struct MulILowering : FastPathLowering<MulILowering> {
  using FastPathLowering::FastPathLowering;
  using Op = MulIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_mul_atomic" : "rcgmp_mpz_mul";
  }

  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *, ValueRange operands) {
    Value lhs = operands[0], rhs = operands[1];
    unsigned width = cast<IntegerType>(lhs.getType()).getWidth();
    auto two = arith::ConstantIntOp::create(rewriter, loc, 2, width);
    auto mask = arith::ConstantIntOp::create(rewriter, loc, -4, width);
    auto x = arith::ShRSIOp::create(rewriter, loc, lhs, two);
    auto scaledY = arith::AndIOp::create(rewriter, loc, rhs, mask);
    // x * (4*y) fits the signed word iff x*y fits the immediate payload.
    // The checked product remains defined for arbitrary heap-handle bits.
    auto product = bridge::MulSIOverflowOp::create(
        rewriter, loc, lhs.getType(), rewriter.getI1Type(), x, scaledY);
    auto yes = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
    auto fits =
        arith::XOrIOp::create(rewriter, loc, product.getOverflow(), yes);
    auto both = allImmediate(rewriter, loc, {lhs, rhs});
    auto valid = arith::AndIOp::create(rewriter, loc, fits, both);
    auto one = arith::ConstantIntOp::create(rewriter, loc, 1, width);
    auto tagged =
        arith::OrIOp::create(rewriter, loc, product.getProduct(), one);
    return {{tagged}, valid};
  }
};

// The greatest common divisor of two immediates, by Stein's binary algorithm,
// which replaces division by subtraction and shifts.
struct GcdILowering : FastPathLowering<GcdILowering> {
  using FastPathLowering::FastPathLowering;
  using Op = GcdIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_gcd_atomic" : "rcgmp_mpz_gcd";
  }

  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *, ValueRange operands) {
    Value lhs = operands[0], rhs = operands[1];
    auto type = cast<IntegerType>(lhs.getType());
    unsigned width = type.getWidth();
    Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, width);
    Value one = arith::ConstantIntOp::create(rewriter, loc, 1, width);
    Value both = allImmediate(rewriter, loc, {lhs, rhs});
    // The loop runs only for immediates; heap handles take the runtime.
    auto branch = scf::IfOp::create(rewriter, loc, TypeRange{type}, both,
                                    /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(&branch.getThenRegion().front());
    // Clearing the tag of 4*x+1 leaves 4*x, whose absolute value 4*|x| is at
    // most 2^(W-1), read as unsigned. The greatest common divisor of these is
    // 4*gcd(x, y), already shifted into place for tagging.
    auto magnitude = [&](Value bits) -> Value {
      return math::AbsIOp::create(
          rewriter, loc, arith::XOrIOp::create(rewriter, loc, bits, one));
    };
    auto [a, b] =
        reduceUnbalanced(rewriter, loc, magnitude(lhs), magnitude(rhs), zero);
    scf::YieldOp::create(rewriter, loc, binaryGcd(rewriter, loc, a, b, zero));
    rewriter.setInsertionPointToStart(&branch.getElseRegion().front());
    scf::YieldOp::create(rewriter, loc, zero);
    rewriter.setInsertionPointAfter(branch);
    // Only 4*2^(W-3), the greatest common divisor of the immediate minimum
    // and zero or itself, is not immediate, and its sign bit is set.
    Value scaled = branch.getResult(0);
    auto fits = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sge,
                                      scaled, zero);
    auto valid = arith::AndIOp::create(rewriter, loc, both, fits);
    auto tagged = arith::OrIOp::create(rewriter, loc, scaled, one);
    return {{tagged}, valid};
  }

private:
  // The binary algorithm removes about two bits per step, so a much larger
  // operand takes many steps, as one or minus one does with any other;
  // dividing it by the smaller once leaves operands of similar size. Eight
  // bits take about as long to remove by steps as a division does on recent
  // processors, so an operand is much larger when it exceeds the other times
  // 2^8. The remainder of 4*x by 4*y is 4*(x mod y).
  static std::pair<Value, Value> reduceUnbalanced(PatternRewriter &rewriter,
                                                  Location loc, Value a,
                                                  Value b, Value zero) {
    Type type = a.getType();
    Value eight = arith::ConstantIntOp::create(
        rewriter, loc, 8, cast<IntegerType>(type).getWidth());
    auto exceeds = [&](Value value, Value other) -> Value {
      auto shifted = arith::ShRUIOp::create(rewriter, loc, value, eight);
      auto above = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::ugt, shifted, other);
      auto nonzero = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::ne, other, zero);
      return arith::AndIOp::create(rewriter, loc, above, nonzero);
    };
    auto unbalanced =
        arith::OrIOp::create(rewriter, loc, exceeds(a, b), exceeds(b, a));
    auto branch = scf::IfOp::create(rewriter, loc, TypeRange{type, type},
                                    unbalanced, /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(&branch.getThenRegion().front());
    Value larger = arith::MaxUIOp::create(rewriter, loc, a, b);
    Value smaller = arith::MinUIOp::create(rewriter, loc, a, b);
    Value remainder = arith::RemUIOp::create(rewriter, loc, larger, smaller);
    scf::YieldOp::create(rewriter, loc, ValueRange{remainder, smaller});
    rewriter.setInsertionPointToStart(&branch.getElseRegion().front());
    scf::YieldOp::create(rewriter, loc, ValueRange{a, b});
    rewriter.setInsertionPointAfter(branch);
    return {branch.getResult(0), branch.getResult(1)};
  }

  // The greatest common divisor of a and b, multiples of four of at most
  // 2^(W-1).
  static Value binaryGcd(PatternRewriter &rewriter, Location loc, Value a,
                         Value b, Value zero) {
    Type type = a.getType();
    unsigned width = cast<IntegerType>(type).getWidth();
    // With the top bit set, a count of trailing zeros never reaches the width,
    // so that shifting by it is defined, and zero shifts to zero.
    Value top = arith::ConstantOp::create(
        rewriter, loc,
        rewriter.getIntegerAttr(type, APInt::getSignedMinValue(width)));
    auto trailingZeros = [&](Value value) -> Value {
      return math::CountTrailingZerosOp::create(
          rewriter, loc, arith::OrIOp::create(rewriter, loc, value, top));
    };
    auto oddPart = [&](Value value) -> Value {
      return arith::ShRUIOp::create(rewriter, loc, value, trailingZeros(value));
    };
    // The power of two that both share is restored at the end. A zero operand
    // takes the odd part of the other, so that the loop finds it at once.
    Value shift = trailingZeros(arith::OrIOp::create(rewriter, loc, a, b));
    auto nonzeroOr = [&](Value value, Value other) -> Value {
      auto isZero = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, value, zero);
      return arith::SelectOp::create(rewriter, loc, isZero, other, value);
    };
    Value u = oddPart(nonzeroOr(a, b)), v = oddPart(nonzeroOr(b, a));
    // Both odd parts are below 2^(W-3), so the sign of their difference orders
    // them. Each step replaces the larger by the odd part of the difference,
    // which is even and nonzero, until the two meet at the odd part of the
    // greatest common divisor.
    auto loop = scf::WhileOp::create(rewriter, loc, TypeRange{type, type, type},
                                     ValueRange{u, v});
    Block *before =
        rewriter.createBlock(&loop.getBefore(), {}, {type, type}, {loc, loc});
    Value first = before->getArgument(0), second = before->getArgument(1);
    Value difference = arith::SubIOp::create(rewriter, loc, first, second);
    auto differ = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne,
                                        difference, zero);
    scf::ConditionOp::create(rewriter, loc, differ,
                             ValueRange{difference, first, second});
    Block *after = rewriter.createBlock(&loop.getAfter(), {},
                                        {type, type, type}, {loc, loc, loc});
    Value step = after->getArgument(0);
    Value smaller = arith::MinUIOp::create(rewriter, loc, after->getArgument(1),
                                           after->getArgument(2));
    Value reduced = arith::ShRUIOp::create(
        rewriter, loc, math::AbsIOp::create(rewriter, loc, step),
        math::CountTrailingZerosOp::create(rewriter, loc, step));
    scf::YieldOp::create(rewriter, loc, ValueRange{reduced, smaller});
    rewriter.setInsertionPointAfter(loop);
    return arith::ShLIOp::create(rewriter, loc, loop.getResult(1), shift);
  }
};

struct AndILowering : FastPathLowering<AndILowering> {
  using FastPathLowering::FastPathLowering;
  using Op = AndIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_and_atomic" : "rcgmp_mpz_and";
  }

  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *, ValueRange operands) {
    Value lhs = operands[0], rhs = operands[1];
    auto bits = arith::AndIOp::create(rewriter, loc, lhs, rhs);
    return {{bits}, allImmediate(rewriter, loc, {lhs, rhs})};
  }
};

struct XOrILowering : FastPathLowering<XOrILowering> {
  using FastPathLowering::FastPathLowering;
  using Op = XOrIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_xor_atomic" : "rcgmp_mpz_xor";
  }

  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *, ValueRange operands) {
    Value lhs = operands[0], rhs = operands[1];
    auto bits = arith::XOrIOp::create(rewriter, loc, lhs, rhs);
    auto one = arith::ConstantIntOp::create(
        rewriter, loc, 1, cast<IntegerType>(lhs.getType()).getWidth());
    auto tagged = arith::OrIOp::create(rewriter, loc, bits, one);
    return {{tagged}, allImmediate(rewriter, loc, {lhs, rhs})};
  }
};

struct OrILowering : FastPathLowering<OrILowering> {
  using FastPathLowering::FastPathLowering;
  using Op = OrIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_or_atomic" : "rcgmp_mpz_or";
  }

  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *, ValueRange operands) {
    Value lhs = operands[0], rhs = operands[1];
    auto bits = arith::OrIOp::create(rewriter, loc, lhs, rhs);
    return {{bits}, allImmediate(rewriter, loc, {lhs, rhs})};
  }
};

struct NotILowering : FastPathLowering<NotILowering> {
  using FastPathLowering::FastPathLowering;
  using Op = NotIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_not_atomic" : "rcgmp_mpz_not";
  }

  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *, ValueRange operands) {
    Value bits = operands[0];
    // Complementing the payload of 4*x+1 keeps the tag and never leaves the
    // immediate range.
    auto payload = arith::ConstantIntOp::create(
        rewriter, loc, -4, cast<IntegerType>(bits.getType()).getWidth());
    auto tagged = arith::XOrIOp::create(rewriter, loc, bits, payload);
    return {{tagged}, allImmediate(rewriter, loc, bits)};
  }
};

// Negation of 4*x+1 is 2 minus the tagged bits. It leaves the immediate range
// only at its minimum, whose tagged bits are the signed word minimum plus one.
struct Negation {
  Value value;
  Value valid;
};

Negation buildNegation(PatternRewriter &rewriter, Location loc, Value bits) {
  unsigned width = cast<IntegerType>(bits.getType()).getWidth();
  auto two = arith::ConstantIntOp::create(rewriter, loc, 2, width);
  auto negated = arith::SubIOp::create(rewriter, loc, two, bits);
  APInt minimum = APInt::getSignedMinValue(width) + 1;
  auto tagMinimum = arith::ConstantOp::create(
      rewriter, loc, rewriter.getIntegerAttr(bits.getType(), minimum));
  auto inRange = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne,
                                       bits, tagMinimum);
  return {negated, inRange};
}

struct NegILowering : FastPathLowering<NegILowering> {
  using FastPathLowering::FastPathLowering;
  using Op = NegIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_neg_atomic" : "rcgmp_mpz_neg";
  }

  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *, ValueRange operands) {
    Value bits = operands[0];
    auto [negated, inRange] = buildNegation(rewriter, loc, bits);
    auto valid = arith::AndIOp::create(rewriter, loc, inRange,
                                       allImmediate(rewriter, loc, bits));
    return {{negated}, valid};
  }
};

struct AbsILowering : FastPathLowering<AbsILowering> {
  using FastPathLowering::FastPathLowering;
  using Op = AbsIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_abs_atomic" : "rcgmp_mpz_abs";
  }

  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *, ValueRange operands) {
    Value bits = operands[0];
    auto [negated, inRange] = buildNegation(rewriter, loc, bits);
    // 4*x+1 has the sign of x.
    auto zero = arith::ConstantIntOp::create(
        rewriter, loc, 0, cast<IntegerType>(bits.getType()).getWidth());
    auto negative = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::slt, bits, zero);
    auto magnitude =
        arith::SelectOp::create(rewriter, loc, negative, negated, bits);
    auto valid = arith::AndIOp::create(rewriter, loc, inRange,
                                       allImmediate(rewriter, loc, bits));
    return {{magnitude}, valid};
  }
};

// Shift amounts are unsigned. One wider than the handle saturates, which no
// shift can tell apart: no integer has that many bits to shift right, and
// shifting a nonzero integer left that far cannot be allocated.
Value saturatingWord(PatternRewriter &rewriter, Location loc, Value amount,
                     IntegerType word) {
  auto type = cast<IntegerType>(amount.getType());
  if (type.getWidth() < word.getWidth())
    return arith::ExtUIOp::create(rewriter, loc, word, amount);
  if (type.getWidth() > word.getWidth()) {
    auto maximum = arith::ConstantOp::create(
        rewriter, loc,
        rewriter.getIntegerAttr(
            type, APInt::getLowBitsSet(type.getWidth(), word.getWidth())));
    amount = arith::MinUIOp::create(rewriter, loc, amount, maximum);
    return arith::TruncIOp::create(rewriter, loc, word, amount);
  }
  return amount;
}

// The shift amount clamped to the handle width minus one, which the shift
// instructions accept.
Value clampedShift(PatternRewriter &rewriter, Location loc, Value amount) {
  unsigned width = cast<IntegerType>(amount.getType()).getWidth();
  auto limit = arith::ConstantIntOp::create(rewriter, loc, width - 1, width);
  return arith::MinUIOp::create(rewriter, loc, amount, limit);
}

struct ShLILowering : FastPathLowering<ShLILowering> {
  using FastPathLowering::FastPathLowering;
  using Op = ShLIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_shl_atomic" : "rcgmp_mpz_shl";
  }

  static Value buildRuntimeOperand(PatternRewriter &rewriter, Location loc,
                                   Value amount, IntegerType word) {
    return saturatingWord(rewriter, loc, amount, word);
  }

  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *, ValueRange operands) {
    Value bits = operands[0], amount = operands[1];
    unsigned width = cast<IntegerType>(bits.getType()).getWidth();
    auto one = arith::ConstantIntOp::create(rewriter, loc, 1, width);
    // 4*x shifted left keeps every bit, the sign included, exactly when x
    // shifted left is immediate, and then shifting back recovers it. Clamping
    // the amount keeps only zero, whose shift is zero for any amount.
    auto scaled = arith::XOrIOp::create(rewriter, loc, bits, one);
    Value shift = clampedShift(rewriter, loc, amount);
    auto shifted = arith::ShLIOp::create(rewriter, loc, scaled, shift);
    auto back = arith::ShRSIOp::create(rewriter, loc, shifted, shift);
    auto kept = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                      back, scaled);
    auto valid = arith::AndIOp::create(rewriter, loc, kept,
                                       allImmediate(rewriter, loc, bits));
    auto tagged = arith::OrIOp::create(rewriter, loc, shifted, one);
    return {{tagged}, valid};
  }
};

struct ShRSILowering : FastPathLowering<ShRSILowering> {
  using FastPathLowering::FastPathLowering;
  using Op = ShRSIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_shr_atomic" : "rcgmp_mpz_shr";
  }

  static Value buildRuntimeOperand(PatternRewriter &rewriter, Location loc,
                                   Value amount, IntegerType word) {
    return saturatingWord(rewriter, loc, amount, word);
  }

  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *, ValueRange operands) {
    Value bits = operands[0], amount = operands[1];
    unsigned width = cast<IntegerType>(bits.getType()).getWidth();
    // The tagged bits 4*x + 1 shifted right by n have floor(4*x / 2^n) above
    // the tag, as the odd tag never reaches a multiple of 2^n, and clearing
    // the two low bits leaves 4 * floor(x / 2^n). That never leaves the
    // immediate range, and a shift by the width or more leaves the sign.
    auto shifted = arith::ShRSIOp::create(rewriter, loc, bits,
                                          clampedShift(rewriter, loc, amount));
    auto mask = arith::ConstantIntOp::create(rewriter, loc, -4, width);
    auto scaled = arith::AndIOp::create(rewriter, loc, shifted, mask);
    auto one = arith::ConstantIntOp::create(rewriter, loc, 1, width);
    auto tagged = arith::OrIOp::create(rewriter, loc, scaled, one);
    return {{tagged}, allImmediate(rewriter, loc, bits)};
  }
};

enum class Rounding { truncate, floor, euclidean };

// Divides immediates, producing the quotient, the remainder, or both, rounded
// toward zero, toward negative infinity, or to a nonnegative remainder.
template <typename Derived, Rounding rounding, bool quotient, bool remainder>
struct DivisionLowering : FastPathLowering<Derived> {
  using FastPathLowering<Derived>::FastPathLowering;

  static DivZero getDivZero(Operation *op) {
    return cast<typename Derived::Op>(op).getDivZero();
  }

  // Undefined division by zero calls the trapping functions, which may still
  // terminate on it.
  static StringRef getRuntimeSymbol(Operation *op, bool atomic) {
    return getDivZero(op) == DivZero::lean ? Derived::getLeanRuntimeName(atomic)
                                           : Derived::getRuntimeName(atomic);
  }

  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *op, ValueRange operands) {
    DivZero divZero = getDivZero(op);
    Value lhs = operands[0], rhs = operands[1];
    auto type = cast<IntegerType>(lhs.getType());
    auto constant = [&](int64_t value) -> Value {
      return arith::ConstantIntOp::create(rewriter, loc, value,
                                          type.getWidth());
    };
    Value two = constant(2), zero = constant(0);
    // Euclidean division by a positive divisor is floored division, which
    // LLVM reduces to shifts and masks for a power of two.
    Rounding mode = rounding;
    if (auto divisor = op->getOperand(1).getDefiningOp<ConstantOp>();
        mode == Rounding::euclidean && divisor &&
        divisor.getValue().isStrictlyPositive() &&
        divisor.getValue().isPowerOf2())
      mode = Rounding::floor;
    // Untagging any bits leaves a value above the signed word minimum, so
    // dividing it by -1 never overflows the word.
    Value x = arith::ShRSIOp::create(rewriter, loc, lhs, two);
    Value y = arith::ShRSIOp::create(rewriter, loc, rhs, two);
    Value valid =
        FastPathLowering<Derived>::allImmediate(rewriter, loc, {lhs, rhs});
    // Dividing by zero is undefined even where the fast path is not taken, so
    // a zero divisor is replaced unless it is undefined anyway. A trapping
    // division replaces it by one and leaves zero to the runtime.
    Value nonzero;
    if (divZero != DivZero::undefined)
      nonzero = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne,
                                      y, zero);
    if (divZero == DivZero::trap) {
      y = arith::SelectOp::create(rewriter, loc, nonzero, y, constant(1));
      valid = arith::AndIOp::create(rewriter, loc, valid, nonzero);
    }
    // A lean division replaces a zero divisor of the word division alone by
    // the signed word minimum. Its magnitude exceeds those of untagged
    // immediates, so the truncated quotient is zero and the remainder x, as
    // lean requires. The rounding adjustments still see the zero divisor.
    auto wordDivisor = [&](Value value) -> Value {
      if (divZero != DivZero::lean)
        return value;
      auto minimum = arith::ConstantOp::create(
          rewriter, loc,
          rewriter.getIntegerAttr(type,
                                  APInt::getSignedMinValue(type.getWidth())));
      return arith::SelectOp::create(rewriter, loc, nonzero, value, minimum);
    };
    Value q, r;
    if (mode == Rounding::euclidean) {
      // Unsigned division is faster than signed division. It divides x, or
      // |x| - 1 = ~x for a negative x, by |y|, giving q and r. The Euclidean
      // quotient is then q, -q, ~q = -q - 1, or q + 1 as x and y are
      // nonnegative, y alone is negative, x alone is, or both are: q
      // complemented when the signs differ, minus the sign mask of y. The
      // remainder is r, or |y| - 1 - r for a negative x.
      Value width = constant(type.getWidth() - 1);
      Value xSign = arith::ShRSIOp::create(rewriter, loc, x, width);
      Value ySign = arith::ShRSIOp::create(rewriter, loc, y, width);
      Value a = arith::XOrIOp::create(rewriter, loc, x, xSign);
      Value b = arith::SubIOp::create(
          rewriter, loc, arith::XOrIOp::create(rewriter, loc, y, ySign), ySign);
      // The minimum, read unsigned, also exceeds a, and a zero b leaves the
      // remainder x.
      Value d = wordDivisor(b);
      q = arith::DivUIOp::create(rewriter, loc, a, d);
      r = arith::RemUIOp::create(rewriter, loc, a, d);
      q = arith::SubIOp::create(
          rewriter, loc,
          arith::XOrIOp::create(
              rewriter, loc, q,
              arith::XOrIOp::create(rewriter, loc, xSign, ySign)),
          ySign);
      r = arith::AddIOp::create(rewriter, loc,
                                arith::XOrIOp::create(rewriter, loc, r, xSign),
                                arith::AndIOp::create(rewriter, loc, b, xSign));
    } else {
      Value d = wordDivisor(y);
      q = arith::DivSIOp::create(rewriter, loc, x, d);
      r = arith::RemSIOp::create(rewriter, loc, x, d);
    }
    if (mode == Rounding::floor) {
      // A nonzero remainder whose sign differs from the divisor's moves the
      // quotient down by one and the remainder by one divisor.
      auto inexact = arith::CmpIOp::create(rewriter, loc,
                                           arith::CmpIPredicate::ne, r, zero);
      auto signs = arith::XOrIOp::create(rewriter, loc, r, y);
      auto differ = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::slt, signs, zero);
      auto adjust = arith::AndIOp::create(rewriter, loc, inexact, differ);
      q = arith::SubIOp::create(
          rewriter, loc, q,
          arith::ExtUIOp::create(rewriter, loc, type, adjust));
      r = arith::AddIOp::create(
          rewriter, loc, r,
          arith::SelectOp::create(rewriter, loc, adjust, y, zero));
    }
    // Rounding a negative x moves its quotient by zero away from zero.
    if (divZero == DivZero::lean && mode != Rounding::truncate)
      q = arith::SelectOp::create(rewriter, loc, nonzero, q, zero);
    auto tag = [&](Value value) -> Value {
      auto scaled = arith::ShLIOp::create(rewriter, loc, value, two);
      return arith::OrIOp::create(rewriter, loc, scaled, constant(1));
    };
    FastPath fast;
    if (quotient) {
      // Only the quotient of the immediate minimum by -1 leaves the immediate
      // range; remainders are smaller than the divisor. Testing the operands
      // rather than the quotient leaves the division to the fast path.
      auto minimum = arith::ConstantOp::create(
          rewriter, loc,
          rewriter.getIntegerAttr(
              type, APInt::getSignedMinValue(type.getWidth()).ashr(2)));
      auto fits = arith::OrIOp::create(
          rewriter, loc,
          arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne, x,
                                minimum),
          arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne, y,
                                constant(-1)));
      valid = arith::AndIOp::create(rewriter, loc, valid, fits);
      fast.values.push_back(tag(q));
    }
    if (remainder)
      fast.values.push_back(tag(r));
    fast.valid = valid;
    return fast;
  }
};

struct DivSILowering
    : DivisionLowering<DivSILowering, Rounding::truncate, true, false> {
  using DivisionLowering::DivisionLowering;
  using Op = DivSIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_tdiv_q_atomic" : "rcgmp_mpz_tdiv_q";
  }

  static StringRef getLeanRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_tdiv_q_lean_atomic" : "rcgmp_mpz_tdiv_q_lean";
  }
};

struct RemSILowering
    : DivisionLowering<RemSILowering, Rounding::truncate, false, true> {
  using DivisionLowering::DivisionLowering;
  using Op = RemSIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_tdiv_r_atomic" : "rcgmp_mpz_tdiv_r";
  }

  static StringRef getLeanRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_tdiv_r_lean_atomic" : "rcgmp_mpz_tdiv_r_lean";
  }
};

struct FloorDivSILowering
    : DivisionLowering<FloorDivSILowering, Rounding::floor, true, false> {
  using DivisionLowering::DivisionLowering;
  using Op = FloorDivSIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_fdiv_q_atomic" : "rcgmp_mpz_fdiv_q";
  }

  static StringRef getLeanRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_fdiv_q_lean_atomic" : "rcgmp_mpz_fdiv_q_lean";
  }
};

struct FloorModSILowering
    : DivisionLowering<FloorModSILowering, Rounding::floor, false, true> {
  using DivisionLowering::DivisionLowering;
  using Op = FloorModSIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_fdiv_r_atomic" : "rcgmp_mpz_fdiv_r";
  }

  static StringRef getLeanRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_fdiv_r_lean_atomic" : "rcgmp_mpz_fdiv_r_lean";
  }
};

struct DivRemSILowering
    : DivisionLowering<DivRemSILowering, Rounding::truncate, true, true> {
  using DivisionLowering::DivisionLowering;
  using Op = DivRemSIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_tdiv_qr_atomic" : "rcgmp_mpz_tdiv_qr";
  }

  static StringRef getLeanRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_tdiv_qr_lean_atomic" : "rcgmp_mpz_tdiv_qr_lean";
  }
};

struct FloorDivModSILowering
    : DivisionLowering<FloorDivModSILowering, Rounding::floor, true, true> {
  using DivisionLowering::DivisionLowering;
  using Op = FloorDivModSIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_fdiv_qr_atomic" : "rcgmp_mpz_fdiv_qr";
  }

  static StringRef getLeanRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_fdiv_qr_lean_atomic" : "rcgmp_mpz_fdiv_qr_lean";
  }
};

struct EDivSILowering
    : DivisionLowering<EDivSILowering, Rounding::euclidean, true, false> {
  using DivisionLowering::DivisionLowering;
  using Op = EDivSIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_ediv_q_atomic" : "rcgmp_mpz_ediv_q";
  }

  static StringRef getLeanRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_ediv_q_lean_atomic" : "rcgmp_mpz_ediv_q_lean";
  }
};

struct EModSILowering
    : DivisionLowering<EModSILowering, Rounding::euclidean, false, true> {
  using DivisionLowering::DivisionLowering;
  using Op = EModSIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_ediv_r_atomic" : "rcgmp_mpz_ediv_r";
  }

  static StringRef getLeanRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_ediv_r_lean_atomic" : "rcgmp_mpz_ediv_r_lean";
  }
};

struct EDivModSILowering
    : DivisionLowering<EDivModSILowering, Rounding::euclidean, true, true> {
  using DivisionLowering::DivisionLowering;
  using Op = EDivModSIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_ediv_qr_atomic" : "rcgmp_mpz_ediv_qr";
  }

  static StringRef getLeanRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_ediv_qr_lean_atomic" : "rcgmp_mpz_ediv_qr_lean";
  }
};

// An integer truncated to a fixed width, and with withOverflow, whether it
// lies outside the signed or unsigned range of that width.
struct Truncation {
  Value value;
  Value overflow;
};

// Truncates the immediate in tagged bits. The truncation carries no overflow
// flags: arith canonicalization rewrites trunci(shrsi) to trunci(shrui) and
// keeps them, though the logical shift falsifies nsw for negative integers.
Truncation truncateImmediate(PatternRewriter &rewriter, Location loc,
                             Value bits, IntegerType type, bool isUnsigned,
                             bool withOverflow) {
  unsigned word = cast<IntegerType>(bits.getType()).getWidth();
  unsigned width = type.getWidth();
  auto two = arith::ConstantIntOp::create(rewriter, loc, 2, word);
  Value x = arith::ShRSIOp::create(rewriter, loc, bits, two);
  Value value = x;
  if (width < word)
    value = arith::TruncIOp::create(rewriter, loc, type, x);
  else if (width > word)
    value = arith::ExtSIOp::create(rewriter, loc, type, x);
  if (!withOverflow)
    return {value, nullptr};
  // An immediate has word - 2 signed bits, and word - 3 bits when nonnegative.
  Value overflow;
  if (width >= word - (isUnsigned ? 3 : 2)) {
    overflow = isUnsigned
                   ? Value(arith::CmpIOp::create(
                         rewriter, loc, arith::CmpIPredicate::slt, x,
                         arith::ConstantIntOp::create(rewriter, loc, 0, word)))
                   : arith::ConstantIntOp::create(rewriter, loc, 0, 1);
  } else {
    Value back =
        isUnsigned
            ? Value(arith::ExtUIOp::create(rewriter, loc, x.getType(), value))
            : arith::ExtSIOp::create(rewriter, loc, x.getType(), value);
    overflow =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne, back, x);
  }
  return {value, overflow};
}

// Truncates a heap integer from the limbs that the result width covers.
Truncation truncateHeap(PatternRewriter &rewriter, Location loc, Value integer,
                        IntegerType word, IntegerType type, bool isUnsigned,
                        bool withOverflow) {
  unsigned limbBits = word.getWidth(), width = type.getWidth();
  unsigned limbs = llvm::divideCeil(width, limbBits);
  auto wide = rewriter.getIntegerType(limbs * limbBits);
  auto constant = [&](IntegerType type, const APInt &value) -> Value {
    return arith::ConstantOp::create(rewriter, loc,
                                     rewriter.getIntegerAttr(type, value));
  };
  Value size = bridge::MpzSizeOp::create(rewriter, loc, word, integer);
  Value zero = constant(word, APInt(limbBits, 0));
  auto negative = arith::CmpIOp::create(rewriter, loc,
                                        arith::CmpIPredicate::slt, size, zero);
  Value count = arith::SelectOp::create(
      rewriter, loc, negative, arith::SubIOp::create(rewriter, loc, zero, size),
      size);
  // A heap integer has at least one limb, and a limb exists only below the
  // count.
  Value magnitude;
  for (unsigned i = 0; i < limbs; ++i) {
    Value index = arith::ConstantIndexOp::create(rewriter, loc, i);
    Value limb;
    if (!i) {
      limb = bridge::MpzLimbOp::create(rewriter, loc, word, integer, index);
    } else {
      auto present =
          arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ugt, count,
                                constant(word, APInt(limbBits, i)));
      auto branch = scf::IfOp::create(rewriter, loc, TypeRange{word}, present,
                                      /*withElseRegion=*/true);
      rewriter.setInsertionPointToStart(&branch.getThenRegion().front());
      scf::YieldOp::create(rewriter, loc,
                           ValueRange{bridge::MpzLimbOp::create(
                               rewriter, loc, word, integer, index)});
      rewriter.setInsertionPointToStart(&branch.getElseRegion().front());
      scf::YieldOp::create(rewriter, loc, ValueRange{zero});
      rewriter.setInsertionPointAfter(branch);
      limb = branch.getResult(0);
    }
    if (limbs > 1)
      limb = arith::ExtUIOp::create(rewriter, loc, wide, limb);
    if (i)
      limb = arith::ShLIOp::create(
          rewriter, loc, limb,
          constant(wide, APInt(wide.getWidth(), i * limbBits)));
    magnitude = i ? arith::OrIOp::create(rewriter, loc, magnitude, limb) : limb;
  }
  Value wideZero = constant(wide, APInt(wide.getWidth(), 0));
  Value value = arith::SelectOp::create(
      rewriter, loc, negative,
      arith::SubIOp::create(rewriter, loc, wideZero, magnitude), magnitude);
  if (width < wide.getWidth())
    value = arith::TruncIOp::create(rewriter, loc, type, value);
  if (!withOverflow)
    return {value, nullptr};
  // The magnitude is exact unless the integer has more limbs than were read.
  Value overflow =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ugt, count,
                            constant(word, APInt(limbBits, limbs)));
  Value outside;
  if (isUnsigned) {
    outside = negative;
    if (width < wide.getWidth()) {
      auto above = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::uge, magnitude,
          constant(wide, APInt::getOneBitSet(wide.getWidth(), width)));
      outside = arith::OrIOp::create(rewriter, loc, outside, above);
    }
  } else {
    // A negative magnitude may reach 2^(width - 1), a nonnegative one not.
    Value limit =
        constant(wide, APInt::getOneBitSet(wide.getWidth(), width - 1));
    auto above = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ugt,
                                       magnitude, limit);
    auto reaches = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::uge, magnitude, limit);
    outside = arith::SelectOp::create(rewriter, loc, negative, above, reaches);
  }
  overflow = arith::OrIOp::create(rewriter, loc, overflow, outside);
  return {value, overflow};
}

// Lowers rcgmp.trunci and rcgmp.trunci_overflow inline, reading immediate bits
// or heap limbs.
template <typename TruncOp>
struct TruncationLowering : OpRewritePattern<TruncOp> {
  using OpRewritePattern<TruncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TruncOp op,
                                PatternRewriter &rewriter) const override {
    constexpr bool withOverflow = std::is_same_v<TruncOp, TruncIOverflowOp>;
    auto loc = op.getLoc();
    auto type = cast<IntegerType>(op.getResult().getType());
    Value integer = op.getValue();
    unsigned width = DataLayout::closest(op)
                         .getTypeSizeInBits(integer.getType())
                         .getFixedValue();
    auto word = rewriter.getIntegerType(width);
    bool isUnsigned = false;
    auto flags = arith::IntegerOverflowFlags::none;
    if constexpr (withOverflow)
      isUnsigned = op.getIsUnsigned();
    else
      flags = op.getOverflowFlags();
    Value bits = bridge::MpzToBitsOp::create(rewriter, loc, word, integer);
    // Canonical integers that fit word - 2 signed bits, or word - 3 unsigned
    // ones, are immediate, so a flag promising such a fit rules out the heap.
    bool nsw = bitEnumContainsAll(flags, arith::IntegerOverflowFlags::nsw);
    bool nuw = bitEnumContainsAll(flags, arith::IntegerOverflowFlags::nuw);
    if ((nsw && type.getWidth() <= width - 2) ||
        (nuw && type.getWidth() <= width - 3)) {
      rewriter.replaceOp(
          op, truncateImmediate(rewriter, loc, bits, type, false, false).value);
      return success();
    }
    SmallVector<Type> types{type};
    if (withOverflow)
      types.push_back(rewriter.getI1Type());
    auto immediate = bridge::MpzIsImmediateOp::create(rewriter, loc, integer);
    auto branch = scf::IfOp::create(rewriter, loc, types, immediate,
                                    /*withElseRegion=*/true);
    auto yield = [&](Truncation truncation) {
      SmallVector<Value> results{truncation.value};
      if (withOverflow)
        results.push_back(truncation.overflow);
      scf::YieldOp::create(rewriter, loc, results);
    };
    rewriter.setInsertionPointToStart(&branch.getThenRegion().front());
    yield(
        truncateImmediate(rewriter, loc, bits, type, isUnsigned, withOverflow));
    rewriter.setInsertionPointToStart(&branch.getElseRegion().front());
    yield(truncateHeap(rewriter, loc, integer, word, type, isUnsigned,
                       withOverflow));
    rewriter.replaceOp(op, branch.getResults());
    return success();
  }
};

// Lowers rcgmp.log2i inline. The magnitude of an integer is a word m preceded
// by k words, so its logarithm is W * k plus that of m, with W the handle
// width; the logarithm of a nonzero m is W - 1 minus its leading zeros. Each
// path yields k and m, so that one count of leading zeros serves both.
struct Log2ILowering : OpRewritePattern<Log2IOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Log2IOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = cast<IntegerType>(op.getType());
    Value integer = op.getValue();
    unsigned width = DataLayout::closest(op)
                         .getTypeSizeInBits(integer.getType())
                         .getFixedValue();
    auto word = rewriter.getIntegerType(width);
    auto constant = [&](IntegerType type, int64_t value) -> Value {
      return arith::ConstantIntOp::create(rewriter, loc, type, value);
    };
    auto immediate = bridge::MpzIsImmediateOp::create(rewriter, loc, integer);
    auto branch = scf::IfOp::create(rewriter, loc, TypeRange{word, word},
                                    immediate, /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(&branch.getThenRegion().front());
    // Untagging leaves a value above the signed word minimum, whose absolute
    // value does not overflow.
    Value bits = bridge::MpzToBitsOp::create(rewriter, loc, word, integer);
    Value x = arith::ShRSIOp::create(rewriter, loc, bits, constant(word, 2));
    scf::YieldOp::create(
        rewriter, loc,
        ValueRange{constant(word, 0), math::AbsIOp::create(rewriter, loc, x)});
    rewriter.setInsertionPointToStart(&branch.getElseRegion().front());
    // The most significant limb of a heap integer is nonzero.
    Value size = bridge::MpzSizeOp::create(rewriter, loc, word, integer);
    Value top = arith::SubIOp::create(rewriter, loc,
                                      math::AbsIOp::create(rewriter, loc, size),
                                      constant(word, 1));
    Value limb = bridge::MpzLimbOp::create(
        rewriter, loc, word, integer,
        arith::IndexCastUIOp::create(rewriter, loc, rewriter.getIndexType(),
                                     top));
    scf::YieldOp::create(rewriter, loc, ValueRange{top, limb});
    rewriter.setInsertionPointAfter(branch);
    // Setting the lowest bit of m maps zero to one, whose logarithm is also
    // zero, and changes no other logarithm. It also proves m nonzero to LLVM,
    // which then counts leading zeros without a zero check.
    Value m = arith::OrIOp::create(rewriter, loc, branch.getResult(1),
                                   constant(word, 1));
    Value zeros = math::CountLeadingZerosOp::create(rewriter, loc, m);
    Value low =
        arith::SubIOp::create(rewriter, loc, constant(word, width - 1), zeros);
    // A result wider than the handle holds every logarithm, which W * k may
    // not fit the handle for.
    auto wide = type.getWidth() > width ? type : word;
    Value k = branch.getResult(0);
    if (wide != word) {
      k = arith::ExtUIOp::create(rewriter, loc, wide, k);
      low = arith::ExtUIOp::create(rewriter, loc, wide, low);
    }
    Value log = arith::AddIOp::create(
        rewriter, loc,
        arith::MulIOp::create(rewriter, loc, k, constant(wide, width)), low);
    if (type.getWidth() < width)
      log = arith::TruncIOp::create(rewriter, loc, type, log);
    rewriter.replaceOp(op, log);
    return success();
  }
};

struct CmpILowering : FastPathLowering<CmpILowering> {
  using FastPathLowering::FastPathLowering;
  using Op = CmpIOp;

  static constexpr bool consumesOperands = false;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_cmp_atomic" : "rcgmp_mpz_cmp";
  }

  static Type getRuntimeResultType(Operation *op) {
    return IntegerType::get(op->getContext(), 32);
  }

  // The runtime result is negative, zero, or positive, like mpz_cmp, so it
  // relates to zero as lhs relates to rhs.
  static Value buildSlowResult(PatternRewriter &rewriter, Location loc,
                               Operation *op, Value order) {
    auto zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    return arith::CmpIOp::create(rewriter, loc, getPredicate(op), order, zero);
  }

  // Signed limb counts order integers, an immediate lying between the
  // negative and the positive heap integers as count zero. Heap integers of
  // equal count share a sign, and their most significant limbs usually differ,
  // ordering their magnitudes; otherwise the runtime compares them.
  static Value buildSlowPath(PatternRewriter &rewriter, Location loc,
                             Operation *op, function_ref<Value()> callRuntime) {
    auto compare = cast<CmpIOp>(op);
    unsigned width = DataLayout::closest(op)
                         .getTypeSizeInBits(compare.getLhs().getType())
                         .getFixedValue();
    auto word = rewriter.getIntegerType(width);
    auto count = [&](Value value) -> Value {
      auto immediate = bridge::MpzIsImmediateOp::create(rewriter, loc, value);
      auto branch = scf::IfOp::create(rewriter, loc, TypeRange{word}, immediate,
                                      /*withElseRegion=*/true);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&branch.getThenRegion().front());
      scf::YieldOp::create(
          rewriter, loc,
          ValueRange{arith::ConstantIntOp::create(rewriter, loc, 0, width)});
      rewriter.setInsertionPointToStart(&branch.getElseRegion().front());
      scf::YieldOp::create(
          rewriter, loc,
          ValueRange{bridge::MpzSizeOp::create(rewriter, loc, word, value)});
      return branch.getResult(0);
    };
    Value lhs = count(compare.getLhs()), rhs = count(compare.getRhs());
    auto equal = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                       lhs, rhs);
    auto branch =
        scf::IfOp::create(rewriter, loc, TypeRange{rewriter.getI1Type()}, equal,
                          /*withElseRegion=*/true);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&branch.getThenRegion().front());
    scf::YieldOp::create(
        rewriter, loc,
        ValueRange{compareTopLimbs(rewriter, loc, op, lhs, callRuntime)});
    rewriter.setInsertionPointToStart(&branch.getElseRegion().front());
    scf::YieldOp::create(rewriter, loc,
                         ValueRange{arith::CmpIOp::create(
                             rewriter, loc, getPredicate(op), lhs, rhs)});
    return branch.getResult(0);
  }

  // Compares heap integers of the signed limb count count.
  static Value compareTopLimbs(PatternRewriter &rewriter, Location loc,
                               Operation *op, Value count,
                               function_ref<Value()> callRuntime) {
    auto compare = cast<CmpIOp>(op);
    auto word = cast<IntegerType>(count.getType());
    unsigned width = word.getWidth();
    auto zero = arith::ConstantIntOp::create(rewriter, loc, 0, width);
    auto negative = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::slt, count, zero);
    auto negated = arith::SubIOp::create(rewriter, loc, zero, count);
    auto magnitude =
        arith::SelectOp::create(rewriter, loc, negative, negated, count);
    auto one = arith::ConstantIntOp::create(rewriter, loc, 1, width);
    Value top = arith::IndexCastUIOp::create(
        rewriter, loc, rewriter.getIndexType(),
        arith::SubIOp::create(rewriter, loc, magnitude, one));
    Value lhs =
        bridge::MpzLimbOp::create(rewriter, loc, word, compare.getLhs(), top);
    Value rhs =
        bridge::MpzLimbOp::create(rewriter, loc, word, compare.getRhs(), top);
    auto differ = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne,
                                        lhs, rhs);
    auto branch = scf::IfOp::create(rewriter, loc,
                                    TypeRange{rewriter.getI1Type()}, differ,
                                    /*withElseRegion=*/true);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&branch.getThenRegion().front());
    // The larger limb makes the larger magnitude, which is the smaller
    // integer when negative.
    auto above = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ugt,
                                       lhs, rhs);
    auto greater = arith::XOrIOp::create(rewriter, loc, above, negative);
    Value order = arith::SelectOp::create(
        rewriter, loc, greater,
        arith::ConstantIntOp::create(rewriter, loc, 1, 32),
        arith::ConstantIntOp::create(rewriter, loc, -1, 32));
    scf::YieldOp::create(rewriter, loc,
                         ValueRange{buildSlowResult(rewriter, loc, op, order)});
    rewriter.setInsertionPointToStart(&branch.getElseRegion().front());
    scf::YieldOp::create(
        rewriter, loc,
        ValueRange{buildSlowResult(rewriter, loc, op, callRuntime())});
    return branch.getResult(0);
  }

  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *op, ValueRange operands) {
    Value lhs = operands[0], rhs = operands[1];
    // 4*n+1 is strictly increasing and fits the signed word over the immediate
    // range, so tagged immediates compare as their values do.
    auto result =
        arith::CmpIOp::create(rewriter, loc, getPredicate(op), lhs, rhs);
    return {{result}, allImmediate(rewriter, loc, {lhs, rhs})};
  }

private:
  static arith::CmpIPredicate getPredicate(Operation *op) {
    switch (cast<CmpIOp>(op).getPredicate()) {
    case CmpIPredicate::eq:
      return arith::CmpIPredicate::eq;
    case CmpIPredicate::ne:
      return arith::CmpIPredicate::ne;
    case CmpIPredicate::slt:
      return arith::CmpIPredicate::slt;
    case CmpIPredicate::sle:
      return arith::CmpIPredicate::sle;
    case CmpIPredicate::sgt:
      return arith::CmpIPredicate::sgt;
    case CmpIPredicate::sge:
      return arith::CmpIPredicate::sge;
    }
    llvm_unreachable("unknown comparison predicate");
  }
};

struct ConvertRCGMPToSCFPass
    : PassWrapper<ConvertRCGMPToSCFPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertRCGMPToSCFPass)
  StringRef getArgument() const override { return "convert-rcgmp-to-scf"; }
  StringRef getDescription() const override {
    return "Expand rcgmp operations into SCF and low-level bridge operations";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<RCGMPDialect, scf::SCFDialect, arith::ArithDialect,
                    cf::ControlFlowDialect, func::FuncDialect,
                    math::MathDialect, memref::MemRefDialect, ub::UBDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConstantLowering, ExtSILowering, ExtUILowering, RetainLowering,
                 ReleaseLowering, AddILowering, SubILowering, MonusILowering,
                 MulILowering, GcdILowering, AndILowering, XOrILowering,
                 OrILowering, NotILowering, NegILowering, AbsILowering,
                 ShLILowering, ShRSILowering, DivSILowering, RemSILowering,
                 FloorDivSILowering, FloorModSILowering, DivRemSILowering,
                 FloorDivModSILowering, EDivSILowering, EModSILowering,
                 EDivModSILowering, CmpILowering, TruncationLowering<TruncIOp>,
                 TruncationLowering<TruncIOverflowOp>, Log2ILowering>(
        &getContext());
    populateMatchLoweringPatterns(patterns);
    ConversionTarget target(getContext());
    target.addIllegalOp<
        ConstantOp, ExtSIOp, ExtUIOp, RetainOp, ReleaseOp, AddIOp, SubIOp,
        MonusIOp, MulIOp, GcdIOp, AndIOp, XOrIOp, OrIOp, NotIOp, NegIOp, AbsIOp,
        ShLIOp, ShRSIOp, DivSIOp, RemSIOp, FloorDivSIOp, FloorModSIOp,
        DivRemSIOp, FloorDivModSIOp, EDivSIOp, EModSIOp, EDivModSIOp, CmpIOp,
        TruncIOp, TruncIOverflowOp, Log2IOp, MatchOp, YieldOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createConvertRCGMPToSCFPass() {
  return std::make_unique<ConvertRCGMPToSCFPass>();
}

} // namespace mlir::rcgmp
