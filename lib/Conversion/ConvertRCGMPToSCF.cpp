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
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Matchers.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/TypeSwitch.h>

#include <bit>
#include <cmath>

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

struct FloorMod2ExpILowering : FastPathLowering<FloorMod2ExpILowering> {
  using FastPathLowering::FastPathLowering;
  using Op = FloorMod2ExpIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_fdiv_r_2exp_atomic" : "rcgmp_mpz_fdiv_r_2exp";
  }

  static Value buildRuntimeOperand(PatternRewriter &rewriter, Location loc,
                                   Value amount, IntegerType word) {
    return saturatingWord(rewriter, loc, amount, word);
  }

  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *, ValueRange operands) {
    Value bits = operands[0], amount = operands[1];
    unsigned width = cast<IntegerType>(bits.getType()).getWidth();
    auto constant = [&](int64_t value) -> Value {
      return arith::ConstantIntOp::create(rewriter, loc, value, width);
    };
    // The tagged bits 4*x + 1 have bit 1 clear, so that their n + 2 low bits
    // are 4*(x mod 2^n) + 1. An amount of W - 2 or more keeps all bits, the
    // sign included, as 4 << (W - 2) wraps to zero.
    auto shift =
        arith::MinUIOp::create(rewriter, loc, amount, constant(width - 2));
    auto power = arith::ShLIOp::create(rewriter, loc, constant(4), shift);
    auto mask = arith::SubIOp::create(rewriter, loc, power, constant(1));
    auto tagged = arith::AndIOp::create(rewriter, loc, bits, mask);
    // Only the remainder of a negative x by 2^n with n at least W - 2 leaves
    // the immediate range, as it is at least 2^(W-3), and only then are the
    // tagged bits negative.
    auto fits = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sge,
                                      tagged, constant(0));
    auto valid = arith::AndIOp::create(rewriter, loc, fits,
                                       allImmediate(rewriter, loc, bits));
    return {{tagged}, valid};
  }

  // Amounts up to W - 3 keep every immediate on the fast path, so that only
  // heap integers reach this one with them. Their remainder is immediate, the
  // low bits of the low limb, negated for a negative integer, and the integer
  // is released without calling the runtime.
  static Value buildSlowPath(PatternRewriter &rewriter, Location loc,
                             Operation *op, function_ref<Value()> callRuntime) {
    auto remainder = cast<FloorMod2ExpIOp>(op);
    Value value = remainder.getValue();
    Type type = value.getType();
    unsigned width =
        DataLayout::closest(op).getTypeSizeInBits(type).getFixedValue();
    auto word = rewriter.getIntegerType(width);
    auto constant = [&](int64_t value) -> Value {
      return arith::ConstantIntOp::create(rewriter, loc, value, width);
    };
    Value amount = saturatingWord(rewriter, loc, remainder.getAmount(), word);
    auto immediate = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ule, amount, constant(width - 3));
    auto branch = scf::IfOp::create(rewriter, loc, TypeRange{type}, immediate,
                                    /*withElseRegion=*/true);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&branch.getThenRegion().front());
    Value size = bridge::MpzSizeOp::create(rewriter, loc, word, value);
    Value limb = bridge::MpzLimbOp::create(
        rewriter, loc, word, value,
        arith::ConstantIndexOp::create(rewriter, loc, 0));
    auto negative = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::slt, size, constant(0));
    Value low = arith::SelectOp::create(
        rewriter, loc, negative,
        arith::SubIOp::create(rewriter, loc, constant(0), limb), limb);
    auto power = arith::ShLIOp::create(rewriter, loc, constant(1), amount);
    auto mask = arith::SubIOp::create(rewriter, loc, power, constant(1));
    auto kept = arith::AndIOp::create(rewriter, loc, low, mask);
    auto tagged = arith::OrIOp::create(
        rewriter, loc, arith::ShLIOp::create(rewriter, loc, kept, constant(2)),
        constant(1));
    ReleaseOp::create(rewriter, loc, value);
    scf::YieldOp::create(
        rewriter, loc,
        ValueRange{bridge::MpzFromBitsOp::create(rewriter, loc, type, tagged)});
    rewriter.setInsertionPointToStart(&branch.getElseRegion().front());
    scf::YieldOp::create(rewriter, loc, ValueRange{callRuntime()});
    return branch.getResult(0);
  }
};

// Exponents are unsigned. One wider than the handle saturates to the largest
// word of the same parity: only the powers of bases of magnitude at most one
// can be allocated for such exponents, and they depend on the parity alone.
Value saturatingExponent(PatternRewriter &rewriter, Location loc,
                         Value exponent, IntegerType word) {
  Value saturated = saturatingWord(rewriter, loc, exponent, word);
  if (cast<IntegerType>(exponent.getType()).getWidth() <= word.getWidth())
    return saturated;
  // The saturated word has all bits set, and the low bit of the exponent
  // keeps the low one.
  auto low = arith::TruncIOp::create(rewriter, loc, word, exponent);
  auto high = arith::ConstantIntOp::create(rewriter, loc, -2, word.getWidth());
  auto parity = arith::OrIOp::create(rewriter, loc, low, high);
  return arith::AndIOp::create(rewriter, loc, saturated, parity);
}

struct IPowILowering : FastPathLowering<IPowILowering> {
  using FastPathLowering::FastPathLowering;
  using Op = IPowIOp;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_pow_atomic" : "rcgmp_mpz_pow";
  }

  static Value buildRuntimeOperand(PatternRewriter &rewriter, Location loc,
                                   Value exponent, IntegerType word) {
    return saturatingExponent(rewriter, loc, exponent, word);
  }

  // A constant exponent from two up to the last at which a base of magnitude
  // two can have an immediate power unrolls into the products that square and
  // multiply, left to right. Other exponents take the power one for exponent
  // zero, the base for odd exponents, and its square for even ones, which
  // holds for exponents up to two and for bases of magnitude at most one,
  // whose squares never overflow.
  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *op, ValueRange operands) {
    Value bits = operands[0], exponent = operands[1];
    unsigned width = cast<IntegerType>(bits.getType()).getWidth();
    APInt value;
    if (matchPattern(op->getOperand(1), m_ConstantInt(&value)) &&
        value.uge(2) && value.ule(width - 3))
      return buildProducts(rewriter, loc, bits, value.getZExtValue());
    auto constant = [&](int64_t value) -> Value {
      return arith::ConstantIntOp::create(rewriter, loc, value, width);
    };
    // x * (4*x) fits the signed word iff x*x fits the immediate payload, as
    // for muli.
    auto x = arith::ShRSIOp::create(rewriter, loc, bits, constant(2));
    auto scaled = arith::AndIOp::create(rewriter, loc, bits, constant(-4));
    auto square = bridge::MulSIOverflowOp::create(
        rewriter, loc, bits.getType(), rewriter.getI1Type(), x, scaled);
    auto squared =
        arith::OrIOp::create(rewriter, loc, square.getProduct(), constant(1));
    auto odd =
        arith::TruncIOp::create(rewriter, loc, rewriter.getI1Type(), exponent);
    auto power = arith::SelectOp::create(rewriter, loc, odd, bits, squared);
    auto zero = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                      exponent, constant(0));
    auto tagged =
        arith::SelectOp::create(rewriter, loc, zero, constant(5), power);
    // Of the immediates, 4*x+1 + 3 is 0, 4, or 8 exactly for x in {-1, 0, 1}.
    auto shifted = arith::AddIOp::create(rewriter, loc, bits, constant(3));
    auto unit = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ule,
                                      shifted, constant(8));
    auto low = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ule,
                                     exponent, constant(1));
    auto two = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                     exponent, constant(2));
    auto yes = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
    auto fits = arith::XOrIOp::create(rewriter, loc, square.getOverflow(), yes);
    auto exact = arith::OrIOp::create(
        rewriter, loc, arith::OrIOp::create(rewriter, loc, low, unit),
        arith::AndIOp::create(rewriter, loc, two, fits));
    auto valid = arith::AndIOp::create(rewriter, loc, exact,
                                       allImmediate(rewriter, loc, bits));
    return {{tagged}, valid};
  }

  // Each product of an immediate x and a scaled power 4*y is 4*x*y, which
  // fits the signed word iff x*y fits the immediate payload, as for muli.
  static FastPath buildProducts(PatternRewriter &rewriter, Location loc,
                                Value bits, std::uint64_t exponent) {
    unsigned width = cast<IntegerType>(bits.getType()).getWidth();
    auto constant = [&](int64_t value) -> Value {
      return arith::ConstantIntOp::create(rewriter, loc, value, width);
    };
    Value x = arith::ShRSIOp::create(rewriter, loc, bits, constant(2));
    Value power = arith::AndIOp::create(rewriter, loc, bits, constant(-4));
    Value overflow;
    auto multiply = [&](Value lhs) {
      auto product = bridge::MulSIOverflowOp::create(
          rewriter, loc, bits.getType(), rewriter.getI1Type(), lhs, power);
      power = product.getProduct();
      overflow = overflow ? Value(arith::OrIOp::create(rewriter, loc, overflow,
                                                       product.getOverflow()))
                          : product.getOverflow();
    };
    for (int bit = std::bit_width(exponent) - 2; bit >= 0; --bit) {
      multiply(arith::ShRSIOp::create(rewriter, loc, power, constant(2)));
      if (exponent >> bit & 1)
        multiply(x);
    }
    auto yes = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
    auto fits = arith::XOrIOp::create(rewriter, loc, overflow, yes);
    auto valid = arith::AndIOp::create(rewriter, loc, fits,
                                       allImmediate(rewriter, loc, bits));
    auto tagged = arith::OrIOp::create(rewriter, loc, power, constant(1));
    return {{tagged}, valid};
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

  // Whether the operation is an exact quotient, which does not depend on its
  // rounding.
  static bool isExact(Operation *op) {
    if constexpr (quotient && !remainder)
      return cast<typename Derived::Op>(op).getExact();
    return false;
  }

  // Undefined division by zero calls the trapping functions, which may still
  // terminate on it. So do exact lean divisions, as their only defined
  // division by zero divides zero, which the fast path does.
  static StringRef getRuntimeSymbol(Operation *op, bool atomic) {
    if (isExact(op))
      return atomic ? "rcgmp_mpz_tdiv_q_exact_atomic"
                    : "rcgmp_mpz_tdiv_q_exact";
    return getDivZero(op) == DivZero::lean ? Derived::getLeanRuntimeName(atomic)
                                           : Derived::getRuntimeName(atomic);
  }

  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *op, ValueRange operands) {
    DivZero divZero = getDivZero(op);
    bool exact = isExact(op);
    Value lhs = operands[0], rhs = operands[1];
    auto type = cast<IntegerType>(lhs.getType());
    auto constant = [&](int64_t value) -> Value {
      return arith::ConstantIntOp::create(rewriter, loc, value,
                                          type.getWidth());
    };
    Value two = constant(2), zero = constant(0);
    // An exact quotient is the truncated one, whose division LLVM reduces to
    // shifts and a multiplication by the inverse of a constant divisor.
    // Euclidean division by a positive divisor is floored division, which
    // LLVM reduces to shifts and masks for a power of two.
    Rounding mode = exact ? Rounding::truncate : rounding;
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
    // lean requires. The rounding adjustments still see the zero divisor. An
    // exact lean division by zero is defined only for x zero, which is a
    // multiple of the minimum.
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
    } else if (exact) {
      q = arith::DivSIOp::create(rewriter, loc, x, wordDivisor(y),
                                 /*isExact=*/true);
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

// Lowers rcgmp.testbiti, reading the bit from the immediate or from the limb
// that holds it, W the handle width. Only a negative integer whose bit lies
// above a zero low limb calls the runtime.
struct TestBitILowering : FastPathLowering<TestBitILowering> {
  using FastPathLowering::FastPathLowering;
  using Op = TestBitIOp;

  static constexpr bool consumesOperands = false;

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "rcgmp_mpz_tstbit_atomic" : "rcgmp_mpz_tstbit";
  }

  static Type getRuntimeResultType(Operation *op) {
    return IntegerType::get(op->getContext(), 32);
  }

  static Value buildSlowResult(PatternRewriter &rewriter, Location loc,
                               Operation *, Value bit) {
    auto zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    return arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne, bit,
                                 zero);
  }

  // Every bit beyond the magnitude is the sign, which a saturated index still
  // reaches.
  static Value buildRuntimeOperand(PatternRewriter &rewriter, Location loc,
                                   Value index, IntegerType word) {
    return saturatingWord(rewriter, loc, index, word);
  }

  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *, ValueRange operands) {
    Value bits = operands[0], index = operands[1];
    unsigned width = cast<IntegerType>(bits.getType()).getWidth();
    auto constant = [&](int64_t value) -> Value {
      return arith::ConstantIntOp::create(rewriter, loc, value, width);
    };
    // The tagged bits 4*x + 1 hold bit n of x at n + 2 for n below W - 2, and
    // the sign above. Shifting them right arithmetically by n, clamped to
    // W - 3, brings the bit to position 2.
    auto shift =
        arith::MinUIOp::create(rewriter, loc, index, constant(width - 3));
    auto shifted = arith::ShRSIOp::create(rewriter, loc, bits, shift);
    auto bit = arith::AndIOp::create(rewriter, loc, shifted, constant(4));
    auto set = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne,
                                     bit, constant(0));
    return {{set}, allImmediate(rewriter, loc, bits)};
  }

  // A nonnegative heap integer is its magnitude m in two's complement, and a
  // negative one is ~(m - 1). The limb of ~(m - 1) at a position is the
  // complement of that of m, plus one exactly when every lower limb of m is
  // zero, which is known without a scan at position zero or when limb 0 is
  // not zero. Beyond the magnitude, every bit is the sign.
  static Value buildSlowPath(PatternRewriter &rewriter, Location loc,
                             Operation *op, function_ref<Value()> callRuntime) {
    auto test = cast<TestBitIOp>(op);
    Value value = test.getValue();
    unsigned width = DataLayout::closest(op)
                         .getTypeSizeInBits(value.getType())
                         .getFixedValue();
    auto word = rewriter.getIntegerType(width);
    auto i1 = rewriter.getI1Type();
    auto constant = [&](int64_t value) -> Value {
      return arith::ConstantIntOp::create(rewriter, loc, value, width);
    };
    auto boolean = [&](bool value) -> Value {
      return arith::ConstantIntOp::create(rewriter, loc, value, 1);
    };
    auto limbAt = [&](Value index) -> Value {
      return bridge::MpzLimbOp::create(rewriter, loc, word, value, index);
    };
    Value index = saturatingWord(rewriter, loc, test.getIndex(), word);
    Value position =
        arith::DivUIOp::create(rewriter, loc, index, constant(width));
    Value offset =
        arith::RemUIOp::create(rewriter, loc, index, constant(width));
    Value limbIndex = arith::IndexCastUIOp::create(
        rewriter, loc, rewriter.getIndexType(), position);
    auto bitOf = [&](Value limb) -> Value {
      return arith::TruncIOp::create(
          rewriter, loc, i1,
          arith::ShRUIOp::create(rewriter, loc, limb, offset));
    };
    // Builds the bit of a magnitude of count limbs with inside when the
    // position lies within it, and yields the sign otherwise.
    auto withinMagnitude = [&](Value count, bool sign,
                               function_ref<Value()> inside) -> Value {
      auto within = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::ult, position, count);
      auto branch = scf::IfOp::create(rewriter, loc, TypeRange{i1}, within,
                                      /*withElseRegion=*/true);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&branch.getThenRegion().front());
      scf::YieldOp::create(rewriter, loc, ValueRange{inside()});
      rewriter.setInsertionPointToStart(&branch.getElseRegion().front());
      scf::YieldOp::create(rewriter, loc, ValueRange{boolean(sign)});
      return branch.getResult(0);
    };
    Value size = bridge::MpzSizeOp::create(rewriter, loc, word, value);
    auto negative = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::slt, size, constant(0));
    auto branch = scf::IfOp::create(rewriter, loc, TypeRange{i1}, negative,
                                    /*withElseRegion=*/true);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&branch.getThenRegion().front());
    Value count = arith::SubIOp::create(rewriter, loc, constant(0), size);
    scf::YieldOp::create(
        rewriter, loc, ValueRange{withinMagnitude(count, true, [&] {
          Value limb = limbAt(limbIndex);
          auto first = arith::CmpIOp::create(
              rewriter, loc, arith::CmpIPredicate::eq, position, constant(0));
          auto lowZero = arith::CmpIOp::create(
              rewriter, loc, arith::CmpIPredicate::eq,
              limbAt(arith::ConstantIndexOp::create(rewriter, loc, 0)),
              constant(0));
          auto scan = arith::AndIOp::create(
              rewriter, loc,
              arith::XOrIOp::create(rewriter, loc, first, boolean(true)),
              lowZero);
          auto local = scf::IfOp::create(rewriter, loc, TypeRange{i1}, scan,
                                         /*withElseRegion=*/true);
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(&local.getThenRegion().front());
          scf::YieldOp::create(
              rewriter, loc,
              ValueRange{buildSlowResult(rewriter, loc, op, callRuntime())});
          rewriter.setInsertionPointToStart(&local.getElseRegion().front());
          Value complement =
              arith::XOrIOp::create(rewriter, loc, limb, constant(-1));
          Value carry = arith::ExtUIOp::create(rewriter, loc, word, first);
          scf::YieldOp::create(rewriter, loc,
                               ValueRange{bitOf(arith::AddIOp::create(
                                   rewriter, loc, complement, carry))});
          return local.getResult(0);
        })});
    rewriter.setInsertionPointToStart(&branch.getElseRegion().front());
    scf::YieldOp::create(rewriter, loc,
                         ValueRange{withinMagnitude(size, false, [&] {
                           return bitOf(limbAt(limbIndex));
                         })});
    return branch.getResult(0);
  }
};

// Lowers rcgmp.limb_count inline. An immediate has one limb unless it is zero,
// whose tagged bits are 1, and a heap integer has as many as its signed count
// says.
struct LimbCountLowering : OpRewritePattern<LimbCountOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LimbCountOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value integer = op.getValue();
    unsigned width = DataLayout::closest(op)
                         .getTypeSizeInBits(integer.getType())
                         .getFixedValue();
    auto word = rewriter.getIntegerType(width);
    auto immediate = bridge::MpzIsImmediateOp::create(rewriter, loc, integer);
    auto branch = scf::IfOp::create(rewriter, loc, TypeRange{word}, immediate,
                                    /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(&branch.getThenRegion().front());
    Value bits = bridge::MpzToBitsOp::create(rewriter, loc, word, integer);
    auto nonzero = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ne, bits,
        arith::ConstantIntOp::create(rewriter, loc, word, 1));
    scf::YieldOp::create(
        rewriter, loc,
        ValueRange{arith::ExtUIOp::create(rewriter, loc, word, nonzero)});
    rewriter.setInsertionPointToStart(&branch.getElseRegion().front());
    Value size = bridge::MpzSizeOp::create(rewriter, loc, word, integer);
    scf::YieldOp::create(rewriter, loc,
                         ValueRange{math::AbsIOp::create(rewriter, loc, size)});
    rewriter.setInsertionPointAfter(branch);
    rewriter.replaceOpWithNewOp<arith::IndexCastUIOp>(
        op, rewriter.getIndexType(), branch.getResult(0));
    return success();
  }
};

// Lowers rcgmp.limb inline. An immediate's only limb is the absolute value of
// its payload, and a heap integer loads the limb if its index lies below the
// count. Every other limb is zero.
struct LimbLowering : OpRewritePattern<LimbOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LimbOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value integer = op.getValue(), index = op.getIndex();
    auto word = cast<IntegerType>(op.getType());
    auto zero = [&]() -> Value {
      return arith::ConstantIntOp::create(rewriter, loc, word, 0);
    };
    auto immediate = bridge::MpzIsImmediateOp::create(rewriter, loc, integer);
    auto branch = scf::IfOp::create(rewriter, loc, TypeRange{word}, immediate,
                                    /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(&branch.getThenRegion().front());
    // Untagging leaves a value above the signed word minimum, whose absolute
    // value does not overflow.
    Value bits = bridge::MpzToBitsOp::create(rewriter, loc, word, integer);
    Value payload = arith::ShRSIOp::create(
        rewriter, loc, bits,
        arith::ConstantIntOp::create(rewriter, loc, word, 2));
    auto first =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq, index,
                              arith::ConstantIndexOp::create(rewriter, loc, 0));
    scf::YieldOp::create(
        rewriter, loc,
        ValueRange{arith::SelectOp::create(
            rewriter, loc, first, math::AbsIOp::create(rewriter, loc, payload),
            zero())});
    rewriter.setInsertionPointToStart(&branch.getElseRegion().front());
    Value size = bridge::MpzSizeOp::create(rewriter, loc, word, integer);
    Value count =
        arith::IndexCastUIOp::create(rewriter, loc, rewriter.getIndexType(),
                                     math::AbsIOp::create(rewriter, loc, size));
    auto inside = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, index, count);
    auto load = scf::IfOp::create(rewriter, loc, TypeRange{word}, inside,
                                  /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&load.getThenRegion().front());
      scf::YieldOp::create(rewriter, loc,
                           ValueRange{bridge::MpzLimbOp::create(
                               rewriter, loc, word, integer, index)});
      rewriter.setInsertionPointToStart(&load.getElseRegion().front());
      scf::YieldOp::create(rewriter, loc, ValueRange{zero()});
    }
    scf::YieldOp::create(rewriter, loc, load.getResults());
    rewriter.replaceOp(op, branch.getResults());
    return success();
  }
};

// Lowers rcgmp.sitofp. An immediate converts inline, rounded as arith.sitofp
// rounds; the runtime rounds a heap integer.
struct SIToFPLowering : FastPathLowering<SIToFPLowering> {
  using FastPathLowering::FastPathLowering;
  using Op = SIToFPOp;

  static constexpr bool consumesOperands = false;

  static StringRef getRuntimeSymbol(Operation *op, bool atomic) {
    if (op->getResult(0).getType().isF32())
      return atomic ? "rcgmp_mpz_to_f32_atomic" : "rcgmp_mpz_to_f32";
    return atomic ? "rcgmp_mpz_to_f64_atomic" : "rcgmp_mpz_to_f64";
  }

  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *op, ValueRange operands) {
    Value bits = operands[0];
    auto word = cast<IntegerType>(bits.getType());
    Value value = arith::ShRSIOp::create(
        rewriter, loc, bits,
        arith::ConstantIntOp::create(rewriter, loc, word, 2));
    Value rounded = arith::SIToFPOp::create(rewriter, loc,
                                            op->getResult(0).getType(), value);
    return {{rounded}, allImmediate(rewriter, loc, bits)};
  }
};

// Lowers rcgmp.fptosi. A value whose magnitude is below the immediate bound
// 2^(W-3), W the handle width, truncates to an immediate inline; the runtime
// truncates any other, terminating on a NaN or an infinity, which fail the
// comparison.
struct FPToSILowering : OpRewritePattern<FPToSIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(FPToSIOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = op.getType();
    unsigned width =
        DataLayout::closest(op).getTypeSizeInBits(type).getFixedValue();
    auto word = rewriter.getIntegerType(width);
    auto f64 = rewriter.getF64Type();
    Value value = op.getValue();
    if (value.getType() != f64)
      value = arith::ExtFOp::create(rewriter, loc, f64, value);
    StringRef name =
        type.getAtomic() ? "rcgmp_mpz_from_f64_atomic" : "rcgmp_mpz_from_f64";
    FailureOr<func::FuncOp> runtime = lookupOrDeclareRuntime(
        rewriter, op, name, rewriter.getFunctionType({f64}, {type}));
    if (failed(runtime))
      return rewriter.notifyMatchFailure(op,
                                         "incompatible runtime declaration");
    auto bound = arith::ConstantOp::create(
        rewriter, loc, rewriter.getF64FloatAttr(std::ldexp(1.0, width - 3)));
    auto magnitude = math::AbsFOp::create(rewriter, loc, value);
    auto small = arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OLT,
                                       magnitude, bound);
    auto branch = scf::IfOp::create(rewriter, loc, TypeRange{type}, small,
                                    /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(&branch.getThenRegion().front());
    Value integer = arith::FPToSIOp::create(rewriter, loc, word, value);
    auto constant = [&](int64_t value) -> Value {
      return arith::ConstantIntOp::create(rewriter, loc, word, value);
    };
    Value tagged = arith::OrIOp::create(
        rewriter, loc,
        arith::ShLIOp::create(rewriter, loc, integer, constant(2)),
        constant(1));
    scf::YieldOp::create(
        rewriter, loc,
        ValueRange{bridge::MpzFromBitsOp::create(rewriter, loc, type, tagged)});
    rewriter.setInsertionPointToStart(&branch.getElseRegion().front());
    auto call = func::CallOp::create(rewriter, loc, *runtime, value);
    scf::YieldOp::create(rewriter, loc, call.getResults());
    rewriter.replaceOp(op, branch.getResults());
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

// How a fusible operation builds its fast path, from the operation's lowering.
struct Fusible {
  FastPath (*buildFastPath)(PatternRewriter &, Location, Operation *,
                            ValueRange);
  Value (*buildRuntimeOperand)(PatternRewriter &, Location, Value, IntegerType);
  bool consumesOperands;
};

template <typename Lowering> Fusible fusibleAs() {
  return {&Lowering::buildFastPath, &Lowering::buildRuntimeOperand,
          Lowering::consumesOperands};
}

// Operations with a single result whose fast path needs no memory access, so
// that it may run on the bits of another fast path's result.
std::optional<Fusible> getFusible(Operation *op) {
  if (op->getNumResults() != 1)
    return std::nullopt;
  return TypeSwitch<Operation *, std::optional<Fusible>>(op)
      .Case([](AddIOp) { return fusibleAs<AddILowering>(); })
      .Case([](SubIOp) { return fusibleAs<SubILowering>(); })
      .Case([](MonusIOp) { return fusibleAs<MonusILowering>(); })
      .Case([](MulIOp) { return fusibleAs<MulILowering>(); })
      .Case([](AndIOp) { return fusibleAs<AndILowering>(); })
      .Case([](XOrIOp) { return fusibleAs<XOrILowering>(); })
      .Case([](OrIOp) { return fusibleAs<OrILowering>(); })
      .Case([](NotIOp) { return fusibleAs<NotILowering>(); })
      .Case([](NegIOp) { return fusibleAs<NegILowering>(); })
      .Case([](AbsIOp) { return fusibleAs<AbsILowering>(); })
      .Case([](ShLIOp) { return fusibleAs<ShLILowering>(); })
      .Case([](ShRSIOp) { return fusibleAs<ShRSILowering>(); })
      .Case([](FloorMod2ExpIOp) { return fusibleAs<FloorMod2ExpILowering>(); })
      .Case([](DivSIOp) { return fusibleAs<DivSILowering>(); })
      .Case([](RemSIOp) { return fusibleAs<RemSILowering>(); })
      .Case([](FloorDivSIOp) { return fusibleAs<FloorDivSILowering>(); })
      .Case([](FloorModSIOp) { return fusibleAs<FloorModSILowering>(); })
      .Case([](EDivSIOp) { return fusibleAs<EDivSILowering>(); })
      .Case([](EModSIOp) { return fusibleAs<EModSILowering>(); })
      .Case([](CmpIOp) { return fusibleAs<CmpILowering>(); })
      .Default([](Operation *) { return std::nullopt; });
}

// Whether value, an operand of user, comes from an operation that joins the
// chain of user: one in the same block whose integer result user consumes,
// and nothing else uses, or user borrows, and only later releases use.
Operation *fusedProducer(Value value, Operation *user, bool userConsumes) {
  Operation *producer = value.getDefiningOp();
  if (!producer || producer->getBlock() != user->getBlock() ||
      !isa<MpzType>(value.getType()) || !getFusible(producer))
    return nullptr;
  if (userConsumes)
    return value.hasOneUse() ? producer : nullptr;
  bool onlyReleased = llvm::all_of(value.getUsers(), [&](Operation *other) {
    return other == user ||
           (isa<ReleaseOp>(other) && user->isBeforeInBlock(other));
  });
  return onlyReleased ? producer : nullptr;
}

struct FusionRewriter : PatternRewriter {
  explicit FusionRewriter(MLIRContext *context) : PatternRewriter(context) {}
};

// Lowers the chain ending at root as one fast path: the fast paths of its
// operations, applied to the bits of the results before them, and valid when
// all of them are. Taken one by one, every result would join its heap result
// in a phi that the next operation tests again. The slow path performs the
// original operations, which lower one by one. A result the root borrows is
// immediate on the fast path, so its later releases move to the slow path,
// right after the root, which no other use separates them from.
void fuseChain(Operation *root, PatternRewriter &rewriter,
               DenseSet<Operation *> &erased) {
  llvm::SetVector<Operation *> chain;
  SmallVector<std::pair<Operation *, bool>> worklist{
      {root, getFusible(root)->consumesOperands}};
  while (!worklist.empty()) {
    auto [user, consumes] = worklist.pop_back_val();
    for (Value operand : user->getOperands())
      if (Operation *producer = fusedProducer(operand, user, consumes))
        if (chain.insert(producer))
          worklist.push_back(
              {producer, getFusible(producer)->consumesOperands});
  }
  if (chain.empty())
    return;
  SmallVector<Operation *> order(chain.begin(), chain.end());
  llvm::sort(order,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  order.push_back(root);

  SmallVector<Value> outputs(root->getResults());
  SmallVector<Operation *> releases;
  for (Operation *op : chain)
    for (Operation *user : op->getUsers())
      if (isa<ReleaseOp>(user))
        releases.push_back(user);

  auto loc = root->getLoc();
  rewriter.setInsertionPoint(root);
  DenseMap<Value, Value> bits;
  Value valid;
  for (Operation *op : order) {
    Fusible fusible = *getFusible(op);
    auto handle = *llvm::find_if(op->getOperandTypes(), llvm::IsaPred<MpzType>);
    auto word = rewriter.getIntegerType(
        DataLayout::closest(op).getTypeSizeInBits(handle).getFixedValue());
    SmallVector<Value> operands;
    for (Value operand : op->getOperands()) {
      if (!isa<MpzType>(operand.getType())) {
        operands.push_back(
            fusible.buildRuntimeOperand(rewriter, loc, operand, word));
        continue;
      }
      Value &operandBits = bits[operand];
      if (!operandBits)
        operandBits = bridge::MpzToBitsOp::create(rewriter, loc, word, operand);
      operands.push_back(operandBits);
    }
    FastPath fast = fusible.buildFastPath(rewriter, loc, op, operands);
    bits[op->getResult(0)] = fast.values.front();
    valid = valid ? arith::AndIOp::create(rewriter, loc, valid, fast.valid)
                        .getResult()
                  : fast.valid;
  }

  auto branch =
      scf::IfOp::create(rewriter, loc, ValueRange(outputs).getTypes(), valid,
                        /*withElseRegion=*/true);
  rewriter.setInsertionPointToStart(&branch.getThenRegion().front());
  SmallVector<Value> fastResults;
  for (Value output : outputs)
    fastResults.push_back(isa<MpzType>(output.getType())
                              ? bridge::MpzFromBitsOp::create(rewriter, loc,
                                                              output.getType(),
                                                              bits[output])
                                    .getResult()
                              : bits[output]);
  scf::YieldOp::create(rewriter, loc, fastResults);
  rewriter.setInsertionPointToStart(&branch.getElseRegion().front());
  IRMapping mapping;
  for (Operation *op : order)
    rewriter.clone(*op, mapping);
  for (Operation *release : releases) {
    rewriter.clone(*release, mapping);
    rewriter.eraseOp(release);
  }
  SmallVector<Value> slowResults;
  for (Value output : outputs)
    slowResults.push_back(mapping.lookup(output));
  scf::YieldOp::create(rewriter, loc, slowResults);

  for (auto [output, result] : llvm::zip(outputs, branch.getResults()))
    rewriter.replaceAllUsesWith(output, result);
  for (Operation *op : llvm::reverse(order)) {
    erased.insert(op);
    rewriter.eraseOp(op);
  }
}

// Fuses every chain, starting from the last operation of each block so that
// a chain is found from its root.
void fuseChains(Operation *module) {
  SmallVector<Operation *> candidates;
  module->walk([&](Operation *op) {
    if (getFusible(op))
      candidates.push_back(op);
  });
  FusionRewriter rewriter(module->getContext());
  DenseSet<Operation *> erased;
  for (Operation *op : llvm::reverse(candidates))
    if (!erased.contains(op))
      fuseChain(op, rewriter, erased);
}

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
    fuseChains(getOperation());
    RewritePatternSet patterns(&getContext());
    patterns
        .add<ConstantLowering, ExtSILowering, ExtUILowering, RetainLowering,
             ReleaseLowering, AddILowering, SubILowering, MonusILowering,
             MulILowering, GcdILowering, AndILowering, XOrILowering,
             OrILowering, NotILowering, NegILowering, AbsILowering,
             ShLILowering, ShRSILowering, FloorMod2ExpILowering, IPowILowering,
             DivSILowering, RemSILowering, FloorDivSILowering,
             FloorModSILowering, DivRemSILowering, FloorDivModSILowering,
             EDivSILowering, EModSILowering, EDivModSILowering, CmpILowering,
             TruncationLowering<TruncIOp>, TruncationLowering<TruncIOverflowOp>,
             Log2ILowering, TestBitILowering, LimbCountLowering, LimbLowering,
             SIToFPLowering, FPToSILowering>(&getContext());
    populateMatchLoweringPatterns(patterns);
    ConversionTarget target(getContext());
    target.addIllegalOp<ConstantOp, ExtSIOp, ExtUIOp, RetainOp, ReleaseOp,
                        AddIOp, SubIOp, MonusIOp, MulIOp, GcdIOp, AndIOp,
                        XOrIOp, OrIOp, NotIOp, NegIOp, AbsIOp, ShLIOp, ShRSIOp,
                        FloorMod2ExpIOp, IPowIOp, DivSIOp, RemSIOp,
                        FloorDivSIOp, FloorModSIOp, DivRemSIOp, FloorDivModSIOp,
                        EDivSIOp, EModSIOp, EDivModSIOp, CmpIOp, TruncIOp,
                        TruncIOverflowOp, Log2IOp, TestBitIOp, LimbCountOp,
                        LimbOp, SIToFPOp, FPToSIOp, MatchOp, YieldOp>();
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
