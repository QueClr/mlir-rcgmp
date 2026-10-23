#ifndef RCGMP_CONVERSION_FAST_PATH_LOWERING_H
#define RCGMP_CONVERSION_FAST_PATH_LOWERING_H

#include "RCGMP/IR/RCGMPBridgeOps.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir::rcgmp {

// One value per result. For an mpz result, the value holds the tagged result
// bits, which must form an immediate whenever valid is true; otherwise it is
// the result itself. The values are used only when valid is true.
struct FastPath {
  SmallVector<Value, 1> values;
  Value valid;
};

// Lowers an operation with at least one mpz operand into an inline fast path
// and a runtime call taken when the fast path is not valid.
//
// Derived supplies:
// - Op, the operation class, or getOperationName() for an operation without
//   one;
// - getRuntimeName(atomic), naming separate symbols per reference-counting
//   mode so that atomic and nonatomic source signatures stay distinct;
// - buildFastPath(rewriter, loc, op, operands), where operands are the
//   operation's operands with each mpz operand replaced by its handle bits
//   and each other one converted by buildRuntimeOperand. The fast path must be
//   safe to evaluate on any live handle. For an mpz result, valid must imply
//   that the result bits form an immediate, since the fast path creates no
//   reference. When the operation consumes its mpz operands, valid must also
//   imply that every mpz operand is immediate, since the fast path drops them
//   without a release.
//
// The runtime receives the mpz operands as they are and the others as
// buildRuntimeOperand converts them. When the operation consumes its mpz
// operands and produces an mpz, an unsigned i32 reuse mask follows, whose bit
// i permits reusing the storage of the i-th mpz operand, so such an operation
// has at most 32 mpz operands. An operation with several results receives
// them from the runtime as they are. For one with a single result, Derived
// may shadow:
// - consumesOperands, false when every mpz operand is borrowed;
// - getRuntimeResultType(op) and buildSlowResult(rewriter, loc, op, result),
//   when the runtime result must be converted to the operation result;
// - buildSlowPath(rewriter, loc, op, callRuntime), when the result can often
//   be found without calling the runtime.
// For any operation, Derived may shadow:
// - buildRuntimeOperand(rewriter, loc, operand, word), to convert a non-mpz
//   operand before the fast path and the runtime receive it, such as to the
//   handle-width word type, so that the runtime signature does not depend on
//   the operand's type.
template <typename Derived> struct FastPathLowering : RewritePattern {
  explicit FastPathLowering(MLIRContext *context)
      : RewritePattern(Derived::getOperationName(), 1, context) {}

  static StringRef getOperationName() {
    return Derived::Op::getOperationName();
  }

  static constexpr bool consumesOperands = true;

  static Type getRuntimeResultType(Operation *op) {
    return op->getResult(0).getType();
  }

  static Value buildSlowResult(PatternRewriter &, Location, Operation *,
                               Value result) {
    return result;
  }

  // The value that both paths receive for a non-mpz operand.
  static Value buildRuntimeOperand(PatternRewriter &, Location, Value operand,
                                   IntegerType) {
    return operand;
  }

  // Builds the result when the fast path is not valid, calling the runtime
  // through callRuntime.
  static Value buildSlowPath(PatternRewriter &rewriter, Location loc,
                             Operation *op, function_ref<Value()> callRuntime) {
    return Derived::buildSlowResult(rewriter, loc, op, callRuntime());
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto handle = llvm::find_if(op->getOperandTypes(), llvm::IsaPred<MpzType>);
    if (!op->getNumResults() || handle == op->getOperandTypes().end())
      return rewriter.notifyMatchFailure(op,
                                         "expected results and an mpz operand");
    auto type = cast<MpzType>(*handle);
    auto loc = op->getLoc();
    bool single = op->getNumResults() == 1;
    bool mpzResult = isa<MpzType>(op->getResult(0).getType());
    bool passReuse = Derived::consumesOperands && mpzResult;
    if (passReuse && llvm::count_if(op->getOperandTypes(),
                                    llvm::IsaPred<MpzType>) > maxReuseFlags)
      return rewriter.notifyMatchFailure(op, "too many mpz operands to reuse");

    unsigned width =
        DataLayout::closest(op).getTypeSizeInBits(type).getFixedValue();
    auto word = rewriter.getIntegerType(width);
    SmallVector<Value> arguments;
    for (Value operand : op->getOperands())
      arguments.push_back(
          isa<MpzType>(operand.getType())
              ? operand
              : Derived::buildRuntimeOperand(rewriter, loc, operand, word));
    SmallVector<Type> argumentTypes(ValueRange(arguments).getTypes());
    if (passReuse)
      argumentTypes.push_back(rewriter.getI32Type());
    SmallVector<Type> resultTypes(op->getResultTypes());
    if (single)
      resultTypes.front() = Derived::getRuntimeResultType(op);
    auto signature = rewriter.getFunctionType(argumentTypes, resultTypes);
    StringRef name = Derived::getRuntimeName(type.getAtomic());
    auto module = op->getParentOfType<ModuleOp>();
    auto function = module.lookupSymbol<func::FuncOp>(name);
    if (module.lookupSymbol(name)) {
      if (!function || function.getFunctionType() != signature)
        return rewriter.notifyMatchFailure(op,
                                           "incompatible runtime declaration");
    } else {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      function = func::FuncOp::create(rewriter, loc, name, signature);
      function.setPrivate();
    }

    SmallVector<Value> handles, operands;
    for (Value operand : arguments) {
      if (!isa<MpzType>(operand.getType())) {
        operands.push_back(operand);
        continue;
      }
      handles.push_back(operand);
      operands.push_back(
          bridge::MpzToBitsOp::create(rewriter, loc, word, operand));
    }
    FastPath fast = Derived::buildFastPath(rewriter, loc, op, operands);
    auto branch =
        scf::IfOp::create(rewriter, loc, op->getResultTypes(), fast.valid,
                          /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(&branch.getThenRegion().front());
    SmallVector<Value> immediates;
    for (auto [value, resultType] :
         llvm::zip_equal(fast.values, op->getResultTypes()))
      immediates.push_back(
          isa<MpzType>(resultType)
              ? bridge::MpzFromBitsOp::create(rewriter, loc, resultType, value)
              : value);
    scf::YieldOp::create(rewriter, loc, immediates);

    rewriter.setInsertionPointToStart(&branch.getElseRegion().front());
    if (passReuse)
      arguments.push_back(reuseMask(rewriter, loc, handles));
    auto call = [&] {
      return func::CallOp::create(rewriter, loc, function, arguments);
    };
    if (single)
      scf::YieldOp::create(rewriter, loc,
                           Derived::buildSlowPath(rewriter, loc, op, [&] {
                             return call().getResult(0);
                           }));
    else
      scf::YieldOp::create(rewriter, loc, call().getResults());
    rewriter.replaceOp(op, branch.getResults());
    return success();
  }

protected:
  // Tests the immediate tag of every handle whose bits are in bits. One
  // handle is tested with rcgmp.bridge.mpz.is_immediate, as other operations
  // test it. Several share one test of their bits ANDed together: separate
  // tests, which LLVM merges with those of the slow path, lead it to lay out
  // slower slow paths, such as that of rcgmp.cmpi.
  static Value allImmediate(PatternRewriter &rewriter, Location loc,
                            ValueRange bits) {
    if (bits.size() == 1)
      return bridge::MpzIsImmediateOp::create(
          rewriter, loc,
          bits.front().getDefiningOp<bridge::MpzToBitsOp>().getValue());
    Value common = bits.front();
    for (Value value : bits.drop_front())
      common = arith::AndIOp::create(rewriter, loc, common, value);
    auto one = arith::ConstantIntOp::create(
        rewriter, loc, 1, cast<IntegerType>(common.getType()).getWidth());
    auto tag = arith::AndIOp::create(rewriter, loc, common, one);
    return arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq, tag,
                                 one);
  }

private:
  static constexpr unsigned maxReuseFlags = 32;

  static Value reuseMask(PatternRewriter &rewriter, Location loc,
                         ValueRange handles) {
    SmallVector<Value> reusable;
    for (Value handle : handles)
      reusable.push_back(canReuse(rewriter, loc, handle));
    Value mask;
    for (auto [index, reuse] : llvm::enumerate(reusable)) {
      Value flag =
          arith::ExtUIOp::create(rewriter, loc, rewriter.getI32Type(), reuse);
      if (!index) {
        mask = flag;
        continue;
      }
      auto amount = arith::ConstantIntOp::create(rewriter, loc, index, 32);
      // flag is zero or one and index is below maxReuseFlags, so the shift
      // never wraps unsigned; only a flag shifted into the sign bit wraps
      // signed.
      auto overflow = arith::IntegerOverflowFlags::nuw;
      if (index < maxReuseFlags - 1)
        overflow = overflow | arith::IntegerOverflowFlags::nsw;
      auto shifted =
          arith::ShLIOp::create(rewriter, loc, flag, amount, overflow);
      mask = arith::OrIOp::create(rewriter, loc, mask, shifted);
    }
    return mask;
  }

  // A heap check must dominate each uniqueness load.
  static Value canReuse(PatternRewriter &rewriter, Location loc, Value value) {
    auto immediate = bridge::MpzIsImmediateOp::create(rewriter, loc, value);
    auto branch = scf::IfOp::create(
        rewriter, loc, TypeRange{rewriter.getI1Type()}, immediate, true);
    rewriter.setInsertionPointToStart(&branch.getThenRegion().front());
    auto no = arith::ConstantIntOp::create(rewriter, loc, 0, 1);
    scf::YieldOp::create(rewriter, loc, no.getResult());
    rewriter.setInsertionPointToStart(&branch.getElseRegion().front());
    auto unique = bridge::MpzIsUniqueOp::create(rewriter, loc, value);
    scf::YieldOp::create(rewriter, loc, unique.getResult());
    rewriter.setInsertionPointAfter(branch);
    return branch.getResult(0);
  }
};

} // namespace mlir::rcgmp

#endif
