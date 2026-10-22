#include "FastPathLowering.h"
#include "RCGMP/IR/RCGMPDialect.h"

#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

// Unregistered operations exercising operand and result shapes of the
// fast-path lowering that no public operation uses yet.

namespace mlir::rcgmp::test {
namespace {

unsigned widthOf(Value bits) {
  return cast<IntegerType>(bits.getType()).getWidth();
}

bool hasTypes(Operation *op, ArrayRef<Type> operands, Type result) {
  return llvm::equal(op->getOperandTypes(), operands) &&
         op->getNumResults() == 1 && op->getResult(0).getType() == result;
}

// Unregistered operations have no verifier, so each pattern checks the shape
// it expects through Derived::hasExpectedShape(op) before lowering.
template <typename Derived> struct TestLowering : FastPathLowering<Derived> {
  explicit TestLowering(MLIRContext *context)
      : FastPathLowering<Derived>(context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!Derived::hasExpectedShape(op))
      return rewriter.notifyMatchFailure(op, "unexpected operation shape");
    return FastPathLowering<Derived>::matchAndRewrite(op, rewriter);
  }
};

// "test.negate"(%x) : (!rcgmp.mpz) -> !rcgmp.mpz borrows %x.
struct NegateLowering : TestLowering<NegateLowering> {
  using TestLowering::TestLowering;

  static constexpr bool consumesOperands = false;

  static StringRef getOperationName() { return "test.negate"; }

  static bool hasExpectedShape(Operation *op) {
    if (op->getNumOperands() != 1)
      return false;
    Type type = op->getOperand(0).getType();
    return isa<MpzType>(type) && hasTypes(op, type, type);
  }

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "test_negate_atomic" : "test_negate";
  }

  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *, ValueRange operands) {
    Value x = operands[0];
    auto two = arith::ConstantIntOp::create(rewriter, loc, 2, widthOf(x));
    // 2 - (4*n+1) = 4*(-n)+1 overflows only for the minimum immediate.
    auto checked = bridge::SubSIExtendedOp::create(
        rewriter, loc, x.getType(), rewriter.getI1Type(), two, x);
    auto yes = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
    auto fits =
        arith::XOrIOp::create(rewriter, loc, checked.getOverflow(), yes);
    auto valid = arith::AndIOp::create(rewriter, loc, fits,
                                       allImmediate(rewriter, loc, x));
    return {checked.getDiff(), valid};
  }
};

// "test.select"(%x, %c, %y) : (!rcgmp.mpz, i1, !rcgmp.mpz) -> !rcgmp.mpz
// consumes %x and %y and returns %x if %c is set, otherwise %y.
struct SelectLowering : TestLowering<SelectLowering> {
  using TestLowering::TestLowering;

  static StringRef getOperationName() { return "test.select"; }

  static bool hasExpectedShape(Operation *op) {
    if (op->getNumOperands() != 3)
      return false;
    Type type = op->getOperand(0).getType();
    Type condition = IntegerType::get(op->getContext(), 1);
    return isa<MpzType>(type) && hasTypes(op, {type, condition, type}, type);
  }

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "test_select_atomic" : "test_select";
  }

  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *, ValueRange operands) {
    Value x = operands[0], condition = operands[1], y = operands[2];
    auto bits = arith::SelectOp::create(rewriter, loc, condition, x, y);
    return {bits, allImmediate(rewriter, loc, {x, y})};
  }
};

// "test.equal_word"(%x, %w) : (!rcgmp.mpz, iW) -> i1 borrows %x, where W is
// the handle width. The runtime returns a nonzero i32 when they are equal.
struct EqualWordLowering : TestLowering<EqualWordLowering> {
  using TestLowering::TestLowering;

  static constexpr bool consumesOperands = false;

  static StringRef getOperationName() { return "test.equal_word"; }

  static bool hasExpectedShape(Operation *op) {
    if (op->getNumOperands() != 2)
      return false;
    auto type = dyn_cast<MpzType>(op->getOperand(0).getType());
    if (!type)
      return false;
    unsigned width =
        DataLayout::closest(op).getTypeSizeInBits(type).getFixedValue();
    auto word = IntegerType::get(op->getContext(), width);
    return hasTypes(op, {type, word}, IntegerType::get(op->getContext(), 1));
  }

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "test_equal_word_atomic" : "test_equal_word";
  }

  static Type getRuntimeResultType(Operation *op) {
    return IntegerType::get(op->getContext(), 32);
  }

  static Value buildSlowResult(PatternRewriter &rewriter, Location loc,
                               Operation *, Value result) {
    auto zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    return arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne,
                                 result, zero);
  }

  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *, ValueRange operands) {
    Value x = operands[0], word = operands[1];
    auto two = arith::ConstantIntOp::create(rewriter, loc, 2, widthOf(x));
    auto payload = arith::ShRSIOp::create(rewriter, loc, x, two);
    auto equal = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                       payload, word);
    return {equal, allImmediate(rewriter, loc, x)};
  }
};

// "test.truncate"(%x) : (!rcgmp.mpz) -> iN consumes %x and wraps it to N bits.
struct TruncateLowering : TestLowering<TruncateLowering> {
  using TestLowering::TestLowering;

  static StringRef getOperationName() { return "test.truncate"; }

  static bool hasExpectedShape(Operation *op) {
    return op->getNumOperands() == 1 &&
           isa<MpzType>(op->getOperand(0).getType()) &&
           op->getNumResults() == 1 &&
           op->getResult(0).getType().isSignlessInteger();
  }

  static StringRef getRuntimeName(bool atomic) {
    return atomic ? "test_truncate_atomic" : "test_truncate";
  }

  static FastPath buildFastPath(PatternRewriter &rewriter, Location loc,
                                Operation *op, ValueRange operands) {
    Value x = operands[0];
    auto two = arith::ConstantIntOp::create(rewriter, loc, 2, widthOf(x));
    Value payload = arith::ShRSIOp::create(rewriter, loc, x, two);
    auto type = cast<IntegerType>(op->getResult(0).getType());
    if (type.getWidth() < widthOf(x))
      payload = arith::TruncIOp::create(rewriter, loc, type, payload);
    else if (type.getWidth() > widthOf(x))
      payload = arith::ExtSIOp::create(rewriter, loc, type, payload);
    return {payload, allImmediate(rewriter, loc, x)};
  }
};

struct TestFastPathLoweringPass
    : PassWrapper<TestFastPathLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestFastPathLoweringPass)
  StringRef getArgument() const override {
    return "test-rcgmp-fast-path-lowering";
  }
  StringRef getDescription() const override {
    return "Lower test operations through the fast-path lowering template";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<RCGMPDialect, scf::SCFDialect, arith::ArithDialect,
                    func::FuncDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<NegateLowering, SelectLowering, EqualWordLowering,
                 TruncateLowering>(context);
    ConversionTarget target(*context);
    for (StringRef name : {NegateLowering::getOperationName(),
                           SelectLowering::getOperationName(),
                           EqualWordLowering::getOperationName(),
                           TruncateLowering::getOperationName()})
      target.addIllegalOp(OperationName(name, context));
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void registerTestPasses() { PassRegistration<TestFastPathLoweringPass>(); }

} // namespace mlir::rcgmp::test
