#include "RCGMP/Conversion/ConvertRCGMPBridgeToLLVM.h"
#include "RCGMP/IR/RCGMPBridgeOps.h"

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::rcgmp {
namespace {
using namespace bridge;

IntegerType wordType(Operation *op) {
  DataLayout layout = DataLayout::closest(op);
  auto pointer = LLVM::LLVMPointerType::get(op->getContext());
  return IntegerType::get(op->getContext(),
                          layout.getTypeSizeInBits(pointer).getFixedValue());
}

Value integerConstant(OpBuilder &builder, Location loc, IntegerType type,
                      llvm::APInt value) {
  return LLVM::ConstantOp::create(builder, loc, type,
                                  builder.getIntegerAttr(type, value));
}

Value integerConstant(OpBuilder &builder, Location loc, IntegerType type,
                      uint64_t value) {
  return integerConstant(builder, loc, type,
                         llvm::APInt(type.getWidth(), value));
}

Value resizeInteger(OpBuilder &builder, Location loc, Value value,
                    IntegerType type, bool isUnsigned) {
  auto source = cast<IntegerType>(value.getType());
  if (source == type)
    return value;
  if (source.getWidth() > type.getWidth())
    return LLVM::TruncOp::create(builder, loc, type, value);
  if (isUnsigned)
    return LLVM::ZExtOp::create(builder, loc, type, value);
  return LLVM::SExtOp::create(builder, loc, type, value);
}

LLVM::CallOp runtimeCall(ConversionPatternRewriter &rewriter, Operation *op,
                         StringRef name, Type result, ValueRange arguments) {
  auto module = op->getParentOfType<ModuleOp>();
  SmallVector<Type> types(arguments.getTypes());
  auto functionType = LLVM::LLVMFunctionType::get(result, types);
  auto function = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
  if (module.lookupSymbol(name)) {
    if (!function || function.getFunctionType() != functionType) {
      op->emitError() << "incompatible declaration of runtime symbol " << name;
      return {};
    }
  } else {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    function =
        LLVM::LLVMFuncOp::create(rewriter, op->getLoc(), name, functionType);
  }
  return LLVM::CallOp::create(rewriter, op->getLoc(), function, arguments);
}

template <typename Op, typename LLVMOp>
struct SignedOverflowLowering : OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto pair = LLVM::LLVMStructType::getLiteral(
        op.getContext(), {adaptor.getLhs().getType(), rewriter.getI1Type()});
    auto checked = LLVMOp::create(rewriter, op.getLoc(), pair, adaptor.getLhs(),
                                  adaptor.getRhs());
    auto sum = LLVM::ExtractValueOp::create(rewriter, op.getLoc(), checked,
                                            ArrayRef<int64_t>{0});
    auto overflow = LLVM::ExtractValueOp::create(rewriter, op.getLoc(), checked,
                                                 ArrayRef<int64_t>{1});
    rewriter.replaceOp(op, {sum.getResult(), overflow.getResult()});
    return success();
  }
};

struct IsImmediateLowering : OpConversionPattern<MpzIsImmediateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MpzIsImmediateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = wordType(op);
    auto loc = op.getLoc();
    auto bits =
        LLVM::PtrToIntOp::create(rewriter, loc, type, adaptor.getValue());
    auto one = integerConstant(rewriter, loc, type, 1);
    auto tag = LLVM::AndOp::create(rewriter, loc, bits, one);
    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(op, LLVM::ICmpPredicate::eq, tag,
                                              one);
    return success();
  }
};

struct FitsImmediateLowering : OpConversionPattern<MpzFitsImmediateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MpzFitsImmediateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = cast<IntegerType>(adaptor.getValue().getType());
    unsigned bits = wordType(op).getWidth() - 2;
    if (type.getWidth() <= bits - (op.getIsUnsigned() ? 1 : 0)) {
      rewriter.replaceOp(
          op, integerConstant(rewriter, op.getLoc(), rewriter.getI1Type(), 1));
      return success();
    }
    auto max = integerConstant(
        rewriter, op.getLoc(), type,
        llvm::APInt::getSignedMaxValue(bits).zextOrTrunc(type.getWidth()));
    Value fits =
        LLVM::ICmpOp::create(rewriter, op.getLoc(),
                             op.getIsUnsigned() ? LLVM::ICmpPredicate::ule
                                                : LLVM::ICmpPredicate::sle,
                             adaptor.getValue(), max);
    if (!op.getIsUnsigned()) {
      auto min = integerConstant(
          rewriter, op.getLoc(), type,
          llvm::APInt::getSignedMinValue(bits).sextOrTrunc(type.getWidth()));
      auto above =
          LLVM::ICmpOp::create(rewriter, op.getLoc(), LLVM::ICmpPredicate::sge,
                               adaptor.getValue(), min);
      fits = LLVM::AndOp::create(rewriter, op.getLoc(), fits, above);
    }
    rewriter.replaceOp(op, fits);
    return success();
  }
};

struct FromImmediateLowering : OpConversionPattern<MpzFromImmediateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MpzFromImmediateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = wordType(op);
    auto loc = op.getLoc();
    auto value = resizeInteger(rewriter, loc, adaptor.getValue(), type,
                               op.getIsUnsigned());
    auto two = integerConstant(rewriter, loc, type, 2);
    auto one = integerConstant(rewriter, loc, type, 1);
    auto shifted = LLVM::ShlOp::create(rewriter, loc, value, two);
    auto tagged = LLVM::OrOp::create(rewriter, loc, shifted, one);
    rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(
        op, LLVM::LLVMPointerType::get(op.getContext()), tagged.getResult(),
        LLVM::DereferenceableAttr{});
    return success();
  }
};

struct PromoteLowering : OpConversionPattern<MpzPromoteOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MpzPromoteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto word = wordType(op);
    auto input = cast<IntegerType>(adaptor.getValue().getType());
    auto pointer = LLVM::LLVMPointerType::get(op.getContext());
    size_t count = llvm::divideCeil(input.getWidth(), word.getWidth());
    auto function = op->getParentOfType<FunctionOpInterface>();
    if (!function)
      return rewriter.notifyMatchFailure(op,
                                         "promotion requires a function scope");
    Value storage;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&function.getFunctionBody().front());
      auto size = integerConstant(rewriter, loc, word, count);
      storage = LLVM::AllocaOp::create(rewriter, loc, pointer, word, size);
    }
    LLVM::LifetimeStartOp::create(rewriter, loc, storage);
    auto zero = integerConstant(rewriter, loc, input, 0);
    Value negative = integerConstant(rewriter, loc, rewriter.getI1Type(), 0);
    Value magnitude = adaptor.getValue();
    if (!op.getIsUnsigned()) {
      negative = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::slt,
                                      magnitude, zero);
      auto negated = LLVM::SubOp::create(rewriter, loc, zero, magnitude);
      magnitude =
          LLVM::SelectOp::create(rewriter, loc, negative, negated, magnitude);
    }
    for (size_t i = 0; i < count; ++i) {
      Value part = magnitude;
      if (i) {
        auto shift = integerConstant(rewriter, loc, input, i * word.getWidth());
        part = LLVM::LShrOp::create(rewriter, loc, magnitude, shift);
      }
      part = resizeInteger(rewriter, loc, part, word, true);
      auto address = LLVM::GEPOp::create(rewriter, loc, pointer, word, storage,
                                         ArrayRef<LLVM::GEPArg>{int32_t(i)});
      LLVM::StoreOp::create(rewriter, loc, part, address);
    }
    auto length = integerConstant(rewriter, loc, word, count);
    auto sign =
        LLVM::ZExtOp::create(rewriter, loc, rewriter.getI32Type(), negative);
    auto atomic = integerConstant(rewriter, loc, rewriter.getI32Type(),
                                  cast<MpzType>(op.getType()).getAtomic());
    auto call = runtimeCall(rewriter, op, "rcgmp_mpz_from_words", pointer,
                            {storage, length, sign, atomic});
    if (!call)
      return failure();
    LLVM::LifetimeEndOp::create(rewriter, loc, storage);
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

// Emit the object header and magnitude limbs as a read-only private global.
// Its name is derived from the value, so equal constants share one object.
struct ImmortalLowering : OpConversionPattern<MpzImmortalOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MpzImmortalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto word = wordType(op);
    unsigned bits = word.getWidth();
    APInt value = op.getValue();
    bool negative = value.isNegative();
    // Widen first so that the magnitude of the minimum value is representable.
    APInt magnitude = value.sext(value.getBitWidth() + 1).abs();
    unsigned count = llvm::divideCeil(magnitude.getActiveBits(), bits);
    magnitude = magnitude.zextOrTrunc(count * bits);
    int64_t size = negative ? -int64_t(count) : int64_t(count);
    // Reference count, signed limb count, capacity, and limbs.
    SmallVector<APInt> words = {APInt::getZero(bits),
                                APInt(bits, size, /*isSigned=*/true),
                                APInt(bits, count)};
    for (unsigned i = 0; i < count; ++i)
      words.push_back(magnitude.extractBits(bits, i * bits));
    auto type = LLVM::LLVMArrayType::get(word, words.size());
    auto data = DenseElementsAttr::get(
        RankedTensorType::get({int64_t(words.size())}, word), words);
    unsigned alignment = bits / 8;

    SmallString<64> name(negative ? "__rcgmp_mpz_neg_" : "__rcgmp_mpz_");
    magnitude.toString(name, 16, /*Signed=*/false, /*formatAsCLiteral=*/false,
                       /*UpperCase=*/false);
    auto module = op->getParentOfType<ModuleOp>();
    Operation *existing = module.lookupSymbol(name);
    auto global = dyn_cast_or_null<LLVM::GlobalOp>(existing);
    if (existing) {
      if (!global || !global.getConstant() ||
          global.getLinkage() != LLVM::Linkage::Private ||
          global.getAlignment().value_or(0) < alignment ||
          global.getGlobalType() != type || global.getValueAttr() != data)
        return op->emitError()
               << "incompatible definition of immortal integer " << name;
    } else {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      global = LLVM::GlobalOp::create(
          rewriter, op.getLoc(), type,
          /*isConstant=*/true, LLVM::Linkage::Private, name, data, alignment);
      global.setUnnamedAddr(LLVM::UnnamedAddr::Global);
    }
    rewriter.replaceOpWithNewOp<LLVM::AddressOfOp>(op, global);
    return success();
  }
};

template <typename Op, bool Increment>
struct RefCountLowering : OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto word = wordType(op);
    auto one = integerConstant(rewriter, loc, word, 1);
    Value old;
    if (cast<MpzType>(op.getValue().getType()).getAtomic()) {
      old = LLVM::AtomicRMWOp::create(
          rewriter, loc,
          Increment ? LLVM::AtomicBinOp::add : LLVM::AtomicBinOp::sub,
          adaptor.getValue(), one,
          Increment ? LLVM::AtomicOrdering::monotonic
                    : LLVM::AtomicOrdering::release);
    } else {
      old = LLVM::LoadOp::create(rewriter, loc, word, adaptor.getValue());
      Value next;
      if constexpr (Increment)
        next = LLVM::AddOp::create(rewriter, loc, old, one);
      else
        next = LLVM::SubOp::create(rewriter, loc, old, one);
      LLVM::StoreOp::create(rewriter, loc, next, adaptor.getValue());
    }
    Value limit =
        Increment ? integerConstant(rewriter, loc, word,
                                    llvm::APInt::getAllOnes(word.getWidth()))
                  : one;
    rewriter.template replaceOpWithNewOp<LLVM::ICmpOp>(
        op, LLVM::ICmpPredicate::eq, old, limit);
    return success();
  }
};

struct FreeLowering : OpConversionPattern<MpzFreeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MpzFreeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto atomic =
        integerConstant(rewriter, op.getLoc(), rewriter.getI32Type(),
                        cast<MpzType>(op.getValue().getType()).getAtomic());
    auto call = runtimeCall(rewriter, op, "rcgmp_mpz_free",
                            LLVM::LLVMVoidType::get(op.getContext()),
                            {adaptor.getValue(), atomic});
    if (!call)
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

struct ToBitsLowering : OpConversionPattern<MpzToBitsOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MpzToBitsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(op, op.getType(),
                                                  adaptor.getValue());
    return success();
  }
};

struct FromBitsLowering : OpConversionPattern<MpzFromBitsOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MpzFromBitsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(
        op, LLVM::LLVMPointerType::get(op.getContext()), adaptor.getValue(),
        LLVM::DereferenceableAttr{});
    return success();
  }
};

// Compare a heap reference count with Count, loading atomic counts with
// Ordering.
template <typename Op, uint64_t Count, LLVM::AtomicOrdering Ordering>
struct RefCountTestLowering : OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto word = wordType(op);
    auto count =
        LLVM::LoadOp::create(rewriter, op.getLoc(), word, adaptor.getValue());
    if (cast<MpzType>(op.getValue().getType()).getAtomic()) {
      count.setOrdering(Ordering);
      // The heap header has native-word alignment (enforced by the runtime).
      count.setAlignment(word.getWidth() / 8);
    }
    auto expected = integerConstant(rewriter, op.getLoc(), word, Count);
    rewriter.template replaceOpWithNewOp<LLVM::ICmpOp>(
        op, LLVM::ICmpPredicate::eq, count, expected);
    return success();
  }
};

using IsUniqueLowering =
    RefCountTestLowering<MpzIsUniqueOp, 1, LLVM::AtomicOrdering::acquire>;
using IsImmortalLowering =
    RefCountTestLowering<MpzIsImmortalOp, 0, LLVM::AtomicOrdering::monotonic>;

} // namespace

void populateRCGMPBridgeToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                 RewritePatternSet &patterns) {
  patterns.add<
      SignedOverflowLowering<AddSIExtendedOp, LLVM::SAddWithOverflowOp>,
      SignedOverflowLowering<SubSIExtendedOp, LLVM::SSubWithOverflowOp>,
      SignedOverflowLowering<MulSIOverflowOp, LLVM::SMulWithOverflowOp>,
      IsImmediateLowering, FitsImmediateLowering, FromImmediateLowering,
      PromoteLowering, ImmortalLowering, RefCountLowering<MpzIncRefOp, true>,
      RefCountLowering<MpzDecRefOp, false>, FreeLowering, ToBitsLowering,
      FromBitsLowering, IsUniqueLowering, IsImmortalLowering>(
      converter, patterns.getContext());
}

} // namespace mlir::rcgmp
