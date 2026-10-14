#include "RCGMP/Conversion/ConvertRCGMPToLLVM.h"
#include "RCGMP/Conversion/ConvertRCGMPBridgeToLLVM.h"
#include "RCGMP/IR/RCGMPDialect.h"
#include "RCGMP/IR/RCGMPTypes.h"

#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::rcgmp {

void populateRCGMPToLLVMTypeConversions(LLVMTypeConverter &converter) {
  converter.addConversion([](MpzType type) -> Type {
    return LLVM::LLVMPointerType::get(type.getContext());
  });
}

namespace {

struct RCGMPConvertToLLVMInterface : ConvertToLLVMPatternInterface {
  explicit RCGMPConvertToLLVMInterface(Dialect *dialect)
      : ConvertToLLVMPatternInterface(dialect) {}

  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &converter,
      RewritePatternSet &patterns) const override {
    populateRCGMPToLLVMTypeConversions(converter);
    populateRCGMPBridgeToLLVMConversionPatterns(converter, patterns);
    target.addIllegalDialect<RCGMPDialect>();
  }
};

} // namespace

void registerConvertRCGMPToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, RCGMPDialect *dialect) {
    dialect->addInterfaces<RCGMPConvertToLLVMInterface>();
  });
}

} // namespace mlir::rcgmp
