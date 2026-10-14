#ifndef RCGMP_CONVERSION_BRIDGE_TO_LLVM_H
#define RCGMP_CONVERSION_BRIDGE_TO_LLVM_H

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
namespace rcgmp {
void populateRCGMPBridgeToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                 RewritePatternSet &patterns);
} // namespace rcgmp
} // namespace mlir

#endif
