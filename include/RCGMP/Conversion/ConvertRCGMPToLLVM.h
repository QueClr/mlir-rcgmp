#ifndef RCGMP_CONVERSION_CONVERTRCGMPTOLLVM_H
#define RCGMP_CONVERSION_CONVERTRCGMPTOLLVM_H

namespace mlir {
class DialectRegistry;
class LLVMTypeConverter;

namespace rcgmp {

/// Lower both reference-counting modes to an opaque pointer-sized handle.
void populateRCGMPToLLVMTypeConversions(LLVMTypeConverter &converter);

/// Enable rcgmp type conversion in the generic convert-to-llvm pass.
void registerConvertRCGMPToLLVMInterface(DialectRegistry &registry);

} // namespace rcgmp
} // namespace mlir

#endif
