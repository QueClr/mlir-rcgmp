#include "RCGMP/IR/RCGMPTypes.h"

#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/DialectImplementation.h>

namespace mlir::rcgmp {

Type MpzType::parse(AsmParser &parser) {
  if (failed(parser.parseOptionalLess()))
    return get(parser.getContext(), false);
  if (failed(parser.parseKeyword("atomic")) || failed(parser.parseGreater()))
    return {};
  return get(parser.getContext(), true);
}

void MpzType::print(AsmPrinter &printer) const {
  if (getAtomic())
    printer << "<atomic>";
}

llvm::TypeSize MpzType::getTypeSizeInBits(const DataLayout &dataLayout,
                                          DataLayoutEntryListRef params) const {
  return dataLayout.getTypeSizeInBits(LLVM::LLVMPointerType::get(getContext()));
}

uint64_t MpzType::getABIAlignment(const DataLayout &dataLayout,
                                  DataLayoutEntryListRef params) const {
  return dataLayout.getTypeABIAlignment(
      LLVM::LLVMPointerType::get(getContext()));
}

uint64_t MpzType::getPreferredAlignment(const DataLayout &dataLayout,
                                        DataLayoutEntryListRef params) const {
  return dataLayout.getTypePreferredAlignment(
      LLVM::LLVMPointerType::get(getContext()));
}

} // namespace mlir::rcgmp
