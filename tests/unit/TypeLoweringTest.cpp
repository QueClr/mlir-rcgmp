#include "RCGMP/Conversion/ConvertRCGMPToLLVM.h"
#include "RCGMP/IR/RCGMPDialect.h"
#include "RCGMP/IR/RCGMPTypes.h"

#include <llvm/Support/raw_ostream.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Parser/Parser.h>

bool checkLayout(mlir::MLIRContext &context, llvm::StringRef source,
                 unsigned size, unsigned abiAlignment,
                 unsigned preferredAlignment) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module)
    return false;

  mlir::DataLayout layout(*module);
  mlir::LowerToLLVMOptions options(&context, layout);
  // Index width is independent of the runtime handle representation.
  options.overrideIndexBitwidth(16);
  mlir::LLVMTypeConverter converter(&context, options);
  mlir::rcgmp::populateRCGMPToLLVMTypeConversions(converter);
  auto pointer = mlir::LLVM::LLVMPointerType::get(&context);

  for (bool atomic : {false, true}) {
    auto type = mlir::rcgmp::MpzType::get(&context, atomic);
    if (converter.convertType(type) != pointer ||
        layout.getTypeSize(type) != llvm::TypeSize::getFixed(size) ||
        layout.getTypeABIAlignment(type) != abiAlignment ||
        layout.getTypePreferredAlignment(type) != preferredAlignment) {
      llvm::errs() << "incorrect lowering or layout for " << type << "\n";
      return false;
    }
  }
  return converter.convertType(mlir::IndexType::get(&context)) ==
         mlir::IntegerType::get(&context, 16);
}

int main() {
  mlir::MLIRContext context;
  context.loadDialect<mlir::rcgmp::RCGMPDialect, mlir::DLTIDialect>();

  bool passed = checkLayout(context, "module {}", 8, 8, 8);
  passed &= checkLayout(context, R"mlir(
    module attributes {
      dlti.dl_spec = #dlti.dl_spec<
        #dlti.dl_entry<!llvm.ptr, dense<[32, 32, 64, 32]> : vector<4xi64>>
      >
    } {}
  )mlir",
                        4, 4, 8);
  passed &= checkLayout(context, R"mlir(
    module attributes {
      dlti.dl_spec = #dlti.dl_spec<
        #dlti.dl_entry<!llvm.ptr, dense<[64, 64, 128, 32]> : vector<4xi64>>
      >
    } {}
  )mlir",
                        8, 8, 16);
  return passed ? 0 : 1;
}
