#ifndef RCGMP_IR_DIALECT_TD
#define RCGMP_IR_DIALECT_TD

include "mlir/IR/OpBase.td"

def RCGMPDialect : Dialect {
  let name = "rcgmp";
  let cppNamespace = "::mlir::rcgmp";
  let summary = "Reference-counted arbitrary-precision arithmetic";
  let description = [{
    The rcgmp dialect represents reference-counted arbitrary-precision values
    backed by GMP. Integer types distinguish atomic and nonatomic reference
    counting. Structured lowering exposes control flow in SCF while bridge
    operations preserve representation details until LLVM lowering.
  }];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let hasConstantMaterializer = 1;
  let dependentDialects = ["::mlir::LLVM::LLVMDialect",
                           "::mlir::arith::ArithDialect"];
  let extraClassDeclaration = [{
    // Marks a function argument whose reference the caller keeps.
    static constexpr ::llvm::StringLiteral getBorrowedAttrName() {
      return ::llvm::StringLiteral("rcgmp.borrowed");
    }
  }];
}

#endif
