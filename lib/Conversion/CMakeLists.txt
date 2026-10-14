add_mlir_conversion_library(MLIRRCGMPToLLVM
  ConvertRCGMPToLLVM.cpp

  LINK_LIBS PUBLIC
    MLIRRCGMP
    MLIRConvertToLLVMInterface
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
)
