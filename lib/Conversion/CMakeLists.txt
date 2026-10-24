set(LLVM_OPTIONAL_SOURCES ConvertRCGMPToLLVM.cpp ConvertRCGMPBridgeToLLVM.cpp
  ConvertRCGMPToSCF.cpp MatchLowering.cpp)

add_mlir_conversion_library(MLIRRCGMPToLLVM
  ConvertRCGMPToLLVM.cpp
  ConvertRCGMPBridgeToLLVM.cpp

  LINK_LIBS PUBLIC
    MLIRRCGMP
    MLIRConvertToLLVMInterface
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
)

add_mlir_conversion_library(MLIRRCGMPToSCF
  ConvertRCGMPToSCF.cpp
  MatchLowering.cpp
  LINK_LIBS PUBLIC
    MLIRRCGMP MLIRArithDialect MLIRControlFlowDialect MLIRSCFDialect
    MLIRFuncDialect MLIRMathDialect MLIRMemRefDialect MLIRUBDialect MLIRPass
    MLIRTransformUtils
)
