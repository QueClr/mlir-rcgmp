add_mlir_dialect_library(MLIRRCGMP
  RCGMPDialect.cpp
  RCGMPTypes.cpp
  RCGMPOps.cpp
  RCGMPBridgeOps.cpp

  DEPENDS RCGMPIncGen
  LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRIR
    MLIRSupport
    MLIRDataLayoutInterfaces
    MLIRLLVMDialect
    MLIRInferTypeOpInterface
    MLIRSideEffectInterfaces
)
