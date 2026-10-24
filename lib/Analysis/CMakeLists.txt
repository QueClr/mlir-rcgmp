add_mlir_library(MLIRRCGMPAnalysis
  VerifyOwnership.cpp

  LINK_LIBS PUBLIC
    MLIRRCGMP
    MLIRCallInterfaces
    MLIRControlFlowInterfaces
    MLIRFunctionInterfaces
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRTransformUtils
)
