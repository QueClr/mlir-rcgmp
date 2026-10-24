add_mlir_library(MLIRRCGMPTransforms
  OptimizeRC.cpp

  LINK_LIBS PUBLIC
    MLIRRCGMP MLIRSCFDialect MLIRPass
)
