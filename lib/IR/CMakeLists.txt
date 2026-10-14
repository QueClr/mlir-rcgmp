add_mlir_dialect_library(MLIRRCGMP
  RCGMPDialect.cpp
  RCGMPTypes.cpp

  DEPENDS RCGMPIncGen
  LINK_LIBS PUBLIC MLIRIR MLIRSupport
)
