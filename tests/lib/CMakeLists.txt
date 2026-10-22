# Test passes registered in rcgmp-test-opt.
add_mlir_library(RCGMPTestPasses
  TestFastPathLowering.cpp

  EXCLUDE_FROM_LIBMLIR
  DISABLE_INSTALL
  LINK_LIBS PUBLIC
    MLIRRCGMP MLIRArithDialect MLIRSCFDialect MLIRFuncDialect MLIRPass
    MLIRTransformUtils
)
target_include_directories(RCGMPTestPasses PRIVATE
  ${PROJECT_SOURCE_DIR}/lib/Conversion)
