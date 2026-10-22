set(RCGMP_OPT_LIBS
  MLIRRCGMP
  MLIRRCGMPToLLVM
  MLIRRCGMPToSCF
  MLIRArithToLLVM
  MLIRSCFToControlFlow
  MLIRUBToLLVM
  MLIRMemRefToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRTransforms
  MLIRControlFlowDialect
  MLIRControlFlowToLLVM
  MLIRConvertToLLVMPass
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIROptLib
)

add_llvm_executable(rcgmp-opt main.cpp)
target_link_libraries(rcgmp-opt PRIVATE ${RCGMP_OPT_LIBS})
mlir_check_all_link_libraries(rcgmp-opt)
install(TARGETS rcgmp-opt RUNTIME DESTINATION bin)

# rcgmp-opt with the test passes registered, built for the test suite only.
if(RCGMP_ENABLE_TESTS)
  add_llvm_executable(rcgmp-test-opt main.cpp)
  target_link_libraries(rcgmp-test-opt PRIVATE ${RCGMP_OPT_LIBS} RCGMPTestPasses)
  target_compile_definitions(rcgmp-test-opt PRIVATE RCGMP_INCLUDE_TESTS)
  mlir_check_all_link_libraries(rcgmp-test-opt)
endif()
