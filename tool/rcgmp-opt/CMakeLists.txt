add_llvm_executable(rcgmp-opt main.cpp)
target_link_libraries(rcgmp-opt PRIVATE
  MLIRRCGMP
  MLIRRCGMPToLLVM
  MLIRRCGMPToSCF
  MLIRArithToLLVM
  MLIRSCFToControlFlow
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
mlir_check_all_link_libraries(rcgmp-opt)
install(TARGETS rcgmp-opt RUNTIME DESTINATION bin)
