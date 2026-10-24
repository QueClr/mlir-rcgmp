# Compiles MLIR through the complete MLIR -> SCF -> LLVM IR path to native
# objects.
find_program(RCGMP_MLIR_TRANSLATE NAMES mlir-translate
             HINTS ${LLVM_TOOLS_BINARY_DIR} REQUIRED)
find_program(RCGMP_CLANG NAMES clang HINTS ${LLVM_TOOLS_BINARY_DIR} REQUIRED)

# Simplifying the SCF form exercises the bridge folds in native code.
option(RCGMP_LOWERING_CANONICALIZE
  "Run CSE and canonicalization on lowered test and benchmark code" OFF)
set(RCGMP_LOWERING_SIMPLIFY)
if(RCGMP_LOWERING_CANONICALIZE)
  set(RCGMP_LOWERING_SIMPLIFY -cse -canonicalize)
endif()

# Lowers the MLIR file source, relative to the current source directory, to
# an object file at the same relative path in the binary directory, and sets
# object to its path. The source must pass the ownership verifier. Further
# arguments are rcgmp-opt passes to run before lowering.
function(rcgmp_lower_mlir source object)
  string(REGEX REPLACE "\\.mlir$" "" base ${CMAKE_CURRENT_BINARY_DIR}/${source})
  add_custom_command(OUTPUT ${base}.o
    COMMAND $<TARGET_FILE:rcgmp-opt> ${CMAKE_CURRENT_SOURCE_DIR}/${source}
      -rcgmp-verify-ownership ${ARGN} -convert-rcgmp-to-scf ${RCGMP_LOWERING_SIMPLIFY}
      -convert-scf-to-cf -convert-to-llvm
      -reconcile-unrealized-casts -o ${base}.mlir
    COMMAND ${RCGMP_MLIR_TRANSLATE} --mlir-to-llvmir ${base}.mlir -o ${base}.ll
    COMMAND ${RCGMP_CLANG} -O2 -Wno-unused-command-line-argument -Wno-override-module -c ${base}.ll -o ${base}.o
    DEPENDS rcgmp-opt ${source} VERBATIM)
  set(${object} ${base}.o PARENT_SCOPE)
endfunction()
