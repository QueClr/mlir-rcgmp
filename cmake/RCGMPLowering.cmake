# Compiles MLIR through the complete MLIR -> SCF -> LLVM IR path to native
# objects.
find_program(RCGMP_MLIR_TRANSLATE NAMES mlir-translate
             HINTS ${LLVM_TOOLS_BINARY_DIR} REQUIRED)
find_program(RCGMP_CLANG NAMES clang HINTS ${LLVM_TOOLS_BINARY_DIR} REQUIRED)

# Lowers the MLIR file source, relative to the current source directory, to
# an object file at the same relative path in the binary directory, and sets
# object to its path. Further arguments are rcgmp-opt passes to run first.
function(rcgmp_lower_mlir source object)
  string(REGEX REPLACE "\\.mlir$" "" base ${CMAKE_CURRENT_BINARY_DIR}/${source})
  add_custom_command(OUTPUT ${base}.o
    COMMAND $<TARGET_FILE:rcgmp-opt> ${CMAKE_CURRENT_SOURCE_DIR}/${source}
      ${ARGN} -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm
      -reconcile-unrealized-casts -o ${base}.mlir
    COMMAND ${RCGMP_MLIR_TRANSLATE} --mlir-to-llvmir ${base}.mlir -o ${base}.ll
    COMMAND ${RCGMP_CLANG} -O2 -Wno-unused-command-line-argument -Wno-override-module -c ${base}.ll -o ${base}.o
    DEPENDS rcgmp-opt ${source} VERBATIM)
  set(${object} ${base}.o PARENT_SCOPE)
endfunction()
