find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_program(RCGMP_FILECHECK NAMES FileCheck
             HINTS ${LLVM_TOOLS_BINARY_DIR} REQUIRED)

configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)
add_test(NAME rcgmp-ir
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_lit.py
          -v ${CMAKE_CURRENT_BINARY_DIR})

add_executable(rcgmp-gmp-link runtime/gmp-link.c)
target_link_libraries(rcgmp-gmp-link PRIVATE RCGMP::Runtime)
add_test(NAME rcgmp-gmp-link COMMAND rcgmp-gmp-link)

add_executable(rcgmp-type-lowering unit/TypeLoweringTest.cpp)
target_link_libraries(rcgmp-type-lowering PRIVATE
  MLIRRCGMPToLLVM MLIRDLTIDialect MLIRParser)
add_test(NAME rcgmp-type-lowering COMMAND rcgmp-type-lowering)

# Compile the complete MLIR -> SCF -> LLVM IR path and exercise its native ABI.
find_program(RCGMP_MLIR_TRANSLATE NAMES mlir-translate
             HINTS ${LLVM_TOOLS_BINARY_DIR} REQUIRED)
find_program(RCGMP_CLANG NAMES clang HINTS ${LLVM_TOOLS_BINARY_DIR} REQUIRED)
function(rcgmp_native_test name)
  set(base ${CMAKE_CURRENT_BINARY_DIR}/${name})
  add_custom_command(OUTPUT ${base}.o
    COMMAND $<TARGET_FILE:rcgmp-opt> ${CMAKE_CURRENT_SOURCE_DIR}/runtime/${name}.mlir
      -convert-rcgmp-to-scf -convert-scf-to-cf -convert-to-llvm
      -reconcile-unrealized-casts -o ${base}.mlir
    COMMAND ${RCGMP_MLIR_TRANSLATE} --mlir-to-llvmir ${base}.mlir -o ${base}.ll
    COMMAND ${RCGMP_CLANG} -O2 -Wno-unused-command-line-argument -Wno-override-module -c ${base}.ll -o ${base}.o
    DEPENDS rcgmp-opt runtime/${name}.mlir VERBATIM)
  add_executable(rcgmp-${name} runtime/${name}.c ${base}.o)
  target_include_directories(rcgmp-${name} PRIVATE ${PROJECT_SOURCE_DIR}/runtime)
  target_compile_definitions(rcgmp-${name} PRIVATE RCGMP_TEST_ALLOCATOR)
  target_link_libraries(rcgmp-${name} PRIVATE RCGMP::Runtime)
  add_test(NAME rcgmp-${name} COMMAND rcgmp-${name})
  add_dependencies(check-rcgmp rcgmp-${name})
endfunction()

add_custom_target(check-rcgmp
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  DEPENDS rcgmp-opt rcgmp-gmp-link rcgmp-type-lowering
  USES_TERMINAL)

rcgmp_native_test(construction)
if(RCGMP_RUNTIME_DEFAULT_ALLOCATOR)
  add_executable(rcgmp-construction-default runtime/construction.c
    ${CMAKE_CURRENT_BINARY_DIR}/construction.o)
  target_include_directories(rcgmp-construction-default PRIVATE ${PROJECT_SOURCE_DIR}/runtime)
  target_link_libraries(rcgmp-construction-default PRIVATE RCGMP::Runtime)
  add_test(NAME rcgmp-construction-default COMMAND rcgmp-construction-default)
  add_dependencies(check-rcgmp rcgmp-construction-default)
endif()

rcgmp_native_test(addition)
rcgmp_native_test(binary)
rcgmp_native_test(signed-overflow)
rcgmp_native_test(differential)
add_test(NAME rcgmp-differential-seed
  COMMAND rcgmp-differential 3141592653 64)
# An invalid argument must print only the usage line and exit with status 2.
add_test(NAME rcgmp-differential-usage
  COMMAND sh -c [=[output=$("$0" abc 2>&1); status=$?; printf '%s\n' "$output"
    test $status -eq 2 &&
    test "$output" = "usage: rcgmp-differential [seed [iterations]]"]=]
    $<TARGET_FILE:rcgmp-differential>)
find_package(Threads REQUIRED)
target_link_libraries(rcgmp-addition PRIVATE Threads::Threads)
