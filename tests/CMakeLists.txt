find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_program(RCGMP_FILECHECK NAMES FileCheck
             HINTS ${LLVM_TOOLS_BINARY_DIR} REQUIRED)

configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)
add_test(NAME rcgmp-ir
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_lit.py
          -v ${CMAKE_CURRENT_BINARY_DIR})

add_executable(rcgmp-type-lowering unit/TypeLoweringTest.cpp)
target_link_libraries(rcgmp-type-lowering PRIVATE
  MLIRRCGMPToLLVM MLIRDLTIDialect MLIRParser)
add_test(NAME rcgmp-type-lowering COMMAND rcgmp-type-lowering)

# FuzzTest builds its own dependencies. FETCHCONTENT_SOURCE_DIR_<NAME> may name
# local sources for FuzzTest and each dependency, as the Nix flake does.
include(FetchContent)
FetchContent_Declare(fuzztest
  URL https://github.com/google/fuzztest/archive/39e5d587af63ba6209e0958a3141460ce003409c.tar.gz
  URL_HASH SHA256=32dd48a81d5d318110f9192ec52cdf191ebd403764aec64f62ba04b980297421
  EXCLUDE_FROM_ALL)
# FuzzTest needs only the ANTLR runtime library.
set(ANTLR_BUILD_CPP_TESTS OFF)
FetchContent_MakeAvailable(fuzztest)
# Abseil's ABI depends on the language standard, so FuzzTest and the
# dependencies it builds, which it sets to C++17, must match the tests.
function(rcgmp_use_test_standard directory)
  get_property(targets DIRECTORY ${directory} PROPERTY BUILDSYSTEM_TARGETS)
  foreach(target IN LISTS targets)
    get_target_property(type ${target} TYPE)
    if(type MATCHES "^(STATIC_LIBRARY|OBJECT_LIBRARY|EXECUTABLE)$")
      set_target_properties(${target} PROPERTIES CXX_STANDARD 26)
    endif()
  endforeach()
  get_property(subdirectories DIRECTORY ${directory} PROPERTY SUBDIRECTORIES)
  foreach(subdirectory IN LISTS subdirectories)
    rcgmp_use_test_standard(${subdirectory})
  endforeach()
endfunction()
rcgmp_use_test_standard(${fuzztest_SOURCE_DIR})
include(GoogleTest)

add_library(RCGMPTesting STATIC)
target_sources(RCGMPTesting PUBLIC FILE_SET CXX_MODULES FILES
  runtime/Testing.cppm)
target_link_libraries(RCGMPTesting PUBLIC RCGMP::Runtime fuzztest::fuzztest
  GTest::gmock)
# Linked as objects, so these hooks replace any default ones in the runtime.
add_library(RCGMPTestAllocator OBJECT)
target_sources(RCGMPTestAllocator PUBLIC FILE_SET CXX_MODULES FILES
  runtime/CountingAllocator.cppm)
target_link_libraries(RCGMPTestAllocator PUBLIC GTest::gtest)
target_compile_features(RCGMPTestAllocator PUBLIC cxx_std_26)
set_target_properties(RCGMPTesting RCGMPTestAllocator PROPERTIES
  CXX_SCAN_FOR_MODULES ON)

add_custom_target(check-rcgmp
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  DEPENDS rcgmp-opt rcgmp-test-opt rcgmp-type-lowering
  USES_TERMINAL)

function(rcgmp_gtest target)
  set_target_properties(${target} PROPERTIES CXX_SCAN_FOR_MODULES ON)
  link_fuzztest(${target})
  gtest_discover_tests(${target} TEST_PREFIX "${target}/"
    DISCOVERY_MODE PRE_TEST)
  add_dependencies(check-rcgmp ${target})
endfunction()

add_executable(rcgmp-gmp-link runtime/gmp-link.cpp)
target_link_libraries(rcgmp-gmp-link PRIVATE GMP::GMP)
rcgmp_gtest(rcgmp-gmp-link)

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
  add_executable(rcgmp-${name} runtime/${name}.cpp ${base}.o)
  target_compile_definitions(rcgmp-${name} PRIVATE RCGMP_TEST_ALLOCATOR)
  target_link_libraries(rcgmp-${name} PRIVATE RCGMPTestAllocator RCGMPTesting)
  rcgmp_gtest(rcgmp-${name})
endfunction()

rcgmp_native_test(construction)
if(RCGMP_RUNTIME_DEFAULT_ALLOCATOR)
  add_executable(rcgmp-construction-default runtime/construction.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/construction.o)
  target_link_libraries(rcgmp-construction-default PRIVATE RCGMPTesting)
  rcgmp_gtest(rcgmp-construction-default)
endif()

rcgmp_native_test(addition)
rcgmp_native_test(binary)
rcgmp_native_test(signed-overflow)
rcgmp_native_test(constant)
rcgmp_native_test(compare)
rcgmp_native_test(differential)
# The differential harness also exercises the comparisons in compare.mlir.
target_sources(rcgmp-differential PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}/compare.o)
find_package(Threads REQUIRED)
target_link_libraries(rcgmp-addition PRIVATE Threads::Threads)
