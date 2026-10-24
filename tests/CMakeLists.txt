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
  EXCLUDE_FROM_ALL SYSTEM)
# FuzzTest needs only the ANTLR runtime library.
set(ANTLR_BUILD_CPP_TESTS OFF)
# GoogleMock exports the include directories that GoogleTest inherits as system
# ones, which would hide warnings in the project's headers from the tests, so
# the dependencies inherit none.
get_directory_property(rcgmp_include_directories INCLUDE_DIRECTORIES)
set_directory_properties(PROPERTIES INCLUDE_DIRECTORIES "")
FetchContent_MakeAvailable(fuzztest)
set_directory_properties(PROPERTIES
  INCLUDE_DIRECTORIES "${rcgmp_include_directories}")
# Abseil's ABI depends on the language standard, so FuzzTest and the
# dependencies it builds, which it sets to C++17, must match the tests. Their
# own sources build without warnings.
function(rcgmp_configure_test_dependencies directory)
  get_property(targets DIRECTORY ${directory} PROPERTY BUILDSYSTEM_TARGETS)
  foreach(target IN LISTS targets)
    get_target_property(type ${target} TYPE)
    if(type MATCHES "^(STATIC_LIBRARY|OBJECT_LIBRARY|EXECUTABLE)$")
      set_target_properties(${target} PROPERTIES CXX_STANDARD 26)
      target_compile_options(${target} PRIVATE -w)
    endif()
  endforeach()
  get_property(subdirectories DIRECTORY ${directory} PROPERTY SUBDIRECTORIES)
  foreach(subdirectory IN LISTS subdirectories)
    rcgmp_configure_test_dependencies(${subdirectory})
  endforeach()
endfunction()
rcgmp_configure_test_dependencies(${fuzztest_SOURCE_DIR})
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

# Folds of operations on constants, compared with GMP.
add_executable(rcgmp-fold unit/FoldTest.cpp)
target_link_libraries(rcgmp-fold PRIVATE MLIRRCGMP RCGMPTesting)
set_target_properties(rcgmp-fold PROPERTIES CXX_STANDARD 26)
rcgmp_gtest(rcgmp-fold)

add_executable(rcgmp-gmp-link runtime/gmp-link.cpp)
target_link_libraries(rcgmp-gmp-link PRIVATE GMP::GMP)
rcgmp_gtest(rcgmp-gmp-link)

# Compile the complete MLIR -> SCF -> LLVM IR path and exercise its native ABI.
include(RCGMPLowering)
function(rcgmp_native_test name)
  rcgmp_lower_mlir(runtime/${name}.mlir object)
  add_executable(rcgmp-${name} runtime/${name}.cpp ${object})
  target_compile_definitions(rcgmp-${name} PRIVATE RCGMP_TEST_ALLOCATOR)
  target_link_libraries(rcgmp-${name} PRIVATE RCGMPTestAllocator RCGMPTesting)
  rcgmp_gtest(rcgmp-${name})
endfunction()

rcgmp_native_test(construction)
if(RCGMP_RUNTIME_DEFAULT_ALLOCATOR)
  add_executable(rcgmp-construction-default runtime/construction.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/runtime/construction.o)
  target_link_libraries(rcgmp-construction-default PRIVATE RCGMPTesting)
  rcgmp_gtest(rcgmp-construction-default)
endif()

rcgmp_native_test(addition)
rcgmp_native_test(binary)
rcgmp_native_test(unary)
rcgmp_native_test(signed-overflow)
rcgmp_native_test(constant)
rcgmp_native_test(compare)
rcgmp_native_test(shift)
rcgmp_native_test(floormod2exp)
rcgmp_native_test(power)
rcgmp_native_test(division)
rcgmp_native_test(log2)
rcgmp_native_test(gcd)
rcgmp_native_test(exact-division)

# The differential harness compares each feature, compiled from its own MLIR
# source, with GMP. It also exercises the comparisons in compare.mlir and the
# logarithms in log2.mlir.
add_executable(rcgmp-differential
  runtime/differential/binary.cpp runtime/differential/chain.cpp
  runtime/differential/compare.cpp runtime/differential/construction.cpp
  runtime/differential/counting.cpp runtime/differential/division.cpp
  runtime/differential/exact.cpp
  runtime/differential/floormod2exp.cpp
  runtime/differential/gcd.cpp runtime/differential/log2.cpp
  runtime/differential/match.cpp runtime/differential/power.cpp
  runtime/differential/shift.cpp runtime/differential/truncation.cpp
  runtime/differential/unary.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/runtime/compare.o
  ${CMAKE_CURRENT_BINARY_DIR}/runtime/log2.o)
target_sources(rcgmp-differential PRIVATE FILE_SET CXX_MODULES FILES
  runtime/differential/Differential.cppm)
foreach(name binary constant counting division exact extend floormod2exp match
    power shift truncation unary)
  rcgmp_lower_mlir(runtime/differential/${name}.mlir object)
  target_sources(rcgmp-differential PRIVATE ${object})
endforeach()
target_link_libraries(rcgmp-differential PRIVATE RCGMPTestAllocator
  RCGMPTesting)
rcgmp_gtest(rcgmp-differential)

find_package(Threads REQUIRED)
target_link_libraries(rcgmp-addition PRIVATE Threads::Threads)
