cmake_minimum_required(VERSION 3.28)
project(RCGMP LANGUAGES C CXX VERSION 0.1.0)

option(RCGMP_ENABLE_TESTS "Build rcgmp tests" ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(cmake/FindLLVM.cmake)
include(cmake/FindMLIR.cmake)
find_package(GMP 6.2 REQUIRED)

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")
add_definitions(${LLVM_DEFINITIONS_LIST})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(runtime)
if(RCGMP_ENABLE_TESTS)
  # rcgmp-test-opt links the test passes, so define them first.
  add_subdirectory(tests/lib)
endif()
add_subdirectory(tool)

if(RCGMP_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

install(DIRECTORY include/RCGMP DESTINATION include
        FILES_MATCHING PATTERN "*.h" PATTERN "*.td")
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/RCGMP DESTINATION include
        FILES_MATCHING PATTERN "*.inc" PATTERN "CMakeFiles" EXCLUDE)
