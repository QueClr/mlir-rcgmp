find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)
