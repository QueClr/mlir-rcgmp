# Google Benchmark, pinned like the test dependencies.
# FETCHCONTENT_SOURCE_DIR_BENCHMARK may name local sources.
include(FetchContent)
FetchContent_Declare(benchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.10.0.tar.gz
  URL_HASH SHA256=acc5c7c0fbed855c4aaa2d7c3c36fb82b0c15c28f4e151e3f679c2fc0aa404af
  EXCLUDE_FROM_ALL)
set(BENCHMARK_ENABLE_TESTING OFF)
set(BENCHMARK_ENABLE_INSTALL OFF)
FetchContent_MakeAvailable(benchmark)

include(RCGMPLowering)
rcgmp_lower_mlir(operations.mlir operations)
add_executable(rcgmp-benchmarks operations.cpp ${operations})
target_link_libraries(rcgmp-benchmarks PRIVATE RCGMP::Runtime
  benchmark::benchmark_main)
