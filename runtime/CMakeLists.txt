option(RCGMP_RUNTIME_DEFAULT_ALLOCATOR "Provide default allocation hooks" ON)
add_library(RCGMPRuntime STATIC)
target_sources(RCGMPRuntime PUBLIC FILE_SET CXX_MODULES FILES
  Runtime.cppm Mpz.cppm Binary.cppm Add.cppm Multiply.cppm Bitwise.cppm
  Unary.cppm Shift.cppm Divide.cppm Compare.cppm)
if(RCGMP_RUNTIME_DEFAULT_ALLOCATOR)
  # Nothing imports this module, so a program's own hooks take precedence over
  # this archive member.
  target_sources(RCGMPRuntime PRIVATE FILE_SET allocation TYPE CXX_MODULES
    FILES Allocation.cppm)
endif()
add_library(RCGMP::Runtime ALIAS RCGMPRuntime)
target_compile_features(RCGMPRuntime PUBLIC cxx_std_26)
target_link_libraries(RCGMPRuntime PUBLIC GMP::GMP)
set_target_properties(RCGMPRuntime PROPERTIES
  POSITION_INDEPENDENT_CODE ON CXX_SCAN_FOR_MODULES ON)
install(TARGETS RCGMPRuntime ARCHIVE DESTINATION lib
  FILE_SET CXX_MODULES DESTINATION include/RCGMP/Runtime)
