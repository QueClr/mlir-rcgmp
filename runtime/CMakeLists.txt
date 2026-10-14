# Dependency target for the C/C++ runtime. Implementation sources will be
# added with the first arithmetic operations.
add_library(RCGMPRuntime INTERFACE)
add_library(RCGMP::Runtime ALIAS RCGMPRuntime)
target_link_libraries(RCGMPRuntime INTERFACE GMP::GMP)
