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

add_custom_target(check-rcgmp
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  DEPENDS rcgmp-opt rcgmp-gmp-link
  USES_TERMINAL)
