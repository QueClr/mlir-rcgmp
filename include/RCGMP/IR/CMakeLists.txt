set(LLVM_TARGET_DEFINITIONS RCGMPDialect.td)
mlir_tablegen(RCGMPDialect.h.inc -gen-dialect-decls -dialect=rcgmp)
mlir_tablegen(RCGMPDialect.cpp.inc -gen-dialect-defs -dialect=rcgmp)

set(LLVM_TARGET_DEFINITIONS RCGMPTypes.td)
mlir_tablegen(RCGMPTypes.h.inc -gen-typedef-decls)
mlir_tablegen(RCGMPTypes.cpp.inc -gen-typedef-defs)

add_public_tablegen_target(RCGMPIncGen)
