set(LLVM_TARGET_DEFINITIONS RCGMPDialect.td)
mlir_tablegen(RCGMPDialect.h.inc -gen-dialect-decls -dialect=rcgmp)
mlir_tablegen(RCGMPDialect.cpp.inc -gen-dialect-defs -dialect=rcgmp)

set(LLVM_TARGET_DEFINITIONS RCGMPTypes.td)
mlir_tablegen(RCGMPTypes.h.inc -gen-typedef-decls)
mlir_tablegen(RCGMPTypes.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS RCGMPOps.td)
mlir_tablegen(RCGMPOps.h.inc -gen-op-decls)
mlir_tablegen(RCGMPOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS RCGMPBridgeOps.td)
mlir_tablegen(RCGMPBridgeOps.h.inc -gen-op-decls)
mlir_tablegen(RCGMPBridgeOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(RCGMPIncGen)
