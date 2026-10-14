#ifndef RCGMP_CONVERSION_CONVERTRCGMPTOSCF_H
#define RCGMP_CONVERSION_CONVERTRCGMPTOSCF_H

#include <mlir/Pass/Pass.h>

namespace mlir::rcgmp {
std::unique_ptr<Pass> createConvertRCGMPToSCFPass();
} // namespace mlir::rcgmp

#endif
