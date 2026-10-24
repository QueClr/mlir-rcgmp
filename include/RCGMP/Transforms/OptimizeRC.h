#ifndef RCGMP_TRANSFORMS_OPTIMIZERC_H
#define RCGMP_TRANSFORMS_OPTIMIZERC_H

#include <mlir/Pass/Pass.h>

namespace mlir::rcgmp {
std::unique_ptr<Pass> createOptimizeRCPass();
} // namespace mlir::rcgmp

#endif
