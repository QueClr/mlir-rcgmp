#ifndef RCGMP_ANALYSIS_VERIFYOWNERSHIP_H
#define RCGMP_ANALYSIS_VERIFYOWNERSHIP_H

#include <mlir/Pass/Pass.h>

namespace mlir::rcgmp {

// Checks that every function in root consumes each owned integer reference
// exactly once on every path, reporting violations as errors.
LogicalResult verifyOwnership(Operation *root);

// A pass running verifyOwnership on a module, registered as
// -rcgmp-verify-ownership. It changes nothing.
std::unique_ptr<Pass> createVerifyOwnershipPass();

} // namespace mlir::rcgmp

#endif
