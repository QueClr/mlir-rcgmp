#ifndef RCGMP_CONVERSION_MATCH_LOWERING_H
#define RCGMP_CONVERSION_MATCH_LOWERING_H

#include <mlir/IR/PatternMatch.h>

namespace mlir::rcgmp {

// Lowers rcgmp.match to SCF and bridge operations, and to the rcgmp
// operations it uses on heap integers.
void populateMatchLoweringPatterns(RewritePatternSet &patterns);

} // namespace mlir::rcgmp

#endif
