#ifndef RCGMP_IR_OPS_H
#define RCGMP_IR_OPS_H

#include "RCGMP/IR/RCGMPTraits.h"
#include "RCGMP/IR/RCGMPTypes.h"
#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <optional>

#include "RCGMP/IR/RCGMPEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "RCGMP/IR/RCGMPAttrs.h.inc"

namespace mlir::rcgmp {

// The integers from start up to the start of the next interval, all selected
// by arm, or by no arm. The first interval of a sequence is unbounded below
// and its start is the minimum of its width; the last is unbounded above.
// Every start in a sequence has one width, which leaves room to add or
// subtract one from any pattern value.
struct MatchInterval {
  llvm::APInt start;
  std::optional<unsigned> arm;
};

// The interval of a sequence that contains value.
const MatchInterval &findInterval(llvm::ArrayRef<MatchInterval> intervals,
                                  const llvm::APInt &value);

} // namespace mlir::rcgmp

#define GET_OP_CLASSES
#include "RCGMP/IR/RCGMPOps.h.inc"

#endif
