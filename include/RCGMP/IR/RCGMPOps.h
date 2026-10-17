#ifndef RCGMP_IR_OPS_H
#define RCGMP_IR_OPS_H

#include "RCGMP/IR/RCGMPTraits.h"
#include "RCGMP/IR/RCGMPTypes.h"
#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#define GET_OP_CLASSES
#include "RCGMP/IR/RCGMPOps.h.inc"

#endif
