#include "RCGMP/IR/RCGMPOps.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>

namespace mlir::rcgmp {

static IntegerAttr getNarrowestAttr(Builder &builder, const APInt &value) {
  unsigned width = value.getSignificantBits();
  return builder.getIntegerAttr(builder.getIntegerType(width),
                                value.sextOrTrunc(width));
}

static ParseResult parseSignedInteger(OpAsmParser &parser, IntegerAttr &attr) {
  auto loc = parser.getCurrentLocation();
  // The generic integer parser also accepts the boolean keywords.
  StringRef keyword;
  if (succeeded(parser.parseOptionalKeyword(&keyword, {"true", "false"})))
    return parser.emitError(loc, "expected integer value");
  APInt value;
  OptionalParseResult result = parser.parseOptionalInteger(value);
  if (!result.has_value())
    return parser.emitError(loc, "expected integer value");
  if (failed(*result))
    return failure();
  if (value.getSignificantBits() > IntegerType::kMaxWidth)
    return parser.emitError(loc, "integer value is too wide");
  attr = getNarrowestAttr(parser.getBuilder(), value);
  return success();
}

static void printSignedInteger(OpAsmPrinter &printer, Operation *,
                               IntegerAttr attr) {
  attr.getValue().print(printer.getStream(), /*isSigned=*/true);
}

} // namespace mlir::rcgmp

#include "RCGMP/IR/RCGMPEnums.cpp.inc"

#define GET_OP_CLASSES
#include "RCGMP/IR/RCGMPOps.cpp.inc"

namespace mlir::rcgmp {

void ConstantOp::build(OpBuilder &builder, OperationState &state, Type type,
                       const APInt &value) {
  build(builder, state, type, getNarrowestAttr(builder, value));
}

LogicalResult ConstantOp::verify() {
  // Every value, zero included, needs at least one bit.
  auto type = dyn_cast<IntegerType>(getValueAttr().getType());
  if (!type || !type.isSignless() || type.getWidth() == 0 ||
      type.getWidth() != getValue().getSignificantBits())
    return emitOpError("requires a signless integer attribute of the narrowest "
                       "width holding its value");
  return success();
}

} // namespace mlir::rcgmp
