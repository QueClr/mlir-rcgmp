#include "RCGMP/IR/RCGMPTypes.h"

#include <mlir/IR/DialectImplementation.h>

namespace mlir::rcgmp {

Type MpzType::parse(AsmParser &parser) {
  if (failed(parser.parseOptionalLess()))
    return get(parser.getContext(), false);
  if (failed(parser.parseKeyword("atomic")) || failed(parser.parseGreater()))
    return {};
  return get(parser.getContext(), true);
}

void MpzType::print(AsmPrinter &printer) const {
  if (getAtomic())
    printer << "<atomic>";
}

} // namespace mlir::rcgmp
