#ifndef RCGMP_IR_ENUMS_TD
#define RCGMP_IR_ENUMS_TD

include "mlir/IR/EnumAttr.td"

// The signed predicates of arith.cmpi, with the same values.
def RCGMP_CmpIPredicateAttr : I64EnumAttr<"CmpIPredicate", "", [
    I64EnumAttrCase<"eq", 0>,
    I64EnumAttrCase<"ne", 1>,
    I64EnumAttrCase<"slt", 2>,
    I64EnumAttrCase<"sle", 3>,
    I64EnumAttrCase<"sgt", 4>,
    I64EnumAttrCase<"sge", 5>]> {
  let cppNamespace = "::mlir::rcgmp";
}

#endif
