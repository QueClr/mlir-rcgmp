#ifndef RCGMP_IR_TRAITS_H
#define RCGMP_IR_TRAITS_H

#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace mlir::rcgmp {

template <typename ConcreteType>
struct NotSpeculatableImplTrait
    : OpTrait::TraitBase<ConcreteType, NotSpeculatableImplTrait> {
  Speculation::Speculatability getSpeculatability() {
    return Speculation::NotSpeculatable;
  }
};

} // namespace mlir::rcgmp

#endif
