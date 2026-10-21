// The default allocation hooks, which a program replaces by defining both.
module;
#include "RCGMP/Runtime/Allocation.h"

#include <cstddef>
#include <new>
export module rcgmp.runtime.allocation;

extern "C" {

void *rcgmp_allocate(std::size_t size, std::size_t alignment) {
  return ::operator new(size, std::align_val_t{alignment}, std::nothrow);
}

void rcgmp_dealloc(void *pointer, std::size_t size, std::size_t alignment) {
  ::operator delete(pointer, size, std::align_val_t{alignment});
}
}
