// The default allocation hooks, which a program replaces by defining both.
module;
#include "RCGMP/Runtime/Allocation.h"

#include <cstddef>
#include <cstdlib>
#include <new>
export module rcgmp.runtime.allocation;

extern "C" {

// Alignments that malloc already provides avoid the aligned allocation path.
void *rcgmp_allocate(std::size_t size, std::size_t alignment) {
  if (alignment <= alignof(std::max_align_t))
    return std::malloc(size);
  return ::operator new(size, std::align_val_t{alignment}, std::nothrow);
}

void rcgmp_dealloc(void *pointer, std::size_t size, std::size_t alignment) {
  if (alignment <= alignof(std::max_align_t))
    std::free(pointer);
  else
    ::operator delete(pointer, size, std::align_val_t{alignment});
}
}
