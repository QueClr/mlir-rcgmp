#include "RCGMP/Runtime/Allocation.h"

#include <stdlib.h>

void *rcgmp_allocate(size_t size, size_t alignment) {
  if (alignment > _Alignof(max_align_t))
    return NULL;
  return malloc(size);
}

void rcgmp_dealloc(void *pointer, size_t size, size_t alignment) {
  (void)size;
  (void)alignment;
  free(pointer);
}
