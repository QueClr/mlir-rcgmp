#ifndef RCGMP_RUNTIME_ALLOCATION_H
#define RCGMP_RUNTIME_ALLOCATION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Return suitably aligned storage, or NULL on allocation failure. The runtime
// terminates on failure. Supply both functions when replacing the allocator.
void *rcgmp_allocate(size_t size, size_t alignment);
void rcgmp_dealloc(void *pointer, size_t size, size_t alignment);

#ifdef __cplusplus
}
#endif

#endif
