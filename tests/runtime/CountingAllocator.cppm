// Allocation hooks that count runtime allocations and check that each
// deallocation repeats the size and alignment of its allocation.
module;
#include "RCGMP/Runtime/Allocation.h"
#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
export module rcgmp.testing.allocator;

namespace rcgmp::test {

struct alignas(std::max_align_t) AllocationHeader {
  std::size_t size;
  std::size_t alignment;
};

std::atomic<std::size_t> totalAllocations, totalReallocations,
    totalDeallocations;

// Counts allocations, reallocations, and deallocations since its
// construction.
export class AllocationCounter {
public:
  std::size_t allocations() const {
    return totalAllocations - allocationsBefore;
  }
  std::size_t reallocations() const {
    return totalReallocations - reallocationsBefore;
  }
  std::size_t deallocations() const {
    return totalDeallocations - deallocationsBefore;
  }
  bool balanced() const { return allocations() == deallocations(); }

private:
  std::size_t allocationsBefore = totalAllocations;
  std::size_t reallocationsBefore = totalReallocations;
  std::size_t deallocationsBefore = totalDeallocations;
};

} // namespace rcgmp::test

extern "C" {

void *rcgmp_allocate(std::size_t size, std::size_t alignment) {
  using rcgmp::test::AllocationHeader;
  EXPECT_LE(alignment, alignof(AllocationHeader));
  auto *header = static_cast<AllocationHeader *>(
      std::malloc(sizeof(AllocationHeader) + size));
  if (!header)
    return nullptr;
  *header = {size, alignment};
  ++rcgmp::test::totalAllocations;
  return header + 1;
}

void *rcgmp_reallocate(void *pointer, std::size_t oldSize, std::size_t size,
                       std::size_t alignment) {
  using rcgmp::test::AllocationHeader;
  auto *header = static_cast<AllocationHeader *>(pointer) - 1;
  EXPECT_EQ(header->size, oldSize);
  EXPECT_EQ(header->alignment, alignment);
  header = static_cast<AllocationHeader *>(
      std::realloc(header, sizeof(AllocationHeader) + size));
  if (!header)
    return nullptr;
  header->size = size;
  ++rcgmp::test::totalReallocations;
  return header + 1;
}

void rcgmp_dealloc(void *pointer, std::size_t size, std::size_t alignment) {
  auto *header = static_cast<rcgmp::test::AllocationHeader *>(pointer) - 1;
  EXPECT_EQ(header->size, size);
  EXPECT_EQ(header->alignment, alignment);
  ++rcgmp::test::totalDeallocations;
  std::free(header);
}
}
