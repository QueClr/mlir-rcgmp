// Checks compiled and runtime reference counting, which share one count,
// against the number of references held.
#include "RCGMP/Runtime/Mpz.h"
#include <fuzztest/fuzztest.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>
import rcgmp.runtime;
import rcgmp.testing;
import rcgmp.testing.allocator;
import rcgmp.testing.differential;

namespace {

using rcgmp::Counting;
using rcgmp::Handle;
using namespace rcgmp::test;

// Retains or releases one reference, in compiled code or the runtime.
struct CountingStep {
  bool retain;
  bool compiled;
};

// Each step keeps at least one reference, so a release of the only one is
// skipped. The references left are then released, and the last release, in
// compiled code or the runtime, frees a counted integer. Immediates, immortal
// integers, and compiled constants are neither counted nor freed.
void CountsReferences(Counting counting,
                      const std::variant<Operand, Constant> &source,
                      bool immortal, const std::vector<CountingStep> &steps,
                      bool compiledLast) {
  std::size_t index = std::to_underlying(counting);
  bool atomic = counting == Counting::atomic;
  auto *constant = std::get_if<Constant>(&source);
  Integer value = constant ? constantValue(constant->index)
                           : std::get<Operand>(source).value;
  AllocationCounter balance;
  void *x = constant
                ? constants[index](constant->index)
                : makeOperand(std::get<Operand>(source), counting, immortal);
  bool heap = !Handle(x).isImmediate();
  bool counted = heap && !constant && !immortal;
  auto release = [&](bool compiled) {
    if (compiled)
      releases[index](x);
    else
      rcgmp_mpz_release(x, atomic);
  };
  AllocationCounter counter;
  std::uintptr_t held = 1;
  for (auto [retain, compiled] : steps) {
    if (retain) {
      EXPECT_EQ(compiled ? retains[index](x) : rcgmp_mpz_retain(x, atomic), x);
      ++held;
    } else if (held > 1) {
      release(compiled);
      --held;
    }
    EXPECT_THAT(x, Represents(value));
    if (heap)
      EXPECT_EQ(references(x, counting), counted ? held : 0);
  }
  for (; held > 1; --held)
    release(true);
  EXPECT_EQ(counter.allocations(), 0u) << "reference counting allocated";
  EXPECT_EQ(counter.deallocations(), 0u) << "freed with references left";
  release(compiledLast);
  EXPECT_EQ(counter.deallocations(), counted ? 1u : 0u)
      << "the last release did not free exactly a counted integer";
  if (heap && immortal && !constant)
    rcgmp_mpz_free(x, atomic);
  EXPECT_TRUE(balance.balanced())
      << "allocations and deallocations do not balance";
}
FUZZ_TEST(Differential, CountsReferences)
    .WithDomains(AnyCounting(),
                 fuzztest::VariantOf(
                     AnyOperand(), fuzztest::StructOf<Constant>(AnyConstant())),
                 fuzztest::Arbitrary<bool>(),
                 fuzztest::VectorOf(fuzztest::StructOf<CountingStep>(
                                        fuzztest::Arbitrary<bool>(),
                                        fuzztest::Arbitrary<bool>()))
                     .WithMaxSize(32),
                 fuzztest::Arbitrary<bool>());

} // namespace
