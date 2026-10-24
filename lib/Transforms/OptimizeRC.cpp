#include "RCGMP/Transforms/OptimizeRC.h"
#include "RCGMP/IR/RCGMPDialect.h"
#include "RCGMP/IR/RCGMPOps.h"

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>

#include <algorithm>
#include <optional>

namespace mlir::rcgmp {
namespace {

// How an operation uses the references owned by one integer value, over every
// path through it: entering, the value must own at least `required`
// references, and leaving, it owns between `fewest` and `most` more than it
// did, which is negative when references are consumed.
struct Usage {
  int64_t required = 0;
  int64_t fewest = 0;
  int64_t most = 0;
};

bool uses(Operation *op, Value value) {
  return llvm::any_of(value.getUsers(),
                      [&](Operation *user) { return op->isAncestor(user); });
}

std::optional<Usage> summarize(Operation *op, Value value);

// The operations of [begin, end) run in order.
std::optional<Usage> summarizeRange(Block::iterator begin, Block::iterator end,
                                    Value value) {
  Usage total;
  for (Operation &op : llvm::make_range(begin, end)) {
    if (!uses(&op, value))
      continue;
    std::optional<Usage> usage = summarize(&op, value);
    if (!usage)
      return std::nullopt;
    total.required = std::max(total.required, usage->required - total.fewest);
    total.fewest += usage->fewest;
    total.most += usage->most;
  }
  return total;
}

std::optional<Usage> summarizeRegion(Region &region, Value value) {
  if (!region.hasOneBlock())
    return std::nullopt;
  return summarizeRange(region.front().begin(), region.front().end(), value);
}

// Exactly one of the non-empty regions runs, or none when mayRunNone is set.
std::optional<Usage> summarizeAlternatives(MutableArrayRef<Region> regions,
                                           bool mayRunNone, Value value) {
  std::optional<Usage> combined;
  if (mayRunNone)
    combined = Usage{};
  for (Region &region : regions) {
    if (region.empty())
      continue;
    std::optional<Usage> usage = summarizeRegion(region, value);
    if (!usage)
      return std::nullopt;
    if (!combined) {
      combined = usage;
      continue;
    }
    combined->required = std::max(combined->required, usage->required);
    combined->fewest = std::min(combined->fewest, usage->fewest);
    combined->most = std::max(combined->most, usage->most);
  }
  return combined.value_or(Usage{});
}

// A loop body runs any number of times, which leaves the count unchanged only
// when each run does.
std::optional<Usage> summarizeLoop(MutableArrayRef<Region> regions,
                                   Value value) {
  Usage combined;
  for (Region &region : regions) {
    std::optional<Usage> usage = summarizeRegion(region, value);
    if (!usage || usage->fewest || usage->most)
      return std::nullopt;
    combined.required = std::max(combined.required, usage->required);
  }
  return combined;
}

// Operations other than rcgmp.retain never add references. A public rcgmp
// operation declares each consumed operand with a free effect, and every other
// operation may consume each operand it receives.
std::optional<Usage> summarizeLeaf(Operation *op, Value value) {
  int64_t occurrences = llvm::count(op->getOperands(), value);
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  if (isa_and_nonnull<RCGMPDialect>(op->getDialect()) && effects) {
    // Bridge operations count references without declaring it.
    if (op->getName().getStringRef().starts_with("rcgmp.bridge."))
      return std::nullopt;
    SmallVector<MemoryEffects::EffectInstance> instances;
    effects.getEffects(instances);
    int64_t consumed = 0;
    bool declared = false;
    for (const MemoryEffects::EffectInstance &instance : instances) {
      if (instance.getValue() != value)
        continue;
      declared = true;
      consumed += isa<MemoryEffects::Free>(instance.getEffect());
    }
    if (declared) {
      int64_t borrowed = occurrences - consumed;
      return Usage{consumed + (borrowed > 0), -consumed, -consumed};
    }
  }
  return Usage{occurrences, -occurrences, 0};
}

std::optional<Usage> summarize(Operation *op, Value value) {
  if (isa<RetainOp>(op))
    return Usage{1, 1, 1};
  if (isa<ReleaseOp>(op))
    return Usage{1, -1, -1};
  if (!op->getNumRegions())
    return summarizeLeaf(op, value);
  if (auto match = dyn_cast<MatchOp>(op)) {
    std::optional<Usage> arms =
        summarizeAlternatives(match.getBodies(), false, value);
    if (!arms || match.getScrutinee() != value)
      return arms;
    // The scrutinee is consumed before an arm runs.
    return Usage{arms->required + 1, arms->fewest - 1, arms->most - 1};
  }
  if (llvm::is_contained(op->getOperands(), value))
    return std::nullopt;
  if (auto branch = dyn_cast<scf::IfOp>(op))
    return summarizeAlternatives(op->getRegions(),
                                 branch.getElseRegion().empty(), value);
  if (isa<scf::IndexSwitchOp>(op))
    return summarizeAlternatives(op->getRegions(), false, value);
  if (isa<scf::ForOp, scf::WhileOp>(op))
    return summarizeLoop(op->getRegions(), value);
  return std::nullopt;
}

// Whether a retain just before begin and a release at end, which may be a
// position where one is about to be inserted, can both be removed. Each use in
// between then holds one reference fewer, so it must have one to spare. A use
// owns at least two references counting from the retain, which had one to
// add to, and at least one more than the uses up to the release consume.
bool canCancel(Block::iterator begin, Block::iterator end, Value value) {
  SmallVector<Usage> usages;
  for (Operation &op : llvm::make_range(begin, end)) {
    if (!uses(&op, value))
      continue;
    std::optional<Usage> usage = summarize(&op, value);
    if (!usage)
      return false;
    usages.push_back(*usage);
  }
  int64_t before = 0, after = 0;
  for (const Usage &usage : usages)
    after += usage.most;
  for (const Usage &usage : usages) {
    if (std::max(2 + before, 1 - after) < usage.required + 1)
      return false;
    before += usage.fewest;
    after -= usage.most;
  }
  return true;
}

// Removes a retain with the nearest later release in its block that it can
// cancel. When the value's last use between them consumes it, that use now
// receives the last reference and can reuse the storage.
bool cancel(RetainOp retain) {
  Value value = retain.getValue();
  SmallVector<ReleaseOp> releases;
  for (Operation *user : value.getUsers()) {
    auto release = dyn_cast<ReleaseOp>(user);
    if (release && release->getBlock() == retain->getBlock() &&
        retain->isBeforeInBlock(release))
      releases.push_back(release);
  }
  llvm::sort(releases, [](ReleaseOp lhs, ReleaseOp rhs) {
    return lhs->isBeforeInBlock(rhs);
  });
  for (ReleaseOp release : releases) {
    if (!canCancel(std::next(retain->getIterator()), release->getIterator(),
                   value))
      continue;
    release.erase();
    retain.erase();
    return true;
  }
  return false;
}

// An operation that runs exactly one of its regions, into which reference
// counting operations next to it can move. An scf.if without results gains an
// empty else region when needed.
bool runsOneRegion(Operation *op, Value value) {
  if (llvm::is_contained(op->getOperands(), value))
    return false;
  if (auto branch = dyn_cast<scf::IfOp>(op))
    return !branch.getElseRegion().empty() || !branch.getNumResults();
  return isa<scf::IndexSwitchOp, MatchOp>(op);
}

void addMissingElse(OpBuilder &builder, Operation *op) {
  auto branch = dyn_cast<scf::IfOp>(op);
  if (!branch || !branch.getElseRegion().empty())
    return;
  OpBuilder::InsertionGuard guard(builder);
  builder.createBlock(&branch.getElseRegion());
  scf::YieldOp::create(builder, branch.getLoc());
}

template <typename Op> SmallVector<Op> usersIn(Block &block, Value value) {
  SmallVector<Op> found;
  for (Operation *user : value.getUsers())
    if (auto op = dyn_cast<Op>(user); op && op->getBlock() == &block)
      found.push_back(op);
  return found;
}

// Moves a retain into each region of the next operation using its value when
// that lets the retain cancel a release in one of them. Moving it past
// operations not using the value only defers the increment.
bool pushIntoRegions(OpBuilder &builder, RetainOp retain) {
  Value value = retain.getValue();
  Operation *next = retain->getNextNode();
  while (next && !uses(next, value))
    next = next->getNextNode();
  if (!next || !runsOneRegion(next, value))
    return false;
  bool cancels = llvm::any_of(next->getRegions(), [&](Region &region) {
    if (region.empty())
      return false;
    return llvm::any_of(usersIn<ReleaseOp>(region.front(), value),
                        [&](ReleaseOp release) {
                          return canCancel(region.front().begin(),
                                           release->getIterator(), value);
                        });
  });
  if (!cancels)
    return false;
  addMissingElse(builder, next);
  for (Region &region : next->getRegions()) {
    builder.setInsertionPointToStart(&region.front());
    RetainOp::create(builder, retain.getLoc(), value);
  }
  retain.erase();
  return true;
}

// Moves a release into the end of each region of the previous operation using
// its value when that lets the release cancel a retain in one of them. Moving
// it past operations not using the value only frees the value sooner.
bool sinkIntoRegions(OpBuilder &builder, ReleaseOp release) {
  Value value = release.getValue();
  Operation *previous = release->getPrevNode();
  while (previous && !uses(previous, value))
    previous = previous->getPrevNode();
  if (!previous || !runsOneRegion(previous, value))
    return false;
  for (Region &region : previous->getRegions())
    if (!region.empty() && uses(region.front().getTerminator(), value))
      return false;
  bool cancels = llvm::any_of(previous->getRegions(), [&](Region &region) {
    if (region.empty())
      return false;
    Block &block = region.front();
    return llvm::any_of(usersIn<RetainOp>(block, value), [&](RetainOp retain) {
      return canCancel(std::next(retain->getIterator()),
                       block.getTerminator()->getIterator(), value);
    });
  });
  if (!cancels)
    return false;
  addMissingElse(builder, previous);
  for (Region &region : previous->getRegions()) {
    builder.setInsertionPoint(region.front().getTerminator());
    ReleaseOp::create(builder, release.getLoc(), value);
  }
  release.erase();
  return true;
}

struct OptimizeRCPass : PassWrapper<OptimizeRCPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizeRCPass)
  StringRef getArgument() const override { return "rcgmp-optimize-rc"; }
  StringRef getDescription() const override {
    return "Cancel retains with releases of rcgmp integers, moving them into "
           "branches where that lets them cancel";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<RCGMPDialect, scf::SCFDialect>();
  }

  // Retains and releases only move into nested regions, and each move enables
  // a cancellation, so the loop ends.
  void runOnOperation() override {
    OpBuilder builder(&getContext());
    bool changed;
    do {
      changed = false;
      SmallVector<RetainOp> retains;
      getOperation()->walk([&](RetainOp retain) { retains.push_back(retain); });
      for (RetainOp retain : retains)
        changed |= cancel(retain) || pushIntoRegions(builder, retain);
      SmallVector<ReleaseOp> releases;
      getOperation()->walk(
          [&](ReleaseOp release) { releases.push_back(release); });
      for (ReleaseOp release : releases)
        changed |= sinkIntoRegions(builder, release);
    } while (changed);
  }
};

} // namespace

std::unique_ptr<Pass> createOptimizeRCPass() {
  return std::make_unique<OptimizeRCPass>();
}

} // namespace mlir::rcgmp
