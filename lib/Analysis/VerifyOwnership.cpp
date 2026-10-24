#include "RCGMP/Analysis/VerifyOwnership.h"
#include "RCGMP/IR/RCGMPDialect.h"
#include "RCGMP/IR/RCGMPOps.h"
#include "RCGMP/IR/RCGMPTypes.h"

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetVector.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Transforms/RegionUtils.h>

#include <optional>

namespace mlir::rcgmp {
namespace {

bool isMpz(Value value) { return isa<MpzType>(value.getType()); }

// Constants need no reference counting, so their uses are not tracked.
bool isConstant(Value value) {
  Operation *op = value.getDefiningOp();
  return op && op->hasTrait<OpTrait::ConstantLike>();
}

// The references owned along one control-flow path, counted per value, and
// the operation that consumed the last reference of each value given up.
struct Path {
  llvm::MapVector<Value, unsigned> owned;
  DenseMap<Value, Operation *> consumers;

  void own(Value value) {
    if (isMpz(value) && !isConstant(value))
      ++owned[value];
  }
};

// The first path reaching a merge point; every other path must agree with it.
using Merge = std::optional<Path>;

// Handles the end of a path through a region: a terminator without block
// successors, or the parent operation when the block has no terminator.
using ExitHandler = function_ref<LogicalResult(Operation *, Path &)>;

MutableArrayRef<OpOperand> getOpOperands(Operation *op, OperandRange range) {
  if (range.empty())
    return {};
  return op->getOpOperands().slice(range.getBeginOperandIndex(), range.size());
}

// How an operation uses one integer operand, from the effects it declares on
// it: freeing it consumes one reference, and any other effect borrows it.
std::optional<bool> consumesOperand(Operation *op, OpOperand &operand) {
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effects)
    return std::nullopt;
  SmallVector<MemoryEffects::EffectInstance> instances;
  effects.getEffects(instances);
  std::optional<bool> consumes;
  for (const MemoryEffects::EffectInstance &instance : instances)
    if (instance.getEffectValue<OpOperand *>() == &operand)
      consumes = consumes.value_or(false) ||
                 isa<MemoryEffects::Free>(instance.getEffect());
  return consumes;
}

// Whether an operation allocates a result, which then owns one reference.
bool allocatesResult(Operation *op, OpResult result) {
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effects)
    return false;
  SmallVector<MemoryEffects::EffectInstance> instances;
  effects.getEffects(instances);
  return llvm::any_of(instances, [&](const MemoryEffects::EffectInstance &e) {
    return e.getEffectValue<OpResult>() == result &&
           isa<MemoryEffects::Allocate>(e.getEffect());
  });
}

LogicalResult emitUnknownOperand(Operation *op, OpOperand &operand) {
  return op->emitOpError() << "cannot determine whether operand #"
                           << operand.getOperandNumber()
                           << " is consumed or borrowed";
}

// Every integer operand of op must be among the handled ones.
LogicalResult checkUnknownOperands(Operation *op,
                                   ArrayRef<MutableArrayRef<OpOperand>> known) {
  SmallPtrSet<OpOperand *, 8> handled;
  for (MutableArrayRef<OpOperand> operands : known)
    for (OpOperand &operand : operands)
      handled.insert(&operand);
  for (OpOperand &operand : op->getOpOperands())
    if (isMpz(operand.get()) && !handled.contains(&operand))
      return emitUnknownOperand(op, operand);
  return success();
}

// References to values defined inside a region must be consumed before the
// path leaves it, at end, an exit of the regions of parent.
LogicalResult checkLeaks(Operation *end, const Path &path, Operation *parent) {
  SmallVector<Value> leaked;
  for (auto [value, count] : path.owned) {
    Operation *owner = value.getParentRegion()->getParentOp();
    if (!parent || parent == owner || parent->isProperAncestor(owner))
      leaked.push_back(value);
  }
  if (leaked.empty())
    return success();
  InFlightDiagnostic diagnostic =
      end->emitOpError("leaves its region without consuming an owned value");
  for (Value value : leaked)
    diagnostic.attachNote(value.getLoc()) << "owned value defined here";
  return diagnostic;
}

LogicalResult merge(Merge &first, Path path, Operation *from) {
  if (!first) {
    first = std::move(path);
    return success();
  }
  SmallVector<Value> differing;
  for (auto [value, count] : path.owned)
    if (first->owned.lookup(value) != count)
      differing.push_back(value);
  for (auto [value, count] : first->owned)
    if (!path.owned.contains(value))
      differing.push_back(value);
  if (differing.empty())
    return success();
  InFlightDiagnostic diagnostic = from->emitOpError(
      "owns different references than another path to the same successor");
  for (Value value : differing) {
    Operation *consumer = path.consumers.lookup(value);
    if (!consumer)
      consumer = first->consumers.lookup(value);
    if (consumer)
      diagnostic.attachNote(consumer->getLoc()) << "consumed here";
    else
      diagnostic.attachNote(value.getLoc()) << "owned value defined here";
  }
  return diagnostic;
}

// The borrowed marker only applies to integer arguments.
LogicalResult verifyBorrowedTypes(FunctionOpInterface function) {
  for (auto [index, type] : llvm::enumerate(function.getArgumentTypes()))
    if (function.getArgAttr(index, RCGMPDialect::getBorrowedAttrName()) &&
        !isa<MpzType>(type))
      return function.emitError()
             << "'" << RCGMPDialect::getBorrowedAttrName() << "' on argument #"
             << index << " requires an mpz type, but found " << type;
  return success();
}

bool borrowsArguments(FunctionOpInterface function) {
  return llvm::any_of(
      llvm::seq(function.getNumArguments()), [&](unsigned index) {
        return function.getArgAttr(index, RCGMPDialect::getBorrowedAttrName());
      });
}

// Indirect calls consume every argument, so a function that borrows any of
// its arguments may only be referenced as the callee of a call.
LogicalResult verifyBorrowingReferences(Operation *root,
                                        SymbolTableCollection &symbols) {
  WalkResult result = root->walk([&](Operation *op) {
    auto call = dyn_cast<CallOpInterface>(op);
    auto callee =
        call ? dyn_cast<SymbolRefAttr>(call.getCallableForCallee()) : nullptr;
    WalkResult references = op->getAttrDictionary().walk<WalkOrder::PreOrder>(
        [&](SymbolRefAttr reference) {
          if (reference == callee)
            return WalkResult::skip();
          auto function = dyn_cast_if_present<FunctionOpInterface>(
              symbols.lookupNearestSymbolFrom(op, reference));
          if (!function || !borrowsArguments(function))
            return WalkResult::skip();
          InFlightDiagnostic diagnostic =
              op->emitOpError() << "refers to " << reference
                                << ", which borrows arguments, other than as "
                                   "the callee of a call";
          diagnostic.attachNote(function.getLoc()) << "function declared here";
          return WalkResult::interrupt();
        });
    return references.wasInterrupted() ? WalkResult::interrupt()
                                       : WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

// Checks one function body. Arguments are owned unless marked borrowed, and
// returning a value consumes it.
class FunctionVerifier {
public:
  explicit FunctionVerifier(SymbolTableCollection &symbols)
      : symbols(symbols) {}

  LogicalResult verify(FunctionOpInterface function);

private:
  LogicalResult verifyRegion(Region &region, Path entry, ExitHandler exit);
  LogicalResult verifyBranch(Operation *terminator, const Path &path,
                             DenseMap<Block *, Merge> &merges);
  LogicalResult verifyOperation(Operation *op, std::optional<Path> &path);
  LogicalResult verifyCall(CallOpInterface call, Path &path);
  LogicalResult verifyRegionBranch(RegionBranchOpInterface op,
                                   std::optional<Path> &path);
  LogicalResult verifyEffects(Operation *op, Path &path,
                              ArrayRef<MutableArrayRef<OpOperand>> skipped);
  LogicalResult verifyUnknownRegions(Operation *op);

  LogicalResult consume(Path &path, OpOperand &operand);
  LogicalResult borrow(const Path &path, OpOperand &operand);
  FailureOr<Path> transfer(Path path, MutableArrayRef<OpOperand> forwarded);

  SymbolTableCollection &symbols;
  // Values that hold no reference of their own: arguments the caller keeps,
  // and results an operation returns without allocating them. They may be
  // borrowed or retained while live, but consuming one needs a retain first.
  DenseSet<Value> borrowed;
};

LogicalResult FunctionVerifier::verify(FunctionOpInterface function) {
  Path entry;
  for (BlockArgument argument : function.getArguments()) {
    if (function.getArgAttr(argument.getArgNumber(),
                            RCGMPDialect::getBorrowedAttrName()))
      borrowed.insert(argument);
    else
      entry.own(argument);
  }
  return verifyRegion(function.getFunctionBody(), std::move(entry),
                      [&](Operation *end, Path &path) -> LogicalResult {
                        if (end->hasTrait<OpTrait::ReturnLike>()) {
                          for (OpOperand &operand : end->getOpOperands())
                            if (isMpz(operand.get()) &&
                                failed(consume(path, operand)))
                              return failure();
                          return checkLeaks(end, path, nullptr);
                        }
                        // A terminator that never returns, such as
                        // llvm.unreachable, may still own references.
                        if (end->hasTrait<OpTrait::IsTerminator>())
                          return checkUnknownOperands(end, {});
                        return checkLeaks(end, path, nullptr);
                      });
}

LogicalResult FunctionVerifier::verifyRegion(Region &region, Path entry,
                                             ExitHandler exit) {
  Block *front = &region.front();
  DenseMap<Block *, Merge> merges;
  merges[front] = std::move(entry);
  // Blocks are verified along the first path reaching them; later paths are
  // compared against it. Blocks that no path reaches are never verified.
  llvm::SetVector<Block *> worklist;
  worklist.insert(front);
  for (size_t i = 0; i < worklist.size(); ++i) {
    Block *block = worklist[i];
    std::optional<Path> path = merges[block];
    if (block != front)
      for (BlockArgument argument : block->getArguments())
        path->own(argument);
    Operation *terminator =
        block->mightHaveTerminator() ? block->getTerminator() : nullptr;
    Block::iterator last =
        terminator ? Block::iterator(terminator) : block->end();
    for (Operation &op : llvm::make_range(block->begin(), last)) {
      if (failed(verifyOperation(&op, path)))
        return failure();
      if (!path)
        break;
    }
    if (!path)
      continue;
    if (terminator && failed(verifyUnknownRegions(terminator)))
      return failure();
    if (terminator && terminator->getNumSuccessors() != 0) {
      if (failed(verifyBranch(terminator, *path, merges)))
        return failure();
      worklist.insert_range(terminator->getSuccessors());
    } else if (failed(exit(terminator ? terminator : region.getParentOp(),
                           *path))) {
      return failure();
    }
  }
  return success();
}

LogicalResult FunctionVerifier::verifyBranch(Operation *terminator,
                                             const Path &path,
                                             DenseMap<Block *, Merge> &merges) {
  auto branch = dyn_cast<BranchOpInterface>(terminator);
  SmallVector<MutableArrayRef<OpOperand>> forwarded;
  for (auto [index, successor] : llvm::enumerate(terminator->getSuccessors())) {
    MutableArrayRef<OpOperand> operands;
    unsigned produced = successor->getNumArguments();
    if (branch) {
      SuccessorOperands successorOperands = branch.getSuccessorOperands(index);
      operands = successorOperands.getMutableForwardedOperands();
      produced = successorOperands.getProducedOperandCount();
    }
    for (BlockArgument argument :
         successor->getArguments().take_front(produced))
      if (isMpz(argument))
        return terminator->emitOpError()
               << "cannot determine ownership of block argument #"
               << argument.getArgNumber();
    forwarded.push_back(operands);
  }
  if (failed(checkUnknownOperands(terminator, forwarded)))
    return failure();
  for (auto [successor, operands] :
       llvm::zip_equal(terminator->getSuccessors(), forwarded)) {
    FailureOr<Path> next = transfer(path, operands);
    if (failed(next) ||
        failed(merge(merges[successor], std::move(*next), terminator)))
      return failure();
  }
  return success();
}

// Resets path when control never reaches the operation after op.
LogicalResult FunctionVerifier::verifyOperation(Operation *op,
                                                std::optional<Path> &path) {
  if (auto retain = dyn_cast<RetainOp>(op)) {
    if (failed(borrow(*path, retain.getValueMutable())))
      return failure();
    path->own(retain.getValue());
    return success();
  }
  if (auto call = dyn_cast<CallOpInterface>(op))
    return verifyCall(call, *path);
  if (auto branch = dyn_cast<RegionBranchOpInterface>(op))
    return verifyRegionBranch(branch, path);
  if (failed(verifyEffects(op, *path, {})))
    return failure();
  return verifyUnknownRegions(op);
}

LogicalResult FunctionVerifier::verifyCall(CallOpInterface call, Path &path) {
  auto callee = dyn_cast_if_present<FunctionOpInterface>(
      call.resolveCallableInTable(&symbols));
  MutableArrayRef<OpOperand> arguments = call.getArgOperandsMutable();
  if (failed(checkUnknownOperands(call, arguments)))
    return failure();
  // Consuming first rejects borrowing a reference that the call consumes.
  SmallVector<OpOperand *> borrows;
  for (auto [index, operand] : llvm::enumerate(arguments)) {
    if (!isMpz(operand.get()))
      continue;
    if (callee && callee.getArgAttr(index, RCGMPDialect::getBorrowedAttrName()))
      borrows.push_back(&operand);
    else if (failed(consume(path, operand)))
      return failure();
  }
  for (OpOperand *operand : borrows)
    if (failed(borrow(path, *operand)))
      return failure();
  for (Value result : call->getResults())
    path.own(result);
  return success();
}

LogicalResult FunctionVerifier::verifyRegionBranch(RegionBranchOpInterface op,
                                                   std::optional<Path> &path) {
  SmallVector<Merge> regionMerges(op->getNumRegions());
  Merge after;
  SmallVector<Region *> worklist;

  auto branch = [&](Operation *from, const Path &current,
                    ArrayRef<RegionSuccessor> successors) -> LogicalResult {
    SmallVector<MutableArrayRef<OpOperand>> forwarded;
    for (RegionSuccessor successor : successors) {
      OperandRange operands =
          from == op ? op.getEntrySuccessorOperands(successor)
                     : cast<RegionBranchTerminatorOpInterface>(from)
                           .getSuccessorOperands(successor);
      forwarded.push_back(getOpOperands(from, operands));
    }
    if (from != op && failed(checkUnknownOperands(from, forwarded)))
      return failure();
    for (auto [successor, operands] : llvm::zip_equal(successors, forwarded)) {
      FailureOr<Path> next = transfer(current, operands);
      if (failed(next) || (from != op && failed(checkLeaks(from, *next, op))))
        return failure();
      Region *region = successor.getSuccessor();
      Merge &slot = region ? regionMerges[region->getRegionNumber()] : after;
      if (region && !slot)
        worklist.push_back(region);
      if (failed(merge(slot, std::move(*next), from)))
        return failure();
    }
    return success();
  };
  auto exit = [&](Operation *end, Path &current) -> LogicalResult {
    if (auto terminator = dyn_cast<RegionBranchTerminatorOpInterface>(end)) {
      SmallVector<RegionSuccessor> successors;
      op.getSuccessorRegions(RegionBranchPoint(terminator), successors);
      return branch(end, current, successors);
    }
    if (end->hasTrait<OpTrait::IsTerminator>())
      return checkUnknownOperands(end, {});
    // Control may return to the parent from an exit that is not known to be
    // a terminator.
    if (failed(checkUnknownOperands(end, {})) ||
        failed(checkLeaks(end, current, op)))
      return failure();
    return merge(after, current, end);
  };

  // Operands entering no region are used by the operation itself, as the
  // scrutinee of rcgmp.match is.
  SmallVector<RegionSuccessor> entries;
  op.getSuccessorRegions(RegionBranchPoint::parent(), entries);
  SmallVector<MutableArrayRef<OpOperand>> entering;
  for (RegionSuccessor successor : entries)
    entering.push_back(
        getOpOperands(op, op.getEntrySuccessorOperands(successor)));
  if (failed(verifyEffects(op, *path, entering)) ||
      failed(branch(op, *path, entries)))
    return failure();
  // Regions are verified along the first path reaching them; later paths are
  // compared against it. Their integer arguments own one reference each,
  // whether forwarded or bound by the operation itself.
  for (size_t i = 0; i < worklist.size(); ++i) {
    Region *region = worklist[i];
    Path entry = *regionMerges[region->getRegionNumber()];
    for (BlockArgument argument : region->getArguments())
      entry.own(argument);
    if (failed(verifyRegion(*region, std::move(entry), exit)))
      return failure();
  }
  // Without a path returning to the parent, the code after the operation is
  // unreachable.
  path = std::move(after);
  if (path)
    for (Value result : op->getResults())
      path->own(result);
  return success();
}

// Uses the integer operands of op, except the skipped ones, as the effects
// declared on them say, and takes the references its results own.
LogicalResult
FunctionVerifier::verifyEffects(Operation *op, Path &path,
                                ArrayRef<MutableArrayRef<OpOperand>> skipped) {
  SmallPtrSet<OpOperand *, 8> skip;
  for (MutableArrayRef<OpOperand> operands : skipped)
    for (OpOperand &operand : operands)
      skip.insert(&operand);
  SmallVector<OpOperand *> borrows;
  for (OpOperand &operand : op->getOpOperands()) {
    if (!isMpz(operand.get()) || skip.contains(&operand))
      continue;
    std::optional<bool> consumes = consumesOperand(op, operand);
    if (!consumes)
      return emitUnknownOperand(op, operand);
    if (!*consumes)
      borrows.push_back(&operand);
    else if (failed(consume(path, operand)))
      return failure();
  }
  for (OpOperand *operand : borrows)
    if (failed(borrow(path, *operand)))
      return failure();
  if (isa<RegionBranchOpInterface>(op))
    return success();
  for (OpResult result : op->getResults()) {
    if (!isMpz(result) || isConstant(result))
      continue;
    if (!isa<MemoryEffectOpInterface>(op))
      return op->emitOpError() << "cannot determine ownership of result #"
                               << result.getResultNumber();
    if (allocatesResult(op, result))
      path.own(result);
    else
      borrowed.insert(result);
  }
  return success();
}

// Without knowing how control enters and leaves the regions of op, they may
// only use references they create and consume themselves.
LogicalResult FunctionVerifier::verifyUnknownRegions(Operation *op) {
  OpOperand *outside = nullptr;
  visitUsedValuesDefinedAbove(op->getRegions(), [&](OpOperand *use) {
    if (!outside && isMpz(use->get()))
      outside = use;
  });
  if (outside)
    return outside->getOwner()->emitOpError()
           << "uses an integer from outside the regions of '" << op->getName()
           << "', whose ownership cannot be determined";
  // Any terminator may return control to the parent, so even a path ending
  // in a terminator such as llvm.unreachable must consume what it creates.
  for (Region &region : op->getRegions()) {
    if (region.empty())
      continue;
    Path entry;
    for (BlockArgument argument : region.getArguments())
      if (isMpz(argument))
        return op->emitOpError()
               << "cannot determine ownership of region argument #"
               << argument.getArgNumber();
    if (failed(verifyRegion(region, std::move(entry),
                            [&](Operation *end, Path &current) {
                              if (failed(checkUnknownOperands(end, {})))
                                return failure();
                              return checkLeaks(end, current, op);
                            })))
      return failure();
  }
  return success();
}

LogicalResult FunctionVerifier::consume(Path &path, OpOperand &operand) {
  Value value = operand.get();
  if (isConstant(value))
    return success();
  auto owned = path.owned.find(value);
  if (owned == path.owned.end()) {
    InFlightDiagnostic diagnostic = operand.getOwner()->emitOpError()
                                    << "operand #" << operand.getOperandNumber()
                                    << " consumes a reference it does not own";
    if (Operation *consumer = path.consumers.lookup(value))
      diagnostic.attachNote(consumer->getLoc())
          << "last reference consumed here";
    return diagnostic;
  }
  if (--owned->second == 0)
    path.owned.erase(owned);
  path.consumers[value] = operand.getOwner();
  return success();
}

LogicalResult FunctionVerifier::borrow(const Path &path, OpOperand &operand) {
  Value value = operand.get();
  if (isConstant(value) || borrowed.contains(value) ||
      path.owned.contains(value))
    return success();
  InFlightDiagnostic diagnostic = operand.getOwner()->emitOpError()
                                  << "operand #" << operand.getOperandNumber()
                                  << " borrows a value after its last "
                                     "reference was consumed";
  if (Operation *consumer = path.consumers.lookup(value))
    diagnostic.attachNote(consumer->getLoc()) << "last reference consumed here";
  return diagnostic;
}

FailureOr<Path>
FunctionVerifier::transfer(Path path, MutableArrayRef<OpOperand> forwarded) {
  for (OpOperand &operand : forwarded)
    if (isMpz(operand.get()) && failed(consume(path, operand)))
      return failure();
  return path;
}

struct VerifyOwnershipPass
    : PassWrapper<VerifyOwnershipPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyOwnershipPass)
  StringRef getArgument() const override { return "rcgmp-verify-ownership"; }
  StringRef getDescription() const override {
    return "Check that each owned rcgmp reference is consumed exactly once on "
           "every path";
  }
  void runOnOperation() override {
    if (failed(verifyOwnership(getOperation())))
      signalPassFailure();
    markAllAnalysesPreserved();
  }
};

} // namespace

LogicalResult verifyOwnership(Operation *root) {
  SymbolTableCollection symbols;
  bool valid = succeeded(verifyBorrowingReferences(root, symbols));
  root->walk<WalkOrder::PreOrder>([&](FunctionOpInterface function) {
    if (failed(verifyBorrowedTypes(function)) ||
        (!function.isExternal() &&
         failed(FunctionVerifier(symbols).verify(function))))
      valid = false;
    return WalkResult::skip();
  });
  return success(valid);
}

std::unique_ptr<Pass> createVerifyOwnershipPass() {
  return std::make_unique<VerifyOwnershipPass>();
}

} // namespace mlir::rcgmp
