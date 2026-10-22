#include "MatchLowering.h"
#include "RCGMP/IR/RCGMPBridgeOps.h"
#include "RCGMP/IR/RCGMPOps.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/BLAKE3.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Matchers.h>

#include <functional>

namespace mlir::rcgmp {
namespace {

// Interior boundaries spanning fewer keys than this are dispatched through a
// table indexed by the clamped key.
constexpr unsigned maxWindow = 64;
// Up to this many isolated keys over one background are tested for equality.
constexpr unsigned maxPointChain = 4;
// Multipliers tried for each table size of a perfect hash.
constexpr unsigned maxHashAttempts = 4096;
// Shared limb runs up to this length are compared inline. Longer runs compare
// the whole integer in the runtime.
constexpr unsigned maxInlineLimbs = 4;

Value integerConstant(OpBuilder &builder, Location loc, Type type,
                      const APInt &value) {
  return arith::ConstantOp::create(builder, loc,
                                   builder.getIntegerAttr(type, value));
}

// Compares integers of any widths as signed values.
bool lessSigned(const APInt &a, const APInt &b) {
  unsigned width = std::max(a.getBitWidth(), b.getBitWidth());
  return a.sext(width).slt(b.sext(width));
}

// The pieces of intervals within [low, high], each starting at the first
// integer of an interval or at low, at a width holding every start. A missing
// bound leaves that side unbounded.
SmallVector<std::pair<APInt, unsigned>> clip(ArrayRef<MatchInterval> intervals,
                                             std::optional<APInt> low,
                                             std::optional<APInt> high) {
  unsigned width = intervals.front().start.getBitWidth();
  for (const std::optional<APInt> &bound : {low, high})
    if (bound)
      width = std::max(width, bound->getBitWidth());
  ++width;
  SmallVector<std::pair<APInt, unsigned>> pieces;
  for (auto [i, interval] : llvm::enumerate(intervals)) {
    bool first = i == 0, last = i + 1 == intervals.size();
    if (!first && high && lessSigned(*high, interval.start))
      break;
    if (!last && low && !lessSigned(*low, intervals[i + 1].start))
      continue;
    APInt start = interval.start.sext(width);
    if (low && (first || lessSigned(interval.start, *low)))
      start = low->sext(width);
    else if (first)
      start = APInt::getSignedMinValue(width);
    pieces.emplace_back(start, *interval.arm);
  }
  return pieces;
}

// Keys from start up to the start of the next piece map to one leaf. The first
// piece starts at the least key that can occur.
struct Piece {
  APInt start;
  unsigned leaf;
};

using Values = SmallVector<Value, 2>;
using LeafBuilder = function_ref<Values(unsigned)>;

// The values of a leaf, if they are integers known before selecting it, which
// lets a table or a hash select it.
using KnownLeaf = std::function<std::optional<SmallVector<APInt>>(unsigned)>;

// Builds structured control flow yielding values of fixed types.
class Selector {
public:
  Selector(PatternRewriter &rewriter, Location loc, TypeRange types,
           KnownLeaf knownLeaf = nullptr)
      : rewriter(rewriter), loc(loc), types(llvm::to_vector(types)),
        knownLeaf(std::move(knownLeaf)) {}

  Values branch(Value condition, function_ref<Values()> then,
                function_ref<Values()> otherwise) {
    auto branch = scf::IfOp::create(rewriter, loc, types, condition,
                                    /*withElseRegion=*/true);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&branch.getThenRegion().front());
    scf::YieldOp::create(rewriter, loc, then());
    rewriter.setInsertionPointToStart(&branch.getElseRegion().front());
    scf::YieldOp::create(rewriter, loc, otherwise());
    return Values(branch.getResults());
  }

  // Selects the leaf of the piece holding key, which must not be below the
  // first piece's start.
  Values dispatch(Value key, bool isSigned, ArrayRef<Piece> pieces,
                  LeafBuilder leaf) {
    if (pieces.size() == 1)
      return leaf(pieces.front().leaf);
    bool points = isPointSet(pieces);
    if (points && pieces.size() / 2 <= maxPointChain)
      return pointChain(key, pieces, leaf);
    APInt low = pieces[1].start, high = pieces.back().start;
    if (pieces.size() > 3 && (high - low).ult(maxWindow))
      return window(key, isSigned, pieces, leaf);
    if (points)
      if (std::optional<Values> hashed = perfectHash(key, pieces))
        return *hashed;
    size_t middle = pieces.size() / 2;
    auto below = arith::CmpIOp::create(
        rewriter, loc,
        isSigned ? arith::CmpIPredicate::slt : arith::CmpIPredicate::ult, key,
        integerConstant(rewriter, loc, key.getType(), pieces[middle].start));
    return branch(
        below,
        [&] {
          return dispatch(key, isSigned, pieces.take_front(middle), leaf);
        },
        [&] {
          return dispatch(key, isSigned, pieces.drop_front(middle), leaf);
        });
  }

private:
  // Single keys between pieces of one leaf.
  static bool isPointSet(ArrayRef<Piece> pieces) {
    if (pieces.size() % 2 == 0)
      return false;
    for (size_t i = 1; i < pieces.size(); i += 2)
      if (pieces[i + 1].start != pieces[i].start + 1 ||
          pieces[i + 1].leaf != pieces.front().leaf)
        return false;
    return true;
  }

  Values pointChain(Value key, ArrayRef<Piece> pieces, LeafBuilder leaf) {
    if (pieces.size() == 1)
      return leaf(pieces.front().leaf);
    auto equal = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, key,
        integerConstant(rewriter, loc, key.getType(), pieces[1].start));
    return branch(
        equal, [&] { return leaf(pieces[1].leaf); },
        [&] { return pointChain(key, pieces.drop_front(2), leaf); });
  }

  // Clamps the key to one below the first interior boundary and the last one,
  // and switches on its offset. Every leaf other than the most frequent one has
  // its own cases.
  Values window(Value key, bool isSigned, ArrayRef<Piece> pieces,
                LeafBuilder leaf) {
    Type type = key.getType();
    APInt base = pieces[1].start - 1, high = pieces.back().start;
    Value lowest = integerConstant(rewriter, loc, type, base);
    Value highest = integerConstant(rewriter, loc, type, high);
    Value clamped;
    if (isSigned)
      clamped = arith::MaxSIOp::create(
          rewriter, loc, arith::MinSIOp::create(rewriter, loc, key, highest),
          lowest);
    else
      clamped = arith::MaxUIOp::create(
          rewriter, loc, arith::MinUIOp::create(rewriter, loc, key, highest),
          lowest);
    Value offset = arith::SubIOp::create(rewriter, loc, clamped, lowest);
    Value index = arith::IndexCastUIOp::create(rewriter, loc,
                                               rewriter.getIndexType(), offset);
    uint64_t entries = (high - base).getZExtValue() + 1;
    SmallVector<unsigned> table;
    const Piece *piece = pieces.begin();
    for (uint64_t entry = 0; entry < entries; ++entry) {
      APInt value = base + entry;
      while (std::next(piece) != pieces.end() &&
             (isSigned ? std::next(piece)->start.sle(value)
                       : std::next(piece)->start.ule(value)))
        ++piece;
      table.push_back(piece->leaf);
    }
    if (std::optional<SmallVector<SmallVector<APInt>>> rows =
            knownLeaves(table))
      return loadRow(*rows, index, types);
    llvm::SmallDenseMap<unsigned, unsigned> frequency;
    for (unsigned entry : table)
      ++frequency[entry];
    unsigned common = llvm::max_element(frequency, [](auto a, auto b) {
                        return a.second < b.second;
                      })->first;
    SmallVector<int64_t> cases;
    for (auto [entry, target] : llvm::enumerate(table))
      if (target != common)
        cases.push_back(entry);
    auto switchOp = scf::IndexSwitchOp::create(rewriter, loc, types, index,
                                               cases, cases.size());
    OpBuilder::InsertionGuard guard(rewriter);
    for (auto [entry, region] : llvm::zip(cases, switchOp.getCaseRegions())) {
      rewriter.setInsertionPointToStart(&region.emplaceBlock());
      scf::YieldOp::create(rewriter, loc, leaf(table[entry]));
    }
    rewriter.setInsertionPointToStart(
        &switchOp.getDefaultRegion().emplaceBlock());
    scf::YieldOp::create(rewriter, loc, leaf(common));
    return Values(switchOp.getResults());
  }

  // The known values of every listed leaf, if there are any.
  std::optional<SmallVector<SmallVector<APInt>>>
  knownLeaves(ArrayRef<unsigned> leaves) {
    if (!knownLeaf)
      return std::nullopt;
    llvm::SmallDenseMap<unsigned, SmallVector<APInt>> known;
    SmallVector<SmallVector<APInt>> rows;
    for (unsigned target : leaves) {
      auto it = known.find(target);
      if (it == known.end()) {
        std::optional<SmallVector<APInt>> values = knownLeaf(target);
        if (!values)
          return std::nullopt;
        it = known.try_emplace(target, std::move(*values)).first;
      }
      rows.push_back(it->second);
    }
    return rows;
  }

  // Loads row index of a constant table holding rows, whose columns have the
  // given integer or index types. Each row is contiguous, and its entries have
  // the narrowest width holding every value. Columns that never vary are
  // constants instead.
  Values loadRow(ArrayRef<SmallVector<APInt>> rows, Value index,
                 TypeRange columns) {
    SmallVector<unsigned> varying;
    unsigned bits = 8;
    for (unsigned column = 0; column < columns.size(); ++column) {
      if (llvm::all_of(rows, [&](const auto &row) {
            return row[column] == rows.front()[column];
          }))
        continue;
      varying.push_back(column);
      for (const auto &row : rows)
        bits = std::max(bits, row[column].getSignificantBits());
    }
    bits = std::max(8u, unsigned(llvm::PowerOf2Ceil(bits)));
    auto element = rewriter.getIntegerType(bits);
    Value base;
    if (!varying.empty()) {
      auto tableType = MemRefType::get(
          {int64_t(rows.size()), int64_t(varying.size())}, element);
      SmallVector<APInt> contents;
      for (const auto &row : rows)
        for (unsigned column : varying)
          contents.push_back(row[column].sextOrTrunc(bits));
      auto global = tableGlobal(
          tableType,
          DenseElementsAttr::get(
              RankedTensorType::get(tableType.getShape(), element), contents));
      base = memref::GetGlobalOp::create(rewriter, loc, tableType,
                                         global.getSymName());
    }
    Values values;
    for (auto [column, type] : llvm::enumerate(columns)) {
      auto integer =
          type.isIndex() ? rewriter.getI64Type() : cast<IntegerType>(type);
      unsigned width = integer.getWidth();
      const auto *position = llvm::find(varying, column);
      Value value;
      if (position == varying.end()) {
        value = integerConstant(rewriter, loc, integer,
                                rows.front()[column].sextOrTrunc(width));
      } else {
        auto offset = arith::ConstantIndexOp::create(
            rewriter, loc, position - varying.begin());
        value = memref::LoadOp::create(rewriter, loc, base,
                                       ValueRange{index, offset});
        if (width > bits)
          value = arith::ExtSIOp::create(rewriter, loc, integer, value);
        else if (width < bits)
          value = arith::TruncIOp::create(rewriter, loc, integer, value);
      }
      if (type.isIndex())
        value = arith::IndexCastUIOp::create(rewriter, loc, type, value);
      values.push_back(value);
    }
    return values;
  }

  // Selects among keys over one background leaf by hashing the key to the
  // only slot that may hold it, and comparing the key stored there. The hash
  // multiplies by an odd constant, searched to be injective on the keys, and
  // keeps the high bits of the product.
  std::optional<Values> perfectHash(Value key, ArrayRef<Piece> pieces) {
    SmallVector<unsigned> leaves{pieces.front().leaf};
    SmallVector<APInt> keys;
    for (size_t i = 1; i < pieces.size(); i += 2) {
      keys.push_back(pieces[i].start);
      leaves.push_back(pieces[i].leaf);
    }
    std::optional<SmallVector<SmallVector<APInt>>> known = knownLeaves(leaves);
    if (!known)
      return std::nullopt;
    unsigned width = key.getType().getIntOrFloatBitWidth();
    unsigned least = llvm::Log2_64_Ceil(keys.size());
    std::optional<std::pair<APInt, unsigned>> found;
    uint64_t state = 0x9e3779b97f4a7c15;
    for (unsigned slotBits = least;
         !found && slotBits <= least + 2 && slotBits < width; ++slotBits) {
      for (unsigned attempt = 0; !found && attempt < maxHashAttempts;
           ++attempt) {
        // SplitMix64 makes the search deterministic.
        state += 0x9e3779b97f4a7c15;
        uint64_t mixed = state;
        mixed = (mixed ^ (mixed >> 30)) * 0xbf58476d1ce4e5b9;
        mixed = (mixed ^ (mixed >> 27)) * 0x94d049bb133111eb;
        mixed ^= mixed >> 31;
        APInt multiplier = APInt(64, mixed | 1).zextOrTrunc(width);
        llvm::SmallDenseSet<uint64_t> slots;
        if (llvm::all_of(keys, [&](const APInt &k) {
              return slots.insert(slotOf(k, multiplier, slotBits)).second;
            }))
          found = {multiplier, slotBits};
      }
    }
    if (!found)
      return std::nullopt;
    auto [multiplier, slotBits] = *found;
    Type type = key.getType();
    // Slots hold their key and the leaf values. An empty slot is no key's
    // slot, so any key stored there never matches.
    SmallVector<SmallVector<APInt>> rows(size_t(1) << slotBits,
                                         SmallVector<APInt>{});
    for (auto [i, k] : llvm::enumerate(keys)) {
      SmallVector<APInt> &row = rows[slotOf(k, multiplier, slotBits)];
      row.push_back(k);
      llvm::append_range(row, (*known)[i + 1]);
    }
    for (SmallVector<APInt> &row : rows) {
      if (!row.empty())
        continue;
      row.push_back(keys.front());
      llvm::append_range(row, known->front());
    }
    Value product = arith::MulIOp::create(
        rewriter, loc, key, integerConstant(rewriter, loc, type, multiplier));
    Value slot = arith::ShRUIOp::create(
        rewriter, loc, product,
        integerConstant(rewriter, loc, type, APInt(width, width - slotBits)));
    Value index = arith::IndexCastUIOp::create(rewriter, loc,
                                               rewriter.getIndexType(), slot);
    SmallVector<Type> columns{type};
    llvm::append_range(columns, types);
    Values loaded = loadRow(rows, index, columns);
    Value hit = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                      loaded.front(), key);
    // A single row has only constant columns.
    Values background = loadRow({known->front()}, index, types);
    Values values;
    for (auto [found, otherwise] :
         llvm::zip_equal(ArrayRef(loaded).drop_front(), background))
      values.push_back(
          arith::SelectOp::create(rewriter, loc, hit, found, otherwise));
    return values;
  }

  static uint64_t slotOf(const APInt &key, const APInt &multiplier,
                         unsigned slotBits) {
    unsigned width = multiplier.getBitWidth();
    return (key.zextOrTrunc(width) * multiplier)
        .lshr(width - slotBits)
        .getZExtValue();
  }

  // A private constant global holding contents. Its name is a BLAKE3 digest
  // of the table, so equal tables share one global.
  memref::GlobalOp tableGlobal(MemRefType type, DenseElementsAttr contents) {
    llvm::BLAKE3 hasher;
    for (int64_t extent : type.getShape())
      hasher.update(llvm::ArrayRef(reinterpret_cast<const uint8_t *>(&extent),
                                   sizeof(extent)));
    uint32_t bits = type.getElementTypeBitWidth();
    hasher.update(
        llvm::ArrayRef(reinterpret_cast<const uint8_t *>(&bits), sizeof(bits)));
    ArrayRef<char> data = contents.getRawData();
    hasher.update(llvm::ArrayRef(reinterpret_cast<const uint8_t *>(data.data()),
                                 data.size()));
    std::array<uint8_t, 16> digest = hasher.final<16>();
    std::string name = "__rcgmp_table_" + llvm::toHex(digest, true);
    auto module = rewriter.getInsertionBlock()
                      ->getParentOp()
                      ->getParentOfType<ModuleOp>();
    if (auto existing = module.lookupSymbol<memref::GlobalOp>(name))
      return existing;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    return memref::GlobalOp::create(rewriter, loc, name,
                                    rewriter.getStringAttr("private"), type,
                                    contents, /*constant=*/true,
                                    /*alignment=*/IntegerAttr());
  }

  PatternRewriter &rewriter;
  Location loc;
  SmallVector<Type, 2> types;
  KnownLeaf knownLeaf;
};

// Unsigned pieces whose leaves select an arm or build a subtree. Pieces
// selecting one arm share its leaf, and consecutive ones merge.
class Leaves {
public:
  void addArm(const APInt &start, unsigned arm,
              function_ref<Values(unsigned)> armLeaf) {
    auto [it, inserted] = armLeaves.try_emplace(arm, builders.size());
    if (inserted)
      builders.push_back([armLeaf, arm] { return armLeaf(arm); });
    if (pieces.empty() || pieces.back().leaf != it->second)
      pieces.push_back({start, it->second});
  }

  void addSubtree(const APInt &start, std::function<Values()> build) {
    pieces.push_back({start, unsigned(builders.size())});
    builders.push_back(std::move(build));
  }

  Values dispatch(Selector &selector, Value key) {
    return selector.dispatch(key, /*isSigned=*/false, pieces,
                             [&](unsigned leaf) { return builders[leaf](); });
  }

private:
  SmallVector<Piece> pieces;
  SmallVector<std::function<Values()>> builders;
  llvm::SmallDenseMap<unsigned, unsigned> armLeaves;
};

// Selects among the heap integers of one sign by their magnitude m, ordering
// them by their limb count and then their limbs from the most significant. A
// run of limbs shared by every boundary in a subrange is compared at once,
// which leaves a switch only where boundaries diverge.
class HeapSelector {
public:
  struct MagnitudePiece {
    APInt start;
    unsigned arm;
  };

  HeapSelector(PatternRewriter &rewriter, Location loc, Selector &selector,
               Value scrutinee, unsigned limbBits, bool negative,
               ArrayRef<MagnitudePiece> pieces,
               function_ref<Values(unsigned)> armLeaf)
      : rewriter(rewriter), loc(loc), selector(selector), scrutinee(scrutinee),
        limbBits(limbBits), negative(negative), pieces(pieces),
        armLeaf(armLeaf), word(rewriter.getIntegerType(limbBits)) {}

  // Selects by count, the number of limbs.
  Values select(Value count) {
    if (pieces.size() == 1)
      return armLeaf(pieces.front().arm);
    unsigned width = pieces.front().start.getBitWidth();
    SmallVector<unsigned> counts;
    for (const MagnitudePiece &piece : pieces.drop_front())
      counts.push_back(limbCount(piece.start));
    llvm::sort(counts);
    counts.erase(llvm::unique(counts), counts.end());

    // Counts holding boundaries strictly inside their range get a subtree;
    // every other count selects the arm at its least magnitude.
    Leaves leaves;
    unsigned next = 1;
    for (unsigned count : counts) {
      if (count > next)
        leaves.addArm(APInt(limbBits, next), armAt(lowest(next, width)),
                      armLeaf);
      APInt low = lowest(count, width);
      APInt high = APInt::getLowBitsSet(width, count * limbBits);
      if (hasBoundaryAbove(low, high))
        leaves.addSubtree(APInt(limbBits, count), [this, count, low, high] {
          return limbs(count, low, high, 0);
        });
      else
        leaves.addArm(APInt(limbBits, count), armAt(low), armLeaf);
      next = count + 1;
    }
    leaves.addArm(APInt(limbBits, next), armAt(lowest(next, width)), armLeaf);
    return leaves.dispatch(selector, count);
  }

private:
  unsigned limbCount(const APInt &magnitude) const {
    return llvm::divideCeil(magnitude.getActiveBits(), limbBits);
  }

  // The least magnitude with count limbs that can occur.
  APInt lowest(unsigned count, unsigned width) const {
    APInt low = APInt::getOneBitSet(width, (count - 1) * limbBits);
    return llvm::APIntOps::umax(low, pieces.front().start);
  }

  unsigned armAt(const APInt &magnitude) const {
    const MagnitudePiece *after =
        llvm::upper_bound(pieces, magnitude, [](const APInt &m, const auto &p) {
          return m.ult(p.start);
        });
    return std::prev(after)->arm;
  }

  // Whether a piece starts in (low, high].
  bool hasBoundaryAbove(const APInt &low, const APInt &high) const {
    return llvm::any_of(pieces.drop_front(), [&](const MagnitudePiece &p) {
      return p.start.ugt(low) && p.start.ule(high);
    });
  }

  APInt limbOf(const APInt &magnitude, unsigned position) const {
    return magnitude.extractBits(limbBits, position * limbBits);
  }

  Value loadLimb(unsigned position) {
    return bridge::MpzLimbOp::create(rewriter, loc, word, scrutinee, position);
  }

  // Selects among magnitudes in [low, high], which have count limbs and share
  // their limbs above position count - 1 - depth.
  Values limbs(unsigned count, APInt low, APInt high, unsigned depth) {
    SmallVector<APInt> boundaries;
    for (const MagnitudePiece &piece : pieces.drop_front())
      if (piece.start.ugt(low) && piece.start.ule(high))
        boundaries.push_back(piece.start);
    if (boundaries.empty())
      return armLeaf(armAt(low));
    unsigned position = count - 1 - depth;

    // The run of limbs, from position downwards, that every boundary shares.
    unsigned run = 0;
    while (run <= position && llvm::all_of(boundaries, [&](const APInt &b) {
             return limbOf(b, position - run) ==
                    limbOf(boundaries.front(), position - run);
           }))
      ++run;
    if (run > maxInlineLimbs)
      return compareInRuntime(low, boundaries);
    if (run)
      return compareRun(count, low, high, depth, run, boundaries);

    // Switch on the limb at position.
    unsigned width = low.getBitWidth();
    unsigned shift = position * limbBits;
    APInt prefix = low.lshr(shift + limbBits).shl(shift + limbBits);
    auto sliceLow = [&](const APInt &limb) {
      return llvm::APIntOps::umax(low, prefix | limb.zext(width).shl(shift));
    };
    SmallVector<APInt> limbValues;
    for (const APInt &b : boundaries)
      limbValues.push_back(limbOf(b, position));
    llvm::sort(limbValues,
               [](const APInt &a, const APInt &b) { return a.ult(b); });
    limbValues.erase(llvm::unique(limbValues), limbValues.end());

    Leaves leaves;
    auto constant = [&](const APInt &limb) {
      leaves.addArm(limb, armAt(sliceLow(limb)), armLeaf);
    };
    APInt first = limbOf(low, position), last = limbOf(high, position);
    std::optional<APInt> next = first;
    for (const APInt &limb : limbValues) {
      if (next && next->ult(limb))
        constant(*next);
      APInt subLow = sliceLow(limb);
      APInt subHigh =
          llvm::APIntOps::umin(high, (prefix | limb.zext(width).shl(shift)) |
                                         APInt::getLowBitsSet(width, shift));
      if (llvm::any_of(boundaries, [&](const APInt &b) {
            return b.ugt(subLow) && b.ule(subHigh);
          }))
        leaves.addSubtree(limb, [this, count, subLow, subHigh, depth] {
          return limbs(count, subLow, subHigh, depth + 1);
        });
      else
        constant(limb);
      next = limb == last ? std::nullopt : std::optional<APInt>(limb + 1);
    }
    if (next)
      constant(*next);
    return leaves.dispatch(selector, loadLimb(position));
  }

  // Every boundary shares the next run limbs. A magnitude whose run differs
  // lies below or above all of them.
  Values compareRun(unsigned count, const APInt &low, const APInt &high,
                    unsigned depth, unsigned run, ArrayRef<APInt> boundaries) {
    unsigned position = count - 1 - depth;
    unsigned width = low.getBitWidth();
    unsigned shift = (position + 1 - run) * limbBits;
    APInt base = boundaries.front().lshr(shift).shl(shift);
    APInt innerLow = llvm::APIntOps::umax(low, base);
    APInt innerHigh =
        llvm::APIntOps::umin(high, base | APInt::getLowBitsSet(width, shift));
    unsigned below = armAt(low);
    unsigned above = armAt(boundaries.back());
    auto inner = [&] { return limbs(count, innerLow, innerHigh, depth + run); };
    SmallVector<Value> equal, less;
    for (unsigned i = 0; i < run; ++i) {
      Value limb = loadLimb(position - i);
      Value expected = integerConstant(
          rewriter, loc, word, limbOf(boundaries.front(), position - i));
      equal.push_back(arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, limb, expected));
      if (below != above)
        less.push_back(arith::CmpIOp::create(
            rewriter, loc, arith::CmpIPredicate::ult, limb, expected));
    }
    Value same = equal.front();
    for (Value e : ArrayRef(equal).drop_front())
      same = arith::AndIOp::create(rewriter, loc, same, e);
    if (below == above)
      return selector.branch(same, inner, [&] { return armLeaf(below); });
    // The first differing limb, from the most significant, orders the run.
    Value isLess = less.back();
    for (size_t i = run - 1; i-- > 0;)
      isLess =
          arith::SelectOp::create(rewriter, loc, equal[i], isLess, less[i]);
    return selector.branch(same, inner, [&] {
      return selector.branch(
          isLess, [&] { return armLeaf(below); },
          [&] { return armLeaf(above); });
    });
  }

  // Binary search over the boundaries, comparing the whole integer.
  Values compareInRuntime(const APInt &low, ArrayRef<APInt> boundaries) {
    SmallVector<APInt> starts{low};
    llvm::append_range(starts, boundaries);
    return search(starts);
  }

  Values search(ArrayRef<APInt> starts) {
    if (starts.size() == 1)
      return armLeaf(armAt(starts.front()));
    size_t middle = starts.size() / 2;
    // For a negative integer n with magnitude m, m < b exactly when n > -b.
    APInt bound = starts[middle].zext(starts[middle].getBitWidth() + 1);
    if (negative)
      bound.negate();
    auto constant =
        ConstantOp::create(rewriter, loc, scrutinee.getType(), bound);
    auto below = CmpIOp::create(
        rewriter, loc, negative ? CmpIPredicate::sgt : CmpIPredicate::slt,
        scrutinee, constant);
    return selector.branch(
        below, [&] { return search(starts.take_front(middle)); },
        [&] { return search(starts.drop_front(middle)); });
  }

  PatternRewriter &rewriter;
  Location loc;
  Selector &selector;
  Value scrutinee;
  unsigned limbBits;
  bool negative;
  ArrayRef<MagnitudePiece> pieces;
  function_ref<Values(unsigned)> armLeaf;
  IntegerType word;
};

// What the region of an arm receives.
enum class Binding {
  // Nothing: the match releases the scrutinee.
  none,
  // The scrutinee itself.
  scrutinee,
  // The scrutinee minus a nonnegative threshold, which keeps an immediate
  // tagged after subtracting four times the threshold from its bits.
  remainder,
  // The scrutinee minus a negative threshold, which may leave the immediate
  // range and is subtracted in the region.
  negativeRemainder,
};

// Lowers one match. Both the immediate and the heap path select an arm and
// yield its dispatch index, the value its region receives, which is the
// scrutinee or its remainder, and, if any arm only yields values known on
// selection, the results of such an arm. Those arms then need no region and
// share one dispatch case, so that the selection can become a table. Every
// other arm has its own case, and its region is emitted once.
class MatchLowerer {
public:
  MatchLowerer(MatchOp op, PatternRewriter &rewriter)
      : op(op), rewriter(rewriter), loc(op.getLoc()),
        type(cast<MpzType>(op.getScrutinee().getType())),
        width(DataLayout::closest(op).getTypeSizeInBits(type).getFixedValue()),
        word(rewriter.getIntegerType(width)) {
    for (auto [arm, body] :
         llvm::zip_equal(op.getArms().getAsRange<ArmAttr>(), op.getBodies())) {
      APInt threshold;
      Binding binding = Binding::none;
      if (body.getNumArguments()) {
        binding = Binding::scrutinee;
        if (arm.getKind() == ArmKind::atLeast) {
          threshold = cast<IntegerAttr>(arm.getValues()[0]).getValue();
          binding = threshold.isNegative() ? Binding::negativeRemainder
                                           : Binding::remainder;
        }
      }
      bindings.push_back(binding);
      thresholds.push_back(threshold);
      if (isConstantArm(body)) {
        hasConstantArms = true;
        dispatchIndex.push_back(~0u);
      } else {
        dispatchIndex.push_back(dispatched++);
      }
    }
    for (unsigned &index : dispatchIndex)
      if (index == ~0u)
        index = dispatched;
  }

  void lower() {
    SmallVector<MatchInterval> intervals = selectedIntervals();
    Value scrutinee = op.getScrutinee();
    Value bits = bridge::MpzToBitsOp::create(rewriter, loc, word, scrutinee);
    auto one = arith::ConstantIntOp::create(rewriter, loc, 1, width);
    auto tag = arith::AndIOp::create(rewriter, loc, bits, one);
    auto immediate = arith::CmpIOp::create(rewriter, loc,
                                           arith::CmpIPredicate::eq, tag, one);

    SmallVector<Type> types{rewriter.getIndexType(), type};
    if (hasConstantArms)
      llvm::append_range(types, op.getResultTypes());
    Selector selector(rewriter, loc, types);
    Values selection = selector.branch(
        immediate, [&] { return selectImmediate(intervals, bits); },
        [&] { return selectHeap(selector, intervals); });
    Value bound = selection[1];
    ValueRange constants = ValueRange(selection).drop_front(2);
    if (!dispatched) {
      rewriter.replaceOp(op, constants);
      return;
    }

    unsigned cases = hasConstantArms ? dispatched : dispatched - 1;
    auto dispatch = scf::IndexSwitchOp::create(
        rewriter, loc, op.getResultTypes(), selection[0],
        llvm::to_vector(llvm::seq<int64_t>(0, cases)), cases);
    if (hasConstantArms) {
      rewriter.setInsertionPointToStart(
          &dispatch.getDefaultRegion().emplaceBlock());
      scf::YieldOp::create(rewriter, loc, constants);
    }
    for (auto [i, body] : llvm::enumerate(op.getBodies())) {
      if (dispatchIndex[i] == dispatched)
        continue;
      Region &target = dispatchIndex[i] < cases
                           ? dispatch.getCaseRegions()[dispatchIndex[i]]
                           : dispatch.getDefaultRegion();
      Block &block = target.emplaceBlock();
      rewriter.setInsertionPointToStart(&block);
      SmallVector<Value> arguments;
      Block &source = body.front();
      // The scrutinee reaches an arm without an argument unchanged, and is
      // released only when it is on the heap.
      if (bindings[i] == Binding::none)
        ReleaseOp::create(rewriter, loc, bound);
      if (source.getNumArguments()) {
        Value argument = bound;
        if (bindings[i] == Binding::negativeRemainder)
          argument = subtract(bound, i);
        arguments.push_back(argument);
      }
      auto yield = cast<YieldOp>(source.getTerminator());
      rewriter.setInsertionPoint(yield);
      scf::YieldOp::create(rewriter, loc, yield.getResults());
      rewriter.eraseOp(yield);
      rewriter.mergeBlocks(&source, &block, arguments);
    }
    rewriter.replaceOp(op, dispatch.getResults());
  }

private:
  // An arm whose region only materializes constants and yields them, or
  // values defined outside the match, so that its results are known once it
  // is selected.
  static bool isConstantArm(Region &body) {
    Block &block = body.front();
    return !block.getNumArguments() &&
           llvm::all_of(block.without_terminator(), [](Operation &nested) {
             return nested.hasTrait<OpTrait::ConstantLike>() ||
                    isa<ConstantOp>(nested);
           });
  }

  bool isConstant(unsigned arm) const {
    return dispatchIndex[arm] == dispatched;
  }

  // The selected intervals, where integers that no arm selects take the arm of
  // a neighboring interval, since their behavior is undefined.
  SmallVector<MatchInterval> selectedIntervals() {
    SmallVector<MatchInterval> intervals = op.getIntervals();
    for (size_t i = 1; i < intervals.size(); ++i)
      if (!intervals[i].arm)
        intervals[i].arm = intervals[i - 1].arm;
    for (size_t i = intervals.size() - 1; i-- > 0;)
      if (!intervals[i].arm)
        intervals[i].arm = intervals[i + 1].arm;
    SmallVector<MatchInterval> merged;
    for (MatchInterval &interval : intervals)
      if (merged.empty() || merged.back().arm != interval.arm)
        merged.push_back(std::move(interval));
    return merged;
  }

  Value subtract(Value value, unsigned arm) {
    return SubIOp::create(
        rewriter, loc, value,
        ConstantOp::create(rewriter, loc, type, thresholds[arm]));
  }

  // What the immediate tagged bits lose to become the region argument.
  APInt offset(unsigned arm) const {
    if (bindings[arm] != Binding::remainder)
      return APInt(width, 0);
    return thresholds[arm].zextOrTrunc(width).shl(2);
  }

  // The results of a constant arm, or poison for any other.
  Values results(unsigned arm) {
    if (!hasConstantArms)
      return {};
    Values results;
    if (!isConstant(arm)) {
      for (Type result : op.getResultTypes())
        results.push_back(ub::PoisonOp::create(rewriter, loc, result));
      return results;
    }
    Block &block = op.getBodies()[arm].front();
    IRMapping mapping;
    for (Operation &nested : block.without_terminator())
      rewriter.clone(nested, mapping);
    for (Value result : block.getTerminator()->getOperands())
      results.push_back(mapping.lookupOrDefault(result));
    return results;
  }

  // The leaf values of an immediate arm, if they are all integer constants:
  // the dispatch index, the offset, and the results, which are poison, here
  // zero, for an arm that is not constant.
  std::optional<SmallVector<APInt>> knownImmediateLeaf(unsigned arm) {
    SmallVector<APInt> values{APInt(64, dispatchIndex[arm]), offset(arm)};
    if (!hasConstantArms)
      return values;
    Operation *yield = op.getBodies()[arm].front().getTerminator();
    for (auto [i, result] : llvm::enumerate(op.getResultTypes())) {
      if (!result.isIntOrIndex())
        return std::nullopt;
      unsigned bits = result.isIndex() ? 64 : result.getIntOrFloatBitWidth();
      APInt value = APInt::getZero(bits);
      if (isConstant(arm)) {
        IntegerAttr attr;
        if (!matchPattern(yield->getOperand(i), m_Constant(&attr)))
          return std::nullopt;
        value = attr.getValue();
      }
      values.push_back(value);
    }
    return values;
  }

  // Selects among immediates by their payload. Each leaf yields the dispatch
  // index, the amount to subtract from the tagged bits to obtain the region
  // argument, and the results of a constant arm.
  Values selectImmediate(ArrayRef<MatchInterval> intervals, Value bits) {
    APInt minimum = APInt::getSignedMinValue(width).ashr(2);
    APInt maximum = APInt::getSignedMaxValue(width).ashr(2);
    SmallVector<Piece> pieces;
    for (auto [start, arm] : clip(intervals, minimum, maximum))
      pieces.push_back({start.trunc(width), arm});
    Value payload;
    if (pieces.size() > 1) {
      auto two = arith::ConstantIntOp::create(rewriter, loc, 2, width);
      payload = arith::ShRSIOp::create(rewriter, loc, bits, two);
    }
    SmallVector<Type> types{rewriter.getIndexType(), word};
    if (hasConstantArms)
      llvm::append_range(types, op.getResultTypes());
    Selector selector(rewriter, loc, types,
                      [this](unsigned arm) { return knownImmediateLeaf(arm); });
    Values selected = selector.dispatch(
        payload, /*isSigned=*/true, pieces, [&](unsigned arm) -> Values {
          Values leaf{
              arith::ConstantIndexOp::create(rewriter, loc, dispatchIndex[arm]),
              integerConstant(rewriter, loc, word, offset(arm))};
          llvm::append_range(leaf, results(arm));
          return leaf;
        });
    Value bound = arith::SubIOp::create(rewriter, loc, bits, selected[1]);
    Values selection{selected[0],
                     bridge::MpzFromBitsOp::create(rewriter, loc, type, bound)};
    llvm::append_range(selection, ArrayRef(selected).drop_front(2));
    return selection;
  }

  // The leaf of a heap arm, which subtracts a nonnegative threshold here and
  // releases the scrutinee for a constant arm.
  Values heapLeaf(unsigned arm) {
    Value scrutinee = op.getScrutinee();
    Value bound = scrutinee;
    if (bindings[arm] == Binding::remainder)
      bound = subtract(scrutinee, arm);
    else if (isConstant(arm))
      ReleaseOp::create(rewriter, loc, scrutinee);
    Values leaf{
        arith::ConstantIndexOp::create(rewriter, loc, dispatchIndex[arm]),
        bound};
    llvm::append_range(leaf, results(arm));
    return leaf;
  }

  // Selects among heap integers, which lie outside the immediate range, by
  // sign and magnitude.
  Values selectHeap(Selector &selector, ArrayRef<MatchInterval> intervals) {
    using MagnitudePiece = HeapSelector::MagnitudePiece;
    // Room for the magnitude of every boundary and two more limbs.
    unsigned bits = width;
    for (const MatchInterval &interval : intervals)
      bits = std::max(bits, interval.start.getSignificantBits());
    unsigned magnitudeBits = (llvm::divideCeil(bits, width) + 2) * width;
    APInt maximum = APInt::getSignedMaxValue(width).ashr(2).sext(magnitudeBits);

    // Positive heap integers are above the maximum immediate. Negative ones are
    // below its negation minus one, and an interval [s, e) of them holds the
    // magnitudes (-e, -s], so their pieces reverse.
    SmallVector<MagnitudePiece> positive, negative;
    for (auto [start, arm] : clip(intervals, maximum + 1, std::nullopt))
      positive.push_back({start.sextOrTrunc(magnitudeBits), arm});
    SmallVector<std::pair<APInt, unsigned>> below =
        clip(intervals, std::nullopt, -maximum - 2);
    for (size_t i = below.size(); i-- > 0;)
      negative.push_back(
          {i + 1 < below.size()
               ? -below[i + 1].first.sextOrTrunc(magnitudeBits) + 1
               : maximum + 2,
           below[i].second});

    auto armLeaf = [this](unsigned arm) { return heapLeaf(arm); };
    if (positive.size() == 1 && negative.size() == 1 &&
        positive.front().arm == negative.front().arm)
      return armLeaf(positive.front().arm);
    Value scrutinee = op.getScrutinee();
    Value size = bridge::MpzSizeOp::create(rewriter, loc, word, scrutinee);
    auto zero = arith::ConstantIntOp::create(rewriter, loc, 0, width);
    auto isPositive = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::sgt, size, zero);
    return selector.branch(
        isPositive,
        [&] {
          HeapSelector heap(rewriter, loc, selector, scrutinee, width,
                            /*negative=*/false, positive, armLeaf);
          return heap.select(size);
        },
        [&] {
          if (negative.size() == 1)
            return armLeaf(negative.front().arm);
          HeapSelector heap(rewriter, loc, selector, scrutinee, width,
                            /*negative=*/true, negative, armLeaf);
          return heap.select(arith::SubIOp::create(rewriter, loc, zero, size));
        });
  }

  MatchOp op;
  PatternRewriter &rewriter;
  Location loc;
  MpzType type;
  unsigned width;
  IntegerType word;
  SmallVector<Binding> bindings;
  SmallVector<APInt> thresholds;
  // The dispatch case of each arm. Constant arms share the last one.
  SmallVector<unsigned> dispatchIndex;
  unsigned dispatched = 0;
  bool hasConstantArms = false;
};

struct MatchLowering : OpRewritePattern<MatchOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatchOp op,
                                PatternRewriter &rewriter) const override {
    MatchLowerer(op, rewriter).lower();
    return success();
  }
};

} // namespace

void populateMatchLoweringPatterns(RewritePatternSet &patterns) {
  patterns.add<MatchLowering>(patterns.getContext());
}

} // namespace mlir::rcgmp
