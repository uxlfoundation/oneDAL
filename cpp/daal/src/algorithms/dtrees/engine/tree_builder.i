/* file: tree_builder.i */
/*******************************************************************************
* Copyright contributors to the oneDAL project
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Implementation of the tree builder: the histogram phase (phase_hist.i) on nodes of at least
//  denseThreshold * maxBins records of a binned fit, the exact phase on the others.
//  Included only by tree_builder_fpt_cpu.cpp.
//--
*/

#ifndef __DTREES_ENGINE_TREE_BUILDER_I__
#define __DTREES_ENGINE_TREE_BUILDER_I__

#include <cmath>
#include <limits>
#include <type_traits>

#include "src/algorithms/dtrees/engine/cost_function.h"
#include "src/algorithms/dtrees/engine/executor.h"
#include "src/algorithms/dtrees/engine/node_rng.h"
#include "src/algorithms/dtrees/engine/phase_hist.i"

#include "src/algorithms/dtrees/engine/prefetch.h"
#include "src/algorithms/dtrees/engine/tree_builder.h"
#include "src/algorithms/dtrees/engine/tree_soa.h"
#include "src/algorithms/service_sort.h"
#include "src/services/service_environment.h"

namespace daal
{
namespace algorithms
{
namespace dtrees
{
namespace engine
{
namespace internal
{
using services::internal::TArray;

// Nodes with at least this many records sort with parallel_sort (design 4.7: root-level sorts of
// defaultDense on large n). Initial value; the exact-phase benchmark tunes it.
constexpr size_t parallelSortRecords = size_t(1) << 17;

static_assert(sizeof(TreeNode) == 56, "TreeNode layout");

namespace exact
{
/// A split of one node on one feature.
struct Candidate
{
    bool valid        = false;
    bool categorical  = false;
    uint32_t feature  = 0;
    uint32_t nLeft    = 0; // records going left
    double threshold  = 0; // ordered: rows with x <= threshold go left; categorical: rows with x == threshold
    double proxy      = 0; // cost function proxy, larger is better (cost_function.h)
    double wLeft      = 0;
    bool byBin        = false; // histogram split: rows route by bin, which agrees with threshold on the training rows
    uint32_t splitBin = 0;     // ordered: bins <= splitBin and the missing bin go left; categorical: bin == splitBin
};

/// Scratch for one node. A node that calls no parallel algorithm uses its thread's scratch, as the thread
/// never runs another node while it holds it. A node that sorts or builds histograms in parallel can have
/// its thread run other nodes while it waits, so it gets scratch of its own. The per-row arrays are
/// allocated only for the exact phase.
template <typename algorithmFPType, CpuType cpu>
struct NodeScratch
{
    DAAL_NEW_DELETE();
    LargeArray<algorithmFPType, cpu> x;
    LargeArray<int32_t, cpu> order;
    LargeArray<double, cpu> w;
    LargeArray<double, cpu> y;
    LargeArray<uint32_t, cpu> label;
    LargeArray<double, cpu> classWeights;
    LargeArray<uint32_t, cpu> permutation;
    LargeArray<uint32_t, cpu> lanes; // histogram phase: the features of a pass
    size_t capacity = 0;
    bool baseReady  = false; // set only once every base array is allocated, so a failed call is retried

    bool ensureBase(size_t nClasses, size_t nFeatures)
    {
        if (baseReady) return true;
        baseReady = permutation.reset(2 * nFeatures) && lanes.reset(nFeatures) && (!nClasses || classWeights.reset(2 * nClasses));
        return baseReady;
    }

    bool ensureRows(size_t n, size_t nClasses)
    {
        if (n <= capacity) return true;
        size_t c = capacity ? capacity : 1;
        while (c < n) c *= 2;
        if (!x.reset(c) || !order.reset(c) || !w.reset(c)) return false;
        if (nClasses ? !label.reset(c) : !y.reset(c)) return false;
        capacity = c;
        return true;
    }
};

/// sklearn's threshold between two adjacent distinct values: their midpoint, or the lower value when
/// the midpoint rounds to the upper one.
inline double midpoint(double a, double b)
{
    const double t = a / 2 + b / 2;
    return (t >= b || t <= a) ? a : t;
}
} // namespace exact

/// The largest algorithmFPType value at or below threshold t. Float prediction compares x > float(t), so a t
/// rounded up onto a training value would send that value right; no float lies between the result and t, so
/// it routes every float as t does.
template <typename algorithmFPType>
inline double representableThreshold(double t)
{
    if constexpr (std::is_same<algorithmFPType, float>::value)
    {
        const float f = float(t);
        return double(f) > t ? double(std::nextafter(f, -std::numeric_limits<float>::infinity())) : double(f);
    }
    return t;
}

inline size_t bitWords(size_t n)
{
    return (n + 63) / 64;
}
inline bool testBit(const uint64_t * bits, size_t i)
{
    return (bits[i >> 6] >> (i & 63)) & 1;
}
inline void setBit(uint64_t * bits, size_t i)
{
    bits[i >> 6] |= uint64_t(1) << (i & 63);
}

/// What a node passes to its children (design 4.6): its constant features, and the histograms of the
/// child's first draws that the parent's fused partition built.
template <typename Record, CpuType cpu>
struct NodeCarry
{
    DAAL_NEW_DELETE();
    NodeCarry * prev = nullptr;
    NodeCarry * next = nullptr;
    // Two lists of one bit per feature: constant by bins, for the histogram phase; constant by value, for
    // the exact phase, as rows in one bin can still differ in value.
    LargeArray<uint64_t, cpu> bits;
    LargeArray<uint32_t, cpu> lanes;
    size_t nLanes = 0;
    hist::Planes<Record, cpu> planes;
};

/// The live carries of one tree. Node tasks of the tree create and destroy them concurrently, hence the
/// lock; the list lets endTree free the carries of nodes a failed build never reached.
template <typename Record, CpuType cpu>
class CarryList
{
public:
    using Carry = NodeCarry<Record, cpu>;

    CarryList()                              = default;
    CarryList(const CarryList &)             = delete;
    CarryList & operator=(const CarryList &) = delete;
    ~CarryList() { clear(); }

    Carry * create()
    {
        Carry * c = new Carry();
        if (!c) return nullptr;
        AutoLock lock(_mutex);
        c->next = _head;
        if (_head) _head->prev = c;
        _head = c;
        return c;
    }

    void destroy(Carry * c)
    {
        if (!c) return;
        {
            AutoLock lock(_mutex);
            if (c->prev) c->prev->next = c->next;
            if (c->next) c->next->prev = c->prev;
            if (_head == c) _head = c->next;
        }
        delete c;
    }

    void clear()
    {
        AutoLock lock(_mutex);
        while (_head)
        {
            Carry * c = _head;
            _head     = c->next;
            delete c;
        }
    }

private:
    Mutex _mutex;
    Carry * _head = nullptr;
};

/// Builds trees for the executor (executor.h): the exact phase at every node.
template <typename algorithmFPType, typename Record, CpuType cpu>
class TreeBuilder
{
    static constexpr bool isClassification =
        std::is_same<Record, ClsCountRecord>::value || std::is_same<Record, ClsWeightRecord<algorithmFPType> >::value;
    using Cost      = typename std::conditional<isClassification, GiniCost<cpu>, MseCost<cpu> >::type;
    using Scratch   = exact::NodeScratch<algorithmFPType, cpu>;
    using Candidate = exact::Candidate;

public:
    struct WorkItem
    {
        uint32_t node      = 0;
        uint32_t start     = 0;
        uint32_t n         = 0;
        uint32_t depth     = 0;
        uint64_t key       = 0;
        double improvement = 0;
        double weight      = 0; // of the node's records, set by evaluate
        Candidate split;
        NodeCarry<Record, cpu> * carry = nullptr; // owned; nullptr: the root lists and no histograms
    };

    struct TreeContext
    {
        ResponseSet<Record, cpu> records;
        NodePool<cpu> nodes;
        CounterRng<cpu> rng;
        CarryList<Record, cpu> carries;
        int numaNode = 0;
    };

    TreeBuilder(const TreeData<algorithmFPType, cpu> & data, const TreeParams & params, TreeInputs<Record, cpu> & inputs)
        : _data(data),
          _params(params),
          _inputs(inputs),
          _k(params.featuresPerNode && params.featuresPerNode < data.nFeatures ? params.featuresPerNode : data.nFeatures),
          _words(bitWords(data.nFeatures)),
          _scratch([]() -> Scratch * { return new Scratch(); }),
          _prefetchRows(defaultPrefetchDistance<cpu>()),
          _l2Bytes(services::internal::getL2CacheSize())
    {}

    ~TreeBuilder()
    {
        _scratch.reduce([](Scratch * s) { delete s; });
    }

    /// The root lists: features with at most one data bin; by value only if that bin is one value.
    services::Status init()
    {
        DAAL_CHECK_MALLOC(_rootBits.reset(2 * _words));
        for (size_t i = 0; i < 2 * _words; ++i) _rootBits[i] = 0;
        if (!_data.bins) return services::Status();
        for (size_t f = 0; f < _data.nFeatures; ++f)
        {
            if (_data.bins->nBins(f) > 1) continue;
            setBit(_rootBits.get(), f);
            if (!_data.bins->nBins(f) || _data.bins->exactBins(f)) setBit(_rootBits.get() + _words, f);
        }
        return services::Status();
    }

    services::Status initContext(TreeContext & ctx, int numaNode)
    {
        ctx.numaNode = numaNode;
        ctx.rng.useStreams(&_streams);
        return ctx.records.allocate(_data.nRows);
    }

    services::Status beginTree(TreeContext & ctx, size_t iTree, WorkItem & root)
    {
        uint64_t treeSeed  = 0;
        services::Status s = _inputs.prepareTree(iTree, ctx.numaNode, ctx.records, treeSeed);
        DAAL_CHECK_STATUS_VAR(s);
        if (_params.counterDraws)
        {
            s = ctx.rng.init(treeSeed);
            DAAL_CHECK_STATUS_VAR(s);
        }
        // A tree of n records has at most n leaves and n - 1 splits. A tree whose in-bag rows all have zero
        // weight has no records and is one leaf.
        const size_t n = ctx.records.size();
        s              = ctx.nodes.reset(n ? 2 * n : 1);
        DAAL_CHECK_STATUS_VAR(s);
        root     = WorkItem();
        root.n   = uint32_t(n);
        root.key = treeSeed;
        if (useHist(n))
        {
            root.carry = newCarry(ctx, nullptr);
            DAAL_CHECK_MALLOC(root.carry);
        }
        return s;
    }

    services::Status evaluate(TreeContext & ctx, WorkItem & item, bool & splittable)
    {
        splittable           = false;
        const bool histNode  = useHist(item.n);
        const size_t nChunks = histNode ? hist::chunkCount(item.n, _params.chunkRows, _params.maxChunks) : 1;
        Scratch own;
        Scratch * scratch = item.n < parallelSortRecords && nChunks == 1 ? _scratch.local() : &own;
        DAAL_CHECK_MALLOC(scratch && scratch->ensureBase(_params.nClasses, _data.nFeatures));
        const Record * records = ctx.records.buffer(item.depth) + item.start;

        // Node statistics from its records.
        Cost cost         = makeCost(*scratch);
        uint64_t nSamples = 0;
        cost.startParent();
        for (size_t i = 0; i < item.n; ++i)
        {
            nSamples += recordCount(records[i]); // 1 per record in the weighted layouts, which keep no count
            addParent(cost, records[i]);
        }
        cost.finishParent();
        // Purity is decided exactly: impurity computed from sums can round to a small positive value
        // for a pure node, which would then split on zero gain.
        const bool pure = isPure(cost, records, item.n);

        TreeNode & node = ctx.nodes[item.node];
        node            = TreeNode();
        node.feature    = -1;
        node.impurity   = pure ? 0 : cost.impurity();
        node.nRecords   = item.n;
        node.nSamples   = uint32_t(nSamples);
        node.depth      = item.depth;
        node.start      = item.start;

        item.weight = cost.weight();
        if (leafBySize(item.n, item.weight, item.depth) || pure || node.impurity < _params.impurityThreshold) return services::Status();

        Candidate best;
        services::Status s;
        if (_params.counterDraws)
        {
            // An exact phase after an all-constant histogram phase restarts the feature draws (design 4.6).
            CounterNodeDraws<cpu> features(ctx.rng, item.key);
            CounterNodeDraws<cpu> restart(ctx.rng, item.key);
            CounterNodeDraws<cpu> splits(ctx.rng, item.key, nodeThresholdWords);
            best = searchNode(ctx, item.carry, records, item.n, histNode, nChunks, *scratch, cost, features, restart, splits, s);
            s |= features.status();
            s |= restart.status();
            s |= splits.status();
        }
        else
        {
            void * stream = _inputs.nodeStream();
            DAAL_CHECK(stream, services::ErrorNullPtr);
            StreamNodeDraws<cpu> draws(stream);
            best = searchNode(ctx, item.carry, records, item.n, histNode, nChunks, *scratch, cost, draws, draws, draws, s);
            s |= draws.status();
        }
        if (item.carry) item.carry->planes.release();
        DAAL_CHECK_STATUS_VAR(s);

        const double decrease = best.proxy - cost.parentProxy();
        if (!best.valid || decrease < _params.minWeightedDecrease) return services::Status();
        item.split       = best;
        item.improvement = decrease;
        splittable       = true;
        return services::Status();
    }

    services::Status commit(TreeContext & ctx, WorkItem & item, WorkItem & left, WorkItem & right)
    {
        const Candidate & split = item.split;
        const uint32_t pair     = ctx.nodes.allocatePair();
        DAAL_CHECK_MALLOC(pair);
        left        = WorkItem();
        left.node   = pair;
        left.start  = item.start;
        left.n      = split.nLeft;
        left.depth  = item.depth + 1;
        left.key    = childNodeKey<cpu>(item.key, 0);
        right       = WorkItem();
        right.node  = pair + 1;
        right.start = item.start + split.nLeft;
        right.n     = item.n - split.nLeft;
        right.depth = item.depth + 1;
        right.key   = childNodeKey<cpu>(item.key, 1);

        // Children that may split in the histogram phase get the node's constant lists, and with fusion their
        // first draws, histogrammed by the partition pass. Exact-phase nodes, most of a tree, get no carry and
        // use the root lists: a carry per node costs two locked list operations (CarryList).
        services::Status s;
        WorkItem * children[2]  = { &left, &right };
        const double weights[2] = { split.wLeft, item.weight - split.wLeft };
        const bool fuse         = split.byBin && _params.fusePartition && _params.counterDraws;
        hist::Target<Record, cpu> targets[2];
        for (size_t c = 0; c < 2 && s; ++c)
        {
            WorkItem & child = *children[c];
            if (leafBySize(child.n, weights[c], child.depth) || !useHist(child.n)) continue;
            child.carry = newCarry(ctx, item.carry);
            if (!child.carry)
            {
                s = services::Status(services::ErrorMemoryAllocationFailed);
                break;
            }
            if (!fuse) continue;
            s = drawChildLanes(ctx, child);
            if (s && child.carry->nLanes) targets[c] = { child.carry->lanes.get(), child.carry->nLanes, &child.carry->planes };
        }
        ctx.carries.destroy(item.carry);
        item.carry = nullptr;
        DAAL_CHECK_STATUS_VAR(s);

        const Record * in    = ctx.records.buffer(item.depth) + item.start;
        Record * out         = ctx.records.buffer(item.depth + 1) + item.start;
        const size_t nChunks = hist::chunkCount(item.n, _params.chunkRows, _params.maxChunks);
        if (!split.byBin)
            s = partition(in, out, item.n, split.nLeft, nChunks, ValueRoute { _data.x->data(ctx.numaNode), _data.nFeatures, split });
        else if (_data.bins->binBytes() == 1)
            s = partitionByBin<uint8_t>(ctx, in, out, item.n, nChunks, split, targets);
        else
            s = partitionByBin<uint16_t>(ctx, in, out, item.n, nChunks, split, targets);
        DAAL_CHECK_STATUS_VAR(s);

        TreeNode & node       = ctx.nodes[item.node];
        node.feature          = int32_t(split.feature);
        node.threshold        = representableThreshold<algorithmFPType>(split.threshold);
        node.categorical      = split.categorical;
        node.weightedDecrease = item.improvement;
        node.left             = pair;
        // Missing values follow the side with more weight (design 4.16).
        node.defaultLeft = split.wLeft >= item.weight - split.wLeft;
        return services::Status();
    }

    services::Status makeLeaf(TreeContext & ctx, WorkItem & item)
    {
        ctx.carries.destroy(item.carry);
        item.carry      = nullptr;
        TreeNode & node = ctx.nodes[item.node];
        node.feature    = -1;
        node.start      = item.start;
        return services::Status();
    }

    services::Status endTree(TreeContext & ctx, size_t iTree, const services::Status & built)
    {
        ctx.carries.clear();
        if (!built) return built;
        EmittedTree tree;
        services::Status s = emit(ctx, tree);
        DAAL_CHECK_STATUS_VAR(s);
        return _inputs.emitTree(iTree, tree);
    }

    size_t rows(const WorkItem & item) const { return item.n; }
    double improvement(const WorkItem & item) const { return item.improvement; }
    uint64_t nodeKey(const WorkItem & item) const { return item.key; }

private:
    // Ordered: x <= threshold, NaN included, as prediction routes x > threshold right. Categorical: x == threshold.
    struct ValueRoute
    {
        const algorithmFPType * x;
        size_t p;
        const Candidate & split;

        const void * address(uint32_t row) const { return x + size_t(row) * p + split.feature; }
        bool left(uint32_t row) const
        {
            const algorithmFPType v = x[size_t(row) * p + split.feature];
            if (split.categorical) return double(v) == split.threshold;
            return daal::algorithms::internal::isNaN<algorithmFPType, cpu>(v) || double(v) <= split.threshold;
        }
    };

    // The same sides as ValueRoute on the training rows, from the bin matrix.
    template <typename BinT>
    struct BinRoute
    {
        BinRoute(const BinMatrix<cpu> & bins, int numaNode, const Candidate & split)
            : bins(bins),
              numaNode(numaNode),
              feature(split.feature),
              splitBin(split.splitBin),
              missing(bins.missingBin(split.feature)),
              categorical(split.categorical)
        {}

        const BinMatrix<cpu> & bins;
        int numaNode;
        size_t feature;
        size_t splitBin;
        size_t missing;
        bool categorical;

        const void * address(uint32_t row) const { return bins.template row<BinT>(numaNode, row) + feature; }
        bool left(uint32_t row) const { return leftBin(bins.template row<BinT>(numaNode, row)[feature]); }
        bool leftBin(size_t b) const { return categorical ? b == splitBin : b <= splitBin || b == missing; }
    };

    /// Stable partition of n records into out, the nLeft left ones first. Several chunks count their left
    /// records in parallel, then each scatters from its offsets in chunk order.
    template <typename Route>
    services::Status partition(const Record * in, Record * out, size_t n, size_t nLeft, size_t nChunks, const Route & route) const
    {
        const size_t distance = prefetchDistance(n);
        auto scatter          = [&](size_t begin, size_t end, size_t iLeft, size_t iRight) {
            prefetchedLoop<cpu>(
                end - begin, distance, [&](size_t i) { DAAL_PREFETCH_READ_T0(route.address(in[begin + i].row)); },
                [&](size_t i) {
                    const Record & r = in[begin + i];
                    if (route.left(r.row))
                        out[iLeft++] = r;
                    else
                        out[iRight++] = r;
                });
            return iLeft;
        };
        if (nChunks <= 1)
        {
            if (scatter(0, n, 0, nLeft) != nLeft) return services::Status(services::ErrorIncorrectInternalFunctionParameter);
            return services::Status();
        }
        TArray<size_t, cpu> lefts(nChunks + 1);
        DAAL_CHECK_MALLOC(lefts.get());
        daal::threader_for(nChunks, 1, [&](size_t c) {
            size_t begin, end, count = 0;
            hist::chunkRange(n, nChunks, c, begin, end);
            prefetchedLoop<cpu>(
                end - begin, distance, [&](size_t i) { DAAL_PREFETCH_READ_T0(route.address(in[begin + i].row)); },
                [&](size_t i) { count += route.left(in[begin + i].row); });
            lefts[c + 1] = count;
        });
        lefts[0] = 0;
        for (size_t c = 0; c < nChunks; ++c) lefts[c + 1] += lefts[c];
        if (lefts[nChunks] != nLeft) return services::Status(services::ErrorIncorrectInternalFunctionParameter);
        daal::threader_for(nChunks, 1, [&](size_t c) {
            size_t begin, end;
            hist::chunkRange(n, nChunks, c, begin, end);
            scatter(begin, end, lefts[c], nLeft + begin - lefts[c]);
        });
        return services::Status();
    }

    bool useHist(size_t n) const { return _data.bins && n >= _params.denseThreshold * _params.maxBins; }
    bool isCategorical(size_t f) const { return _data.categorical && _data.categorical[f]; }

    /// The size a node's minimum sizes apply to: its records, or with sizesInDraws its bootstrap draws, which on
    /// the count layouts are its weight. The weighted layouts fold the counts into the weight and use records.
    size_t sizeOf(size_t nRecords, double w) const
    {
        constexpr bool countLayout = std::is_same<Record, ClsCountRecord>::value || std::is_same<Record, RegCountRecord<algorithmFPType> >::value;
        return _params.sizesInDraws && countLayout ? size_t(w + 0.5) : nRecords;
    }

    /// A node too small, light or deep to split.
    bool leafBySize(size_t n, double w, size_t depth) const
    {
        const size_t size = sizeOf(n, w);
        return size < _params.minSplitRecords || size < 2 * _params.minLeafRecords || w < 2 * _params.minLeafWeight
               || (_params.maxDepth && depth >= _params.maxDepth);
    }

    /// A carry with the parent's lists, or the root lists for the root and with carryConstants off.
    NodeCarry<Record, cpu> * newCarry(TreeContext & ctx, const NodeCarry<Record, cpu> * parent) const
    {
        NodeCarry<Record, cpu> * c = ctx.carries.create();
        if (!c) return nullptr;
        if (!c->bits.reset(2 * _words))
        {
            ctx.carries.destroy(c);
            return nullptr;
        }
        const uint64_t * from = parent && _params.carryConstants ? parent->bits.get() : _rootBits.get();
        for (size_t i = 0; i < 2 * _words; ++i) c->bits[i] = from[i];
        return c;
    }

    const uint64_t * constantBits(const NodeCarry<Record, cpu> * c) const { return c ? c->bits.get() : _rootBits.get(); }
    const uint64_t * exactConstantBits(const NodeCarry<Record, cpu> * c) const { return constantBits(c) + _words; }

    /// Adds f to the node's lists: constant by bins, and by value if found by value or if its bins are values.
    void markConstant(NodeCarry<Record, cpu> * c, uint32_t f, bool byValue) const
    {
        if (!c || !_params.carryConstants) return;
        setBit(c->bits.get(), f);
        if (byValue || (_data.bins && _data.bins->exactBins(f))) setBit(c->bits.get() + _words, f);
    }

    /// The next up to want draws of the permutation that are not in the constant list.
    template <typename Draws>
    static size_t drawLanes(FeaturePermutation<cpu> & permutation, Draws & draws, const uint64_t * constant, size_t want, uint32_t * lanes)
    {
        size_t n = 0;
        while (n < want && !permutation.exhausted())
        {
            const uint32_t f = permutation.next(draws);
            if (!testBit(constant, f)) lanes[n++] = f;
        }
        return n;
    }

    /// A child's first histogram pass draws, as its histSearch will make them.
    services::Status drawChildLanes(TreeContext & ctx, WorkItem & child) const
    {
        // No parallel call until the draws are made, so the thread's scratch is free (NodeScratch).
        Scratch * scratch = _scratch.local();
        DAAL_CHECK_MALLOC(scratch && scratch->ensureBase(_params.nClasses, _data.nFeatures));
        DAAL_CHECK_MALLOC(child.carry->lanes.reset(_k));
        CounterNodeDraws<cpu> draws(ctx.rng, child.key);
        FeaturePermutation<cpu> permutation(uint32_t(_data.nFeatures), _k, scratch->permutation.get());
        child.carry->nLanes = drawLanes(permutation, draws, child.carry->bits.get(), _k, child.carry->lanes.get());
        return draws.status();
    }

    template <typename BinT>
    services::Status partitionByBin(TreeContext & ctx, const Record * in, Record * out, size_t n, size_t nChunks, const Candidate & split,
                                    const hist::Target<Record, cpu> (&targets)[2]) const
    {
        const BinRoute<BinT> route(*_data.bins, ctx.numaNode, split);
        if (!targets[0].planes && !targets[1].planes) return partition(in, out, n, split.nLeft, nChunks, route);
        const size_t f = split.feature;
        return hist::fusedPartition<Record, BinT>(
            in, out, n, split.nLeft, *_data.bins, ctx.numaNode, split.feature, [&](const BinT * b) { return route.leftBin(b[f]); }, targets,
            _params.nClasses, nChunks, prefetchDistance(n));
    }

    // No software prefetch for nodes whose rows already fit half of L2 after the parent's pass (design 4.5).
    size_t prefetchDistance(size_t n) const { return n * cacheLineBytes * 2 < _l2Bytes ? 0 : _prefetchRows; }

    static void setResponse(Scratch & s, size_t i, const Record & r)
    {
        if constexpr (isClassification)
            s.label[i] = recordClass(r);
        else
            s.y[i] = recordResponse(r);
    }

    Cost makeCost(Scratch & s) const
    {
        if constexpr (isClassification)
            return Cost(_params.nClasses, s.classWeights.get());
        else
            return Cost();
    }
    static void addParent(Cost & cost, const Record & r)
    {
        if constexpr (isClassification)
            cost.addParent(recordClass(r), recordWeight(r));
        else
            cost.addParent(recordResponse(r), recordWeight(r));
    }
    void addLeft(Cost & cost, const Scratch & s, size_t i) const
    {
        if constexpr (isClassification)
            cost.addLeft(s.label[i], s.w[i]);
        else
            cost.addLeft(s.y[i], s.w[i]);
    }

    /// One class holds all the weight, or every response is equal.
    bool isPure(const Cost & cost, const Record * r, size_t n) const
    {
        if constexpr (isClassification)
        {
            size_t nonEmpty = 0;
            for (size_t c = 0; c < _params.nClasses; ++c) nonEmpty += cost.classWeights()[c] > 0;
            return nonEmpty <= 1;
        }
        else
        {
            for (size_t i = 1; i < n; ++i)
                if (recordResponse(r[i]) != recordResponse(r[0])) return false;
            return true;
        }
    }

    bool admissible(size_t nLeft, size_t n, double wLeft, double w) const
    {
        const double wRight = w - wLeft;
        const size_t left   = sizeOf(nLeft, wLeft);
        const size_t total  = sizeOf(n, w);
        const size_t right  = total > left ? total - left : 0;
        return nLeft > 0 && nLeft < n && left >= _params.minLeafRecords && right >= _params.minLeafRecords && wLeft >= _params.minLeafWeight
               && wRight >= _params.minLeafWeight && wLeft > 0 && wRight > 0;
    }

    /// The histogram phase on hist nodes, the exact phase on the others and on hist nodes whose drawn
    /// features are all constant by bins (design 4.6, 4.7).
    template <typename FeatureDraws, typename SplitDraws>
    Candidate searchNode(TreeContext & ctx, NodeCarry<Record, cpu> * carry, const Record * records, size_t n, bool histNode, size_t nChunks,
                         Scratch & scratch, Cost & cost, FeatureDraws & features, FeatureDraws & restart, SplitDraws & splits,
                         services::Status & status) const
    {
        if (histNode)
        {
            bool allConstant  = false;
            const Candidate c = histSearch(ctx, carry, records, n, nChunks, scratch, cost, features, splits, allConstant, status);
            if (!status || !allConstant) return c;
        }
        if (!scratch.ensureRows(n, _params.nClasses))
        {
            status = services::Status(services::ErrorMemoryAllocationFailed);
            return Candidate();
        }
        for (size_t i = 0; i < n; ++i)
        {
            scratch.w[i] = recordWeight(records[i]);
            setResponse(scratch, i, records[i]);
        }
        return exactSearch(ctx, carry, records, n, scratch, cost, histNode ? restart : features, splits, status);
    }

    /// The best split over the node's first k non-constant features in draw order (design 4.6, 4.13).
    /// Features on the by-value constant list are skipped as the search would find them constant.
    template <typename FeatureDraws, typename SplitDraws>
    Candidate exactSearch(TreeContext & ctx, NodeCarry<Record, cpu> * carry, const Record * records, size_t n, Scratch & scratch, Cost & cost,
                          FeatureDraws & features, SplitDraws & splits, services::Status & status) const
    {
        FeaturePermutation<cpu> permutation(uint32_t(_data.nFeatures), _k, scratch.permutation.get());
        const uint64_t * known = exactConstantBits(carry);
        Candidate best;
        size_t visited = 0;
        while (visited < _k && !permutation.exhausted())
        {
            const uint32_t f = permutation.next(features);
            if (testBit(known, f)) continue;
            bool constant     = false;
            const Candidate c = splitOnFeature(ctx, records, n, scratch, cost, splits, f, constant, status);
            if (!status) return best;
            if (constant) // constant features do not count toward k, as in sklearn
            {
                markConstant(carry, f, true);
                continue;
            }
            ++visited;
            if (c.valid && (!best.valid || c.proxy > best.proxy)) best = c;
        }
        return best;
    }

    /// The histogram phase's search: each pass draws the features still needed for k non-constant ones
    /// and histograms them together; lanes constant by bins, and features on the node's list, do not
    /// count toward k (design 4.6). The first pass uses the histograms the parent built, if any.
    template <typename FeatureDraws, typename SplitDraws>
    Candidate histSearch(TreeContext & ctx, NodeCarry<Record, cpu> * carry, const Record * records, size_t n, size_t nChunks, Scratch & scratch,
                         Cost & cost, FeatureDraws & features, SplitDraws & splits, bool & allConstant, services::Status & status) const
    {
        FeaturePermutation<cpu> permutation(uint32_t(_data.nFeatures), _k, scratch.permutation.get());
        const uint64_t * known = constantBits(carry);
        uint32_t * lanes       = scratch.lanes.get();
        hist::Planes<Record, cpu> own;
        bool carried = carry && carry->nLanes;
        Candidate best;
        size_t visited = 0;
        allConstant    = true;
        while (visited < _k)
        {
            const size_t nLanes = drawLanes(permutation, features, known, _k - visited, lanes);
            if (!nLanes) break;
            const hist::Planes<Record, cpu> * planes = &own;
            if (carried)
            {
                // The parent drew these lanes from the same key and lists.
                bool same = carry->nLanes == nLanes;
                for (size_t j = 0; j < nLanes && same; ++j) same = carry->lanes[j] == lanes[j];
                if (!same)
                {
                    status = services::Status(services::ErrorIncorrectInternalFunctionParameter);
                    return best;
                }
                planes  = &carry->planes;
                carried = false;
            }
            else
            {
                status = hist::build(records, n, *_data.bins, ctx.numaNode, lanes, nLanes, _params.nClasses, nChunks, prefetchDistance(n), own);
                if (!status) return best;
            }
            for (size_t j = 0; j < nLanes; ++j)
            {
                bool constant     = false;
                const Candidate c = scanLane(*planes, j, lanes[j], n, cost, splits, constant);
                if (constant)
                {
                    markConstant(carry, lanes[j], false);
                    continue;
                }
                allConstant = false;
                ++visited;
                if (c.valid && (!best.valid || c.proxy > best.proxy)) best = c;
            }
        }
        return best;
    }

    /// The split of one lane from its histogram (design 4.4). The missing bin goes left on ordered features
    /// and right on categorical ones, as NaN in the exact phase. A lane with at most one nonempty data bin
    /// is constant.
    template <typename SplitDraws>
    Candidate scanLane(const hist::Planes<Record, cpu> & h, size_t lane, uint32_t f, size_t n, Cost & cost, SplitDraws & draws, bool & constant) const
    {
        const BinMatrix<cpu> & bins = *_data.bins;
        const size_t nData          = bins.nBins(f);
        auto records                = [&](size_t b) { return size_t(h.records[h.at(b, lane)]); };
        size_t first = nData, last = 0, nonEmpty = 0;
        for (size_t b = 0; b < nData; ++b)
        {
            if (!records(b)) continue;
            first = first == nData ? b : first;
            last  = b;
            ++nonEmpty;
        }

        Candidate best;
        best.feature     = f;
        best.categorical = isCategorical(f);
        best.byBin       = true;
        if (nonEmpty <= 1)
        {
            constant = true;
            return best;
        }
        const double * border = bins.borders(f);
        const double w        = cost.weight();
        auto addBin           = [&](size_t b) {
            if constexpr (isClassification)
            {
                for (size_t c = 0; c < _params.nClasses; ++c)
                {
                    const double a = double(h.classStat(c, b, lane));
                    if (a > 0) cost.addLeft(uint32_t(c), a);
                }
            }
            else
            {
                cost.addLeftSums(double(h.stat[h.at(b, lane)]), h.sumY[h.at(b, lane)]);
            }
        };
        auto consider = [&](size_t nLeft, size_t b) {
            if (!admissible(nLeft, n, cost.leftWeight(), w)) return;
            const double proxy = cost.proxy();
            if (best.valid && !(proxy > best.proxy)) return;
            best.valid     = true;
            best.threshold = border[b];
            best.splitBin  = uint32_t(b);
            best.nLeft     = uint32_t(nLeft);
            best.wLeft     = cost.leftWeight();
            best.proxy     = proxy;
        };

        if (best.categorical)
        {
            // One category against the rest; the random splitter takes the u-th category present.
            const size_t chosen = _params.randomSplitter ? size_t(draws.below(uint32_t(nonEmpty))) : nonEmpty;
            size_t run          = 0;
            for (size_t b = first; b <= last; ++b)
            {
                if (!records(b)) continue;
                if (chosen == nonEmpty || run == chosen)
                {
                    cost.resetLeft();
                    addBin(b);
                    consider(records(b), b);
                }
                ++run;
            }
            return best;
        }

        cost.resetLeft();
        size_t nLeft = 0;
        if (bins.hasMissing(f))
        {
            addBin(bins.missingBin(f));
            nLeft += records(bins.missingBin(f));
        }
        if (_params.randomSplitter)
        {
            // Uniform in [upper border of the first nonempty bin, that of the last); the split goes after the
            // last bin whose border is <= the draw, so both sides are nonempty (defect 1).
            const double lo = border[first];
            const double hi = border[last];
            double t        = lo + draws.unit() * (hi - lo);
            if (t >= hi) t = lo;
            size_t a = first, z = last; // border[a] <= t < border[z]
            while (z - a > 1)
            {
                const size_t mid = a + (z - a) / 2;
                if (border[mid] <= t)
                    a = mid;
                else
                    z = mid;
            }
            for (size_t b = first; b <= a; ++b)
            {
                if (!records(b)) continue;
                addBin(b);
                nLeft += records(b);
            }
            consider(nLeft, a);
            return best;
        }
        for (size_t b = first; b < last; ++b)
        {
            if (!records(b)) continue;
            addBin(b);
            nLeft += records(b);
            consider(nLeft, b);
        }
        return best;
    }

    template <typename Draws>
    Candidate splitOnFeature(TreeContext & ctx, const Record * records, size_t n, Scratch & scratch, Cost & cost, Draws & draws, uint32_t f,
                             bool & constant, services::Status & status) const
    {
        const bool categorical    = isCategorical(f);
        const algorithmFPType * x = _data.x->data(ctx.numaNode);
        const size_t p            = _data.nFeatures;
        algorithmFPType * values  = scratch.x.get();
        prefetchedLoop<cpu>(
            n, prefetchDistance(n), [&](size_t i) { DAAL_PREFETCH_READ_T0(x + size_t(records[i].row) * p + f); },
            [&](size_t i) { values[i] = x[size_t(records[i].row) * p + f]; });

        Candidate best;
        best.feature     = f;
        best.categorical = categorical;
        const double w   = cost.weight();

        if (_params.randomSplitter && !categorical)
        {
            // One threshold uniform in [min, max) of the node's values; no sort (design 4.4).
            bool any  = false;
            double lo = 0, hi = 0;
            for (size_t i = 0; i < n; ++i)
            {
                if (daal::algorithms::internal::isNaN<algorithmFPType, cpu>(values[i])) continue;
                const double v = double(values[i]);
                lo             = any && lo <= v ? lo : v;
                hi             = any && hi >= v ? hi : v;
                any            = true;
            }
            if (!any || lo == hi)
            {
                constant = true;
                return best;
            }
            double t = lo + draws.unit() * (hi - lo);
            if (t >= hi) t = lo;
            cost.resetLeft();
            size_t nLeft = 0;
            for (size_t i = 0; i < n; ++i)
            {
                if (daal::algorithms::internal::isNaN<algorithmFPType, cpu>(values[i]) || double(values[i]) <= t)
                {
                    addLeft(cost, scratch, i);
                    ++nLeft;
                }
            }
            if (admissible(nLeft, n, cost.leftWeight(), w))
            {
                best.valid     = true;
                best.threshold = t;
                best.nLeft     = uint32_t(nLeft);
                best.wLeft     = cost.leftWeight();
                best.proxy     = cost.proxy();
            }
            return best;
        }

        int32_t * order = scratch.order.get();
        for (size_t i = 0; i < n; ++i) order[i] = int32_t(i);
        // NaN keys move to the end; m values are sorted at the front.
        const size_t m = sortFeature(values, order, n);
        if (m == ~size_t(0))
        {
            status = services::Status(services::ErrorMemoryAllocationFailed);
            return best;
        }
        if (!m || values[0] == values[m - 1])
        {
            constant = true;
            return best;
        }

        if (categorical)
        {
            // One category against the rest; NaN rows are in the rest. The random splitter takes one
            // category uniformly among those present (design 4.4, defects 6 and 7).
            size_t nRuns = 1;
            for (size_t i = 1; i < m; ++i) nRuns += values[i] != values[i - 1];
            const size_t chosen = _params.randomSplitter ? size_t(draws.below(uint32_t(nRuns))) : nRuns;
            size_t run          = 0;
            for (size_t a = 0; a < m; ++run)
            {
                size_t b = a + 1;
                while (b < m && values[b] == values[a]) ++b;
                if (chosen == nRuns || run == chosen)
                {
                    cost.resetLeft();
                    for (size_t i = a; i < b; ++i) addLeft(cost, scratch, size_t(order[i]));
                    if (admissible(b - a, n, cost.leftWeight(), w))
                    {
                        const double proxy = cost.proxy();
                        if (!best.valid || proxy > best.proxy)
                        {
                            best.valid     = true;
                            best.threshold = double(values[a]);
                            best.nLeft     = uint32_t(b - a);
                            best.wLeft     = cost.leftWeight();
                            best.proxy     = proxy;
                        }
                    }
                }
                a = b;
            }
            return best;
        }

        // Every boundary between adjacent distinct values; NaN rows start on the left.
        cost.resetLeft();
        for (size_t i = m; i < n; ++i) addLeft(cost, scratch, size_t(order[i]));
        size_t nLeft = n - m;
        for (size_t i = 0; i + 1 < m; ++i)
        {
            addLeft(cost, scratch, size_t(order[i]));
            ++nLeft;
            if (values[i] == values[i + 1] || !admissible(nLeft, n, cost.leftWeight(), w)) continue;
            const double proxy = cost.proxy();
            if (best.valid && !(proxy > best.proxy)) continue;
            best.valid     = true;
            best.threshold = exact::midpoint(double(values[i]), double(values[i + 1]));
            best.nLeft     = uint32_t(nLeft);
            best.wLeft     = cost.leftWeight();
            best.proxy     = proxy;
        }
        return best;
    }

    /// Sorts values with NaN moved to the end, permuting order alike; returns the non-NaN count, or
    /// ~0 if the parallel path cannot allocate.
    static size_t sortFeature(algorithmFPType * values, int32_t * order, size_t n)
    {
        if (n < parallelSortRecords) return daal::algorithms::internal::qSort<algorithmFPType, int32_t, cpu>(n, values, order, true);
        // parallel_sort's comparison is undefined for NaN, so NaN keys leave first.
        const size_t m = daal::algorithms::internal::moveNaNsToEnd<algorithmFPType, int32_t, cpu>(n, values, order);
        if (!m) return 0;
        TArray<daal::IdxValType<algorithmFPType>, cpu> pairs(m);
        if (!pairs.get()) return ~size_t(0);
        daal::threader_for(m, tableBlockRows, [&](size_t i) {
            pairs[i].value = values[i];
            pairs[i].index = size_t(order[i]);
        });
        daal::parallel_sort(pairs.get(), pairs.get() + m);
        daal::threader_for(m, tableBlockRows, [&](size_t i) {
            values[i] = pairs[i].value;
            order[i]  = int32_t(pairs[i].index);
        });
        return m;
    }

    /// Writes the tree in the DecisionTreeTable layout of nodeToTable: root in row 0, each split's
    /// children in the next free pair of rows, left subtree first. Leaf values come from the records.
    services::Status emit(TreeContext & ctx, EmittedTree & tree) const
    {
        const size_t nNodes   = ctx.nodes.size();
        const size_t nClasses = _params.nClasses;
        tree.nodes            = dtrees::internal::DecisionTreeTablePtr(new dtrees::internal::DecisionTreeTable(nNodes));
        tree.impurity         = data_management::HomogenNumericTable<double>::create(1, nNodes, data_management::NumericTable::doAllocate);
        tree.nodeSamples      = data_management::HomogenNumericTable<int>::create(1, nNodes, data_management::NumericTable::doAllocate);
        tree.probabilities    = nClasses ? data_management::HomogenNumericTable<double>::create(data_management::DictionaryIface::equal, nNodes,
                                                                                                nClasses, data_management::NumericTable::doAllocate) :
                                           data_management::HomogenNumericTable<double>::create(0, 0, data_management::NumericTable::doAllocate);
        DAAL_CHECK_MALLOC(tree.nodes && tree.nodes->getArray() && tree.impurity && tree.nodeSamples && tree.probabilities);
        DAAL_CHECK_MALLOC(tree.featureImportance.reset(_data.nFeatures));
        for (size_t f = 0; f < _data.nFeatures; ++f) tree.featureImportance[f] = 0;

        dtrees::internal::DecisionTreeNode * rows = static_cast<dtrees::internal::DecisionTreeNode *>(tree.nodes->getArray());
        double * impurity                         = tree.impurity->getArray();
        int * samples                             = tree.nodeSamples->getArray();
        double * prob                             = nClasses ? tree.probabilities->getArray() : nullptr;

        // Depth-first from the root: (pool node, table row). The stack holds at most one pending right
        // child per level plus the current node.
        TArray<uint32_t, cpu> stackNode(nNodes + 1);
        TArray<uint32_t, cpu> stackRow(nNodes + 1);
        TArray<double, cpu> classWeights(nClasses ? nClasses : 1);
        DAAL_CHECK_MALLOC(stackNode.get() && stackRow.get() && classWeights.get());
        size_t top     = 0;
        size_t nextRow = 1;
        stackNode[top] = 0;
        stackRow[top]  = 0;
        ++top;
        while (top)
        {
            --top;
            const TreeNode & node                    = ctx.nodes[stackNode[top]];
            const size_t r                           = stackRow[top];
            dtrees::internal::DecisionTreeNode & row = rows[r];
            impurity[r]                              = node.impurity;
            samples[r]                               = int(node.nSamples);
            row.cover                                = 0;
            if (node.feature >= 0)
            {
                row.featureIndex           = node.feature;
                row.featureValueOrResponse = node.threshold;
                row.defaultLeft            = node.defaultLeft;
                row.leftIndexOrClass       = nextRow;
                if (prob)
                    for (size_t c = 0; c < nClasses; ++c) prob[r * nClasses + c] = 0;
                tree.featureImportance[size_t(node.feature)] += node.weightedDecrease;
                stackNode[top] = node.left + 1;
                stackRow[top]  = uint32_t(nextRow + 1);
                ++top;
                stackNode[top] = node.left;
                stackRow[top]  = uint32_t(nextRow);
                ++top;
                nextRow += 2;
                continue;
            }
            row.featureIndex           = -1;
            row.defaultLeft            = 0;
            const Record * leafRecords = ctx.records.buffer(node.depth) + node.start;
            if (nClasses)
            {
                writeClassLeaf(leafRecords, node.nRecords, classWeights.get(), row, prob + r * nClasses);
            }
            else
            {
                row.leftIndexOrClass       = 0;
                row.featureValueOrResponse = regressionLeaf(leafRecords, node.nRecords);
            }
        }
        DAAL_ASSERT(nextRow == nNodes);
        return services::Status();
    }

    void writeClassLeaf(const Record * r, size_t n, double * cw, dtrees::internal::DecisionTreeNode & row, double * prob) const
    {
        if constexpr (isClassification)
        {
            const size_t nClasses = _params.nClasses;
            for (size_t c = 0; c < nClasses; ++c) cw[c] = 0;
            double total = 0;
            for (size_t i = 0; i < n; ++i)
            {
                cw[recordClass(r[i])] += recordWeight(r[i]);
                total += recordWeight(r[i]);
            }
            size_t best = 0;
            for (size_t c = 1; c < nClasses; ++c) best = cw[c] > cw[best] ? c : best;
            row.leftIndexOrClass       = best;
            row.featureValueOrResponse = 0;
            // Probabilities are the weighted class shares; they sum to 1 with user weights too. A leaf without
            // records (a tree of zero-weight rows) is uniform.
            for (size_t c = 0; c < nClasses; ++c) prob[c] = total > 0 ? cw[c] / total : 1. / double(nClasses);
        }
    }

    double regressionLeaf(const Record * r, size_t n) const
    {
        if constexpr (!isClassification)
        {
            // Records hold the responses less ResponseSource::responseShift() (see computeShift); a leaf without
            // records predicts the mean response.
            double w = 0, s = 0;
            for (size_t i = 0; i < n; ++i)
            {
                w += recordWeight(r[i]);
                s += recordWeight(r[i]) * recordResponse(r[i]);
            }
            return w > 0 ? _data.responses->responseShift() + s / w : _data.responses->responseMean();
        }
        return 0;
    }

    const TreeData<algorithmFPType, cpu> & _data;
    const TreeParams _params;
    TreeInputs<Record, cpu> & _inputs;
    const size_t _k;
    const size_t _words; // per constant list
    LargeArray<uint64_t, cpu> _rootBits;
    CounterStreams<cpu> _streams;
    mutable daal::tls<Scratch *> _scratch;
    const size_t _prefetchRows;
    const size_t _l2Bytes;
};

template <typename algorithmFPType, typename Record, CpuType cpu>
services::Status trainTrees(const TreeData<algorithmFPType, cpu> & data, const TreeParams & params, size_t nTrees, TreeInputs<Record, cpu> & inputs)
{
    DAAL_CHECK(data.x && data.responses && data.nFeatures && data.nRows, services::ErrorNullInput);
    DAAL_CHECK_STATUS_VAR(checkRowCount(data.nRows));
    TreeBuilder<algorithmFPType, Record, cpu> builder(data, params, inputs);
    DAAL_CHECK_STATUS_VAR(builder.init());

    ExecutorParams executorParams;
    executorParams.nTrees        = nTrees;
    executorParams.maxLeafNodes  = params.maxLeafNodes;
    executorParams.treesInFlight = params.treesInFlight;
    executorParams.numaNodes     = params.numaNodes;
    Executor<TreeBuilder<algorithmFPType, Record, cpu>, cpu> executor(builder, executorParams);
    return executor.run();
}

} // namespace internal
} // namespace engine
} // namespace dtrees
} // namespace algorithms
} // namespace daal

#endif
