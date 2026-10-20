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
//  Implementation of the tree builder with the exact phase at every node.
//  Included only by tree_builder_fpt_cpu.cpp.
//--
*/

#ifndef __DTREES_ENGINE_TREE_BUILDER_I__
#define __DTREES_ENGINE_TREE_BUILDER_I__

#include <type_traits>

#include "src/algorithms/dtrees/engine/cost_function.h"
#include "src/algorithms/dtrees/engine/executor.h"
#include "src/algorithms/dtrees/engine/node_rng.h"
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
    bool valid       = false;
    bool categorical = false;
    uint32_t feature = 0;
    uint32_t nLeft   = 0; // records going left
    double threshold = 0; // ordered: rows with x <= threshold go left; categorical: rows with x == threshold
    double proxy     = 0; // cost function proxy, larger is better (cost_function.h)
    double wLeft     = 0;
};

/// Scratch for one node. Below parallelSortRecords a thread's own scratch is used: those nodes call no
/// parallel algorithm, so the thread never runs another node while it holds it. Larger nodes sort in
/// parallel, and a thread waiting in the sort can run other nodes, so they get scratch of their own.
template <typename algorithmFPType, CpuType cpu>
struct NodeScratch
{
    LargeArray<algorithmFPType, cpu> x;
    LargeArray<int32_t, cpu> order;
    LargeArray<double, cpu> w;
    LargeArray<double, cpu> y;
    LargeArray<uint32_t, cpu> label;
    LargeArray<double, cpu> classWeights;
    LargeArray<uint32_t, cpu> permutation;
    size_t capacity = 0;

    bool ensure(size_t n, size_t nClasses, size_t nFeatures)
    {
        if (!permutation.get())
        {
            if (!permutation.reset(2 * nFeatures)) return false;
            if (nClasses && !classWeights.reset(2 * nClasses)) return false;
        }
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
        Candidate split;
    };

    struct TreeContext
    {
        ResponseSet<Record, cpu> records;
        NodePool<cpu> nodes;
        CounterRng<cpu> rng;
        int numaNode = 0;
    };

    TreeBuilder(const TreeData<algorithmFPType, cpu> & data, const TreeParams & params, TreeInputs<Record, cpu> & inputs)
        : _data(data),
          _params(params),
          _inputs(inputs),
          _k(params.featuresPerNode && params.featuresPerNode < data.nFeatures ? params.featuresPerNode : data.nFeatures),
          _scratch([]() -> Scratch * { return new Scratch(); }),
          _prefetchRows(defaultPrefetchDistance<cpu>()),
          _l2Bytes(services::internal::getL2CacheSize())
    {}

    ~TreeBuilder()
    {
        _scratch.reduce([](Scratch * s) { delete s; });
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
        const size_t n = ctx.records.size();
        if (!n) return services::Status(services::ErrorIncorrectNumberOfObservations);
        // A tree of n records has at most n leaves and n - 1 splits.
        s = ctx.nodes.reset(2 * n);
        DAAL_CHECK_STATUS_VAR(s);
        root     = WorkItem();
        root.n   = uint32_t(n);
        root.key = treeSeed;
        return s;
    }

    services::Status evaluate(TreeContext & ctx, WorkItem & item, bool & splittable)
    {
        splittable = false;
        Scratch own;
        Scratch * scratch = item.n >= parallelSortRecords ? &own : _scratch.local();
        DAAL_CHECK_MALLOC(scratch && scratch->ensure(item.n, _params.nClasses, _data.nFeatures));
        const Record * records = ctx.records.buffer(item.depth) + item.start;

        // Node statistics from its records.
        uint64_t nSamples = 0;
        for (size_t i = 0; i < item.n; ++i)
        {
            scratch->w[i] = recordWeight(records[i]);
            nSamples += recordCount(records[i]);
            setResponse(*scratch, i, records[i]);
        }
        Cost cost = makeCost(*scratch);
        setParent(cost, *scratch, item.n);
        // Purity is decided exactly: impurity computed from sums can round to a small positive value
        // for a pure node, which would then split on zero gain.
        const bool pure = isPure(cost, *scratch, item.n);

        TreeNode & node = ctx.nodes[item.node];
        node            = TreeNode();
        node.feature    = -1;
        node.impurity   = pure ? 0 : cost.impurity();
        node.nRecords   = item.n;
        node.nSamples   = uint32_t(nSamples);
        node.depth      = item.depth;
        node.start      = item.start;

        const double w = cost.weight();
        if (item.n < _params.minSplitRecords || item.n < 2 * _params.minLeafRecords || w < 2 * _params.minLeafWeight
            || (_params.maxDepth && item.depth >= _params.maxDepth) || pure || node.impurity <= _params.impurityThreshold)
            return services::Status();

        Candidate best;
        services::Status s;
        if (_params.counterDraws)
        {
            CounterNodeDraws<cpu> draws(ctx.rng, item.key);
            best = searchSplit(ctx, records, item.n, *scratch, cost, draws, s);
            s |= draws.status();
        }
        else
        {
            void * stream = _inputs.nodeStream();
            DAAL_CHECK(stream, services::ErrorNullPtr);
            StreamNodeDraws<cpu> draws(stream);
            best = searchSplit(ctx, records, item.n, *scratch, cost, draws, s);
            s |= draws.status();
        }
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
        const Candidate & split   = item.split;
        const Record * in         = ctx.records.buffer(item.depth) + item.start;
        Record * out              = ctx.records.buffer(item.depth + 1) + item.start;
        const algorithmFPType * x = _data.x->data(ctx.numaNode);
        const size_t p            = _data.nFeatures;
        size_t iLeft              = 0;
        size_t iRight             = split.nLeft;
        prefetchedLoop<cpu>(
            item.n, prefetchDistance(item.n), [&](size_t i) { DAAL_PREFETCH_READ_T0(x + size_t(in[i].row) * p + split.feature); },
            [&](size_t i) {
                const algorithmFPType v = x[size_t(in[i].row) * p + split.feature];
                if (goesLeft(v, split))
                    out[iLeft++] = in[i];
                else
                    out[iRight++] = in[i];
            });
        if (iLeft != split.nLeft || iRight != item.n) return services::Status(services::ErrorIncorrectInternalFunctionParameter);

        const uint32_t pair = ctx.nodes.allocatePair();
        DAAL_CHECK_MALLOC(pair);
        TreeNode & node       = ctx.nodes[item.node];
        node.feature          = int32_t(split.feature);
        node.threshold        = split.threshold;
        node.categorical      = split.categorical;
        node.weightedDecrease = item.improvement;
        node.left             = pair;
        // Missing values follow the side with more weight (design 4.16).
        const double wTotal = recordsWeight(in, item.n);
        node.defaultLeft    = split.wLeft >= wTotal - split.wLeft;

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
        return services::Status();
    }

    services::Status makeLeaf(TreeContext & ctx, WorkItem & item)
    {
        TreeNode & node = ctx.nodes[item.node];
        node.feature    = -1;
        node.start      = item.start;
        return services::Status();
    }

    services::Status endTree(TreeContext & ctx, size_t iTree, const services::Status & built)
    {
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
    static bool goesLeft(algorithmFPType v, const Candidate & split)
    {
        if (split.categorical) return double(v) == split.threshold;
        return daal::algorithms::internal::isNaN<algorithmFPType, cpu>(v) || double(v) <= split.threshold;
    }

    static double recordsWeight(const Record * r, size_t n)
    {
        double w = 0;
        for (size_t i = 0; i < n; ++i) w += recordWeight(r[i]);
        return w;
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
    void setParent(Cost & cost, Scratch & s, size_t n) const
    {
        if constexpr (isClassification)
            cost.setParent(s.label.get(), s.w.get(), n);
        else
            cost.setParent(s.y.get(), s.w.get(), n);
    }
    void addLeft(Cost & cost, const Scratch & s, size_t i) const
    {
        if constexpr (isClassification)
            cost.addLeft(s.label[i], s.w[i]);
        else
            cost.addLeft(s.y[i], s.w[i]);
    }

    /// One class holds all the weight, or every response is equal.
    bool isPure(const Cost & cost, const Scratch & s, size_t n) const
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
                if (s.y[i] != s.y[0]) return false;
            return true;
        }
    }

    bool admissible(size_t nLeft, size_t n, double wLeft, double w) const
    {
        const double wRight = w - wLeft;
        return nLeft >= _params.minLeafRecords && n - nLeft >= _params.minLeafRecords && wLeft >= _params.minLeafWeight
               && wRight >= _params.minLeafWeight && wLeft > 0 && wRight > 0;
    }

    /// The best split over the node's first k non-constant features in draw order (design 4.6, 4.13).
    template <typename Draws>
    Candidate searchSplit(TreeContext & ctx, const Record * records, size_t n, Scratch & scratch, Cost & cost, Draws & draws,
                          services::Status & status) const
    {
        FeaturePermutation<cpu> permutation(uint32_t(_data.nFeatures), _k, scratch.permutation.get());
        Candidate best;
        size_t visited = 0;
        while (visited < _k && !permutation.exhausted())
        {
            const uint32_t f  = permutation.next(draws);
            bool constant     = false;
            const Candidate c = splitOnFeature(ctx, records, n, scratch, cost, draws, f, constant, status);
            if (!status) return best;
            if (constant) continue; // constant features do not count toward k, as in sklearn
            ++visited;
            if (c.valid && (!best.valid || c.proxy > best.proxy)) best = c;
        }
        return best;
    }

    template <typename Draws>
    Candidate splitOnFeature(TreeContext & ctx, const Record * records, size_t n, Scratch & scratch, Cost & cost, Draws & draws, uint32_t f,
                             bool & constant, services::Status & status) const
    {
        const bool categorical    = _data.categorical && _data.categorical[f];
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
            // Probabilities are the weighted class shares; they sum to 1 with user weights too.
            for (size_t c = 0; c < nClasses; ++c) prob[c] = total > 0 ? cw[c] / total : 0;
        }
    }

    double regressionLeaf(const Record * r, size_t n) const
    {
        if constexpr (!isClassification)
        {
            double w = 0, s = 0;
            for (size_t i = 0; i < n; ++i)
            {
                w += recordWeight(r[i]);
                s += recordWeight(r[i]) * recordResponse(r[i]);
            }
            return w > 0 ? s / w : 0;
        }
        return 0;
    }

    const TreeData<algorithmFPType, cpu> & _data;
    const TreeParams _params;
    TreeInputs<Record, cpu> & _inputs;
    const size_t _k;
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
    ExecutorParams executorParams;
    executorParams.nTrees            = nTrees;
    executorParams.maxLeafNodes      = params.maxLeafNodes;
    executorParams.treesInFlight     = params.treesInFlight;
    executorParams.serialSubtreeRows = params.serialSubtreeRows;
    Executor<TreeBuilder<algorithmFPType, Record, cpu>, cpu> executor(builder, executorParams);
    return executor.run();
}

} // namespace internal
} // namespace engine
} // namespace dtrees
} // namespace algorithms
} // namespace daal

#endif
