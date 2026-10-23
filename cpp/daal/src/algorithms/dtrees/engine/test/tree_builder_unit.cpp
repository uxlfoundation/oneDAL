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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "oneapi/dal/test/engine/common.hpp"
#include "data_management/data/homogen_numeric_table.h"
#include "services/env_detect.h"
#include "services/internal/daal_kernel_defines.h"
#include "src/algorithms/dtrees/engine/node_rng.h"
#include "src/algorithms/dtrees/engine/quantile_binner.h"

#include "src/algorithms/dtrees/engine/tree_builder.h"

namespace daal::algorithms::dtrees::engine::internal::test
{
using namespace daal::data_management;
using daal::internal::CpuType;
using dtrees::internal::DecisionTreeNode;

constexpr auto baseCpu = DAAL_BASE_CPU;

template <typename F>
void forEachCpu(const F & f)
{
    f(std::integral_constant<CpuType, DAAL_BASE_CPU>());
#if defined(TARGET_X86_64) && defined(DAAL_KERNEL_AVX512)
    if (services::Environment::getInstance()->getCpuId() >= int(daal::internal::avx512)) f(std::integral_constant<CpuType, daal::internal::avx512>());
#endif
}

class ThreadCountGuard
{
public:
    explicit ThreadCountGuard(size_t n) : _saved(services::Environment::getInstance()->getNumberOfThreads())
    {
        services::Environment::getInstance()->setNumberOfThreads(n);
    }
    ~ThreadCountGuard() { services::Environment::getInstance()->setNumberOfThreads(_saved); }

private:
    size_t _saved;
};

/// A training set: row-major x, responses (class or value) and optional weights.
struct Dataset
{
    size_t nRows = 0, p = 0, nClasses = 0;
    std::vector<double> x, y, w;
    std::vector<bool> categorical;
};

/// Per-tree bootstrap counts: n draws with replacement, or empty for every row once.
std::vector<std::vector<uint16_t> > bootstrapCounts(size_t nTrees, size_t nRows, uint64_t seed)
{
    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<size_t> row(0, nRows - 1);
    std::vector<std::vector<uint16_t> > counts(nTrees, std::vector<uint16_t>(nRows, 0));
    for (auto & c : counts)
        for (size_t i = 0; i < nRows; ++i) ++c[row(gen)];
    return counts;
}

template <typename Record, CpuType cpu>
class Inputs : public TreeInputs<Record, cpu>
{
public:
    Inputs(const ResponseSource<double, cpu> & src, const std::vector<std::vector<uint16_t> > & counts, uint64_t seed, size_t nTrees)
        : trees(nTrees), _src(src), _counts(counts), _seed(seed)
    {}

    services::Status prepareTree(size_t iTree, int numaNode, ResponseSet<Record, cpu> & records, uint64_t & treeSeed) override
    {
        double total = 0;
        treeSeed     = mix64<cpu>(_seed + iTree);
        return records.fill(_counts.empty() ? nullptr : _counts[iTree].data(), _src, numaNode, total);
    }

    services::Status emitTree(size_t iTree, EmittedTree & tree) override
    {
        std::lock_guard<std::mutex> lock(_mutex);
        trees[iTree] = std::move(tree);
        return services::Status();
    }

    void * nodeStream() override
    {
        std::lock_guard<std::mutex> lock(_mutex);
        const auto id = std::this_thread::get_id();
        auto it       = _streams.find(id);
        if (it == _streams.end())
            it = _streams
                     .emplace(id, std::unique_ptr<daal::internal::BaseRNGsInst<cpu> >(
                                      new daal::internal::BaseRNGsInst<cpu>(unsigned(_seed + _streams.size()), __DAAL_BRNG_MT19937)))
                     .first;
        return it->second->getState();
    }

    std::vector<EmittedTree> trees;

private:
    const ResponseSource<double, cpu> & _src;
    const std::vector<std::vector<uint16_t> > & _counts;
    uint64_t _seed;
    std::mutex _mutex;
    std::map<std::thread::id, std::unique_ptr<daal::internal::BaseRNGsInst<cpu> > > _streams;
};

/// Trains nTrees trees and returns their tables; with binning, a hist fit over the binned data.
template <typename Record, CpuType cpu>
std::vector<EmittedTree> train(const Dataset & d, const TreeParams & params, size_t nTrees, const std::vector<std::vector<uint16_t> > & counts,
                               uint64_t seed = 777, const BinningParams * binning = nullptr)
{
    std::vector<double> xCopy = d.x;
    NumericTablePtr xt        = HomogenNumericTable<double>::create(xCopy.data(), d.p, d.nRows);
    for (size_t f = 0; f < d.p; ++f)
        if (!d.categorical.empty() && d.categorical[f]) (*xt->getDictionarySharedPtr())[f].featureType = features::DAAL_CATEGORICAL;
    std::vector<double> yCopy = d.y;
    NumericTablePtr yt        = HomogenNumericTable<double>::create(yCopy.data(), 1, d.nRows);
    std::vector<double> wCopy = d.w;
    NumericTablePtr wt        = d.w.empty() ? NumericTablePtr() : NumericTablePtr(HomogenNumericTable<double>::create(wCopy.data(), 1, d.nRows));

    NumaReplicated<double, cpu> x;
    REQUIRE((copyTableRows<double, cpu>(*xt, false, x)).ok());
    ResponseSource<double, cpu> src;
    REQUIRE(src.build(*yt, wt.get(), d.nClasses > 0, false).ok());
    std::unique_ptr<bool[]> cat(new bool[d.p]);
    for (size_t f = 0; f < d.p; ++f) cat[f] = !d.categorical.empty() && d.categorical[f];

    TreeData<double, cpu> data;
    data.x           = &x;
    data.responses   = &src;
    data.categorical = cat.get();
    data.nRows       = d.nRows;
    data.nFeatures   = d.p;
    BinMatrix<cpu> bins;
    if (binning)
    {
        bool binned = false;
        daal::internal::BaseRNGsInst<cpu> engine(unsigned(seed), __DAAL_BRNG_PHILOX4X32X10);
        REQUIRE((QuantileBinner<double, cpu>::build(*xt, *binning, engine.getState(), bins, binned)).ok());
        REQUIRE(binned);
        data.bins = &bins;
    }
    Inputs<Record, cpu> inputs(src, counts, seed, nTrees);
    REQUIRE((trainTrees<double, Record, cpu>(data, params, nTrees, inputs)).ok());
    return std::move(inputs.trees);
}

const DecisionTreeNode * rowsOf(const EmittedTree & t)
{
    return static_cast<const DecisionTreeNode *>(t.nodes->getArray());
}

/// The leaf row a sample reaches, routing as the prediction kernels do.
size_t leafOf(const EmittedTree & t, const double * x, const std::vector<bool> & categorical)
{
    const DecisionTreeNode * rows = rowsOf(t);
    size_t r                      = 0;
    while (rows[r].featureIndex >= 0)
    {
        const size_t f   = size_t(rows[r].featureIndex);
        const bool cat   = !categorical.empty() && categorical[f];
        const bool right = cat ? (x[f] != rows[r].featureValueOrResponse) : (x[f] > rows[r].featureValueOrResponse);
        r                = rows[r].leftIndexOrClass + (right ? 1 : 0);
    }
    return r;
}

void requireSameTables(const EmittedTree & a, const EmittedTree & b)
{
    const size_t n = a.nodes->getNumberOfRows();
    REQUIRE(n == b.nodes->getNumberOfRows());
    const DecisionTreeNode * ra = rowsOf(a);
    const DecisionTreeNode * rb = rowsOf(b);
    for (size_t i = 0; i < n; ++i)
    {
        REQUIRE(ra[i].featureIndex == rb[i].featureIndex);
        REQUIRE(ra[i].leftIndexOrClass == rb[i].leftIndexOrClass);
        REQUIRE(ra[i].featureValueOrResponse == rb[i].featureValueOrResponse);
        REQUIRE(ra[i].defaultLeft == rb[i].defaultLeft);
        REQUIRE(a.impurity->getArray()[i] == b.impurity->getArray()[i]);
        REQUIRE(a.nodeSamples->getArray()[i] == b.nodeSamples->getArray()[i]);
    }
    const size_t np = a.probabilities->getNumberOfRows() * a.probabilities->getNumberOfColumns();
    REQUIRE(np == b.probabilities->getNumberOfRows() * b.probabilities->getNumberOfColumns());
    for (size_t i = 0; i < np; ++i) REQUIRE(a.probabilities->getArray()[i] == b.probabilities->getArray()[i]);
}

/// Every in-bag row reaches a leaf whose sample count, class shares or mean match the rows reaching it;
/// children are adjacent and every node is reached.
void requireConsistent(const Dataset & d, const EmittedTree & t, const std::vector<uint16_t> & counts)
{
    const size_t nNodes           = t.nodes->getNumberOfRows();
    const DecisionTreeNode * rows = rowsOf(t);
    for (size_t r = 0; r < nNodes; ++r)
        if (rows[r].featureIndex >= 0) REQUIRE((rows[r].leftIndexOrClass > r && rows[r].leftIndexOrClass + 1 < nNodes));
    std::vector<double> samples(nNodes, 0), wSum(nNodes, 0), ySum(nNodes, 0);
    std::vector<std::vector<double> > classW(nNodes, std::vector<double>(d.nClasses ? d.nClasses : 1, 0));
    for (size_t i = 0; i < d.nRows; ++i)
    {
        const double c = counts.empty() ? 1 : counts[i];
        const double w = c * (d.w.empty() ? 1 : d.w[i] / *std::max_element(d.w.begin(), d.w.end()));
        if (!(w > 0)) continue;
        const size_t leaf = leafOf(t, d.x.data() + i * d.p, d.categorical);
        samples[leaf] += d.w.empty() ? c : 1;
        wSum[leaf] += w;
        ySum[leaf] += w * d.y[i];
        if (d.nClasses) classW[leaf][size_t(d.y[i])] += w;
    }
    for (size_t r = 0; r < nNodes; ++r)
    {
        if (rows[r].featureIndex >= 0) continue;
        REQUIRE(samples[r] > 0);
        REQUIRE(double(t.nodeSamples->getArray()[r]) == samples[r]);
        if (d.nClasses)
        {
            const double * prob = t.probabilities->getArray() + r * d.nClasses;
            double total        = 0;
            for (size_t c = 0; c < d.nClasses; ++c)
            {
                REQUIRE(std::abs(prob[c] - classW[r][c] / wSum[r]) < 1e-12);
                total += prob[c];
            }
            REQUIRE(std::abs(total - 1) < 1e-12);
            REQUIRE(classW[r][rows[r].leftIndexOrClass] == *std::max_element(classW[r].begin(), classW[r].end()));
        }
        else
        {
            REQUIRE(std::abs(rows[r].featureValueOrResponse - ySum[r] / wSum[r]) <= 1e-12 * (1 + std::abs(ySum[r] / wSum[r])));
        }
    }
}

size_t depthOf(const EmittedTree & t, size_t r = 0)
{
    const DecisionTreeNode * rows = rowsOf(t);
    if (rows[r].featureIndex < 0) return 0;
    return 1 + std::max(depthOf(t, rows[r].leftIndexOrClass), depthOf(t, rows[r].leftIndexOrClass + 1));
}

size_t leavesOf(const EmittedTree & t)
{
    size_t n = 0;
    for (size_t r = 0; r < t.nodes->getNumberOfRows(); ++r) n += rowsOf(t)[r].featureIndex < 0;
    return n;
}

Dataset classificationData(size_t nRows, size_t p, size_t nClasses, uint64_t seed, bool weighted = false)
{
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> normal;
    std::uniform_real_distribution<double> unit(0.1, 2.0);
    Dataset d;
    d.nRows    = nRows;
    d.p        = p;
    d.nClasses = nClasses;
    d.x.resize(nRows * p);
    d.y.resize(nRows);
    for (size_t i = 0; i < nRows; ++i)
    {
        double s = 0;
        for (size_t f = 0; f < p; ++f)
        {
            d.x[i * p + f] = normal(gen);
            s += (f + 1) * d.x[i * p + f];
        }
        d.y[i] = double(size_t(std::abs(s + normal(gen))) % nClasses);
    }
    if (weighted)
    {
        d.w.resize(nRows);
        for (auto & v : d.w) v = unit(gen);
    }
    return d;
}

Dataset regressionData(size_t nRows, size_t p, uint64_t seed, bool weighted = false)
{
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> normal;
    std::uniform_real_distribution<double> unit(0.1, 2.0);
    Dataset d;
    d.nRows = nRows;
    d.p     = p;
    d.x.resize(nRows * p);
    d.y.resize(nRows);
    for (size_t i = 0; i < nRows; ++i)
    {
        for (size_t f = 0; f < p; ++f) d.x[i * p + f] = normal(gen);
        d.y[i] = std::sin(2 * d.x[i * p]) + 0.5 * d.x[i * p + 1] + 0.1 * normal(gen);
    }
    if (weighted)
    {
        d.w.resize(nRows);
        for (auto & v : d.w) v = unit(gen);
    }
    return d;
}

// --- Independent reference: brute-force best split over every feature and boundary, min leaf 1. ---

/// Weight of row i: its user weight, or 1.
double weightOf(const Dataset & d, size_t i)
{
    return d.w.empty() ? 1. : d.w[i];
}

/// Proxy of a partition (cost_function.h), computed directly: sum_c L_c^2 / wL + sum_c R_c^2 / wR for Gini,
/// sL^2 / wL + sR^2 / wR for MSE, with the rows' weights. The engine's weights are scaled to a maximum of 1,
/// which scales every proxy alike.
double partitionProxy(const Dataset & d, const std::vector<size_t> & l, const std::vector<size_t> & r)
{
    auto side = [&](const std::vector<size_t> & rows) {
        double w = 0;
        for (size_t i : rows) w += weightOf(d, i);
        if (d.nClasses)
        {
            std::vector<double> cw(d.nClasses, 0);
            for (size_t i : rows) cw[size_t(d.y[i])] += weightOf(d, i);
            double q = 0;
            for (double c : cw) q += c * c;
            return q / w;
        }
        double s = 0;
        for (size_t i : rows) s += weightOf(d, i) * d.y[i];
        return s * s / w;
    };
    return side(l) + side(r);
}

/// The best proxy over all features and boundaries between distinct values, or -1 without any.
double bestProxy(const Dataset & d, const std::vector<size_t> & rows)
{
    double best = -1;
    for (size_t f = 0; f < d.p; ++f)
    {
        std::vector<size_t> sorted = rows;
        std::stable_sort(sorted.begin(), sorted.end(), [&](size_t a, size_t b) { return d.x[a * d.p + f] < d.x[b * d.p + f]; });
        for (size_t i = 0; i + 1 < sorted.size(); ++i)
        {
            if (d.x[sorted[i] * d.p + f] == d.x[sorted[i + 1] * d.p + f]) continue;
            const std::vector<size_t> l(sorted.begin(), sorted.begin() + i + 1), r(sorted.begin() + i + 1, sorted.end());
            best = std::max(best, partitionProxy(d, l, r));
        }
    }
    return best;
}

/// Every split is optimal up to rounding (ties between features may go either way), every leaf is pure
/// or has no split, and leaf values match their rows. Follows the engine's own partition.
void requireOptimal(const Dataset & d, const EmittedTree & t, size_t r, const std::vector<size_t> & rows)
{
    const DecisionTreeNode & row = rowsOf(t)[r];
    bool pure                    = true;
    for (size_t i : rows) pure = pure && d.y[i] == d.y[rows[0]];
    const double best = pure ? -1 : bestProxy(d, rows);
    INFO("row " << r << " with " << rows.size() << " rows");
    if (row.featureIndex < 0)
    {
        REQUIRE(best < 0);
        if (d.nClasses)
        {
            std::vector<double> cw(d.nClasses, 0);
            for (size_t i : rows) cw[size_t(d.y[i])] += weightOf(d, i);
            REQUIRE(cw[row.leftIndexOrClass] == *std::max_element(cw.begin(), cw.end()));
        }
        else
        {
            double s = 0, w = 0;
            for (size_t i : rows)
            {
                s += weightOf(d, i) * d.y[i];
                w += weightOf(d, i);
            }
            const double mean = s / w;
            REQUIRE(std::abs(row.featureValueOrResponse - mean) <= 1e-12 * (1 + std::abs(mean)));
        }
        return;
    }
    std::vector<size_t> l, rr;
    for (size_t i : rows) (d.x[i * d.p + size_t(row.featureIndex)] <= row.featureValueOrResponse ? l : rr).push_back(i);
    REQUIRE(!l.empty());
    REQUIRE(!rr.empty());
    REQUIRE(best > 0);
    REQUIRE(partitionProxy(d, l, rr) >= best * (1 - 1e-12));
    requireOptimal(d, t, row.leftIndexOrClass, l);
    requireOptimal(d, t, row.leftIndexOrClass + 1, rr);
}

// --- Tests ---

TEST("with every feature at every node, every split is optimal and every leaf unsplittable", "[unit]")
{
    for (size_t nClasses : { 0, 2, 3 })
    {
        const Dataset d = nClasses ? classificationData(300, 5, nClasses, 11) : regressionData(300, 5, 12);
        TreeParams params;
        params.nClasses = nClasses;
        std::vector<size_t> all(d.nRows);
        for (size_t i = 0; i < d.nRows; ++i) all[i] = i;
        const std::vector<std::vector<uint16_t> > noBootstrap;
        const auto trees =
            nClasses ? train<ClsCountRecord, baseCpu>(d, params, 3, noBootstrap) : train<RegCountRecord<double>, baseCpu>(d, params, 3, noBootstrap);
        for (const auto & t : trees) requireOptimal(d, t, 0, all);
    }
}

TEST("with user weights every split is still optimal for the weighted cost", "[unit]")
{
    // Weighted left sums once took the weight of the wrong row (#2092); unit weights cannot show that.
    for (size_t nClasses : { 0, 2, 3 })
    {
        const Dataset d = nClasses ? classificationData(300, 5, nClasses, 13, true) : regressionData(300, 5, 14, true);
        TreeParams params;
        params.nClasses = nClasses;
        std::vector<size_t> all(d.nRows);
        for (size_t i = 0; i < d.nRows; ++i) all[i] = i;
        const std::vector<std::vector<uint16_t> > noBootstrap;
        const auto trees = nClasses ? train<ClsWeightRecord<double>, baseCpu>(d, params, 3, noBootstrap) :
                                      train<RegWeightRecord<double>, baseCpu>(d, params, 3, noBootstrap);
        for (const auto & t : trees) requireOptimal(d, t, 0, all);
    }
}

TEST("every in-bag row reaches a leaf that matches its rows, with bootstrap and weights", "[unit]")
{
    constexpr size_t nTrees = 6;
    for (bool weighted : { false, true })
    {
        for (bool random : { false, true })
        {
            const Dataset c   = classificationData(2000, 7, 4, 21, weighted);
            const Dataset r   = regressionData(2000, 7, 22, weighted);
            const auto counts = bootstrapCounts(nTrees, 2000, 23);
            TreeParams params;
            params.featuresPerNode = 3;
            params.randomSplitter  = random;
            params.nClasses        = 4;
            const auto ct          = weighted ? train<ClsWeightRecord<double>, baseCpu>(c, params, nTrees, counts) :
                                                train<ClsCountRecord, baseCpu>(c, params, nTrees, counts);
            params.nClasses        = 0;
            const auto rt          = weighted ? train<RegWeightRecord<double>, baseCpu>(r, params, nTrees, counts) :
                                                train<RegCountRecord<double>, baseCpu>(r, params, nTrees, counts);
            for (size_t i = 0; i < nTrees; ++i)
            {
                requireConsistent(c, ct[i], counts[i]);
                requireConsistent(r, rt[i], counts[i]);
            }
        }
    }
}

// Identity across CpuTypes is not required (design 4.13): FMA contraction in the AVX-512 objects
// rounds sums differently. Each CpuType is checked against its own single-thread trees.
TEST("trees do not depend on threads, serial subtrees or trees in flight", "[unit]")
{
    constexpr size_t nTrees = 8;
    const Dataset c         = classificationData(3000, 9, 3, 31);
    const Dataset r         = regressionData(3000, 9, 32, true);
    const auto counts       = bootstrapCounts(nTrees, 3000, 33);
    forEachCpu([&](auto cpu) {
        constexpr CpuType ct = decltype(cpu)::value;
        for (bool random : { false, true })
        {
            for (size_t maxLeaves : { size_t(0), size_t(40) })
            {
                TreeParams params;
                params.featuresPerNode = 3;
                params.randomSplitter  = random;
                params.maxLeafNodes    = maxLeaves;
                std::vector<EmittedTree> refC, refR;
                {
                    ThreadCountGuard one(1);
                    params.nClasses = 3;
                    refC            = train<ClsCountRecord, ct>(c, params, nTrees, counts);
                    params.nClasses = 0;
                    refR            = train<RegWeightRecord<double>, ct>(r, params, nTrees, counts);
                }
                for (size_t threads : { 4, 0 })
                {
                    std::unique_ptr<ThreadCountGuard> guard(threads ? new ThreadCountGuard(threads) : nullptr);
                    for (size_t serial : { size_t(0), size_t(64), std::numeric_limits<size_t>::max() })
                    {
                        for (size_t inFlight : { 0, 2 })
                        {
                            params.serialSubtreeRows = serial;
                            params.treesInFlight     = inFlight;
                            params.nClasses          = 3;
                            const auto tc            = train<ClsCountRecord, ct>(c, params, nTrees, counts);
                            params.nClasses          = 0;
                            const auto tr            = train<RegWeightRecord<double>, ct>(r, params, nTrees, counts);
                            for (size_t i = 0; i < nTrees; ++i)
                            {
                                requireSameTables(refC[i], tc[i]);
                                requireSameTables(refR[i], tr[i]);
                            }
                        }
                    }
                }
            }
        }
    });
}

TEST("maxDepth, minLeafRecords and maxLeafNodes bound the trees", "[unit]")
{
    const Dataset d   = classificationData(1500, 6, 3, 41);
    const auto counts = bootstrapCounts(4, 1500, 42);
    TreeParams params;
    params.nClasses        = 3;
    params.featuresPerNode = 2;
    params.maxDepth        = 4;
    for (const auto & t : train<ClsCountRecord, baseCpu>(d, params, 4, counts)) REQUIRE(depthOf(t) <= 4);
    params.maxDepth     = 0;
    params.maxLeafNodes = 17;
    for (const auto & t : train<ClsCountRecord, baseCpu>(d, params, 4, counts)) REQUIRE(leavesOf(t) <= 17);
    params.maxLeafNodes    = 0;
    params.minLeafRecords  = 25;
    params.minSplitRecords = 50;
    const std::vector<std::vector<uint16_t> > noBootstrap;
    for (const auto & t : train<ClsCountRecord, baseCpu>(d, params, 2, noBootstrap))
    {
        std::vector<size_t> perLeaf(t.nodes->getNumberOfRows(), 0);
        for (size_t i = 0; i < d.nRows; ++i) ++perLeaf[leafOf(t, d.x.data() + i * d.p, d.categorical)];
        for (size_t r = 0; r < perLeaf.size(); ++r)
            if (rowsOf(t)[r].featureIndex < 0) REQUIRE(perLeaf[r] >= 25);
    }
}

/// Chi-square statistic of counts against equal expectation.
double chiSquare(const std::vector<size_t> & counts)
{
    double total = 0;
    for (size_t c : counts) total += double(c);
    const double e = total / double(counts.size());
    double x2      = 0;
    for (size_t c : counts) x2 += (double(c) - e) * (double(c) - e) / e;
    return x2;
}

// Upper 0.1% points of chi-square with 7 and 2 degrees of freedom.
constexpr double chiSquare7Dof999 = 24.32;
constexpr double chiSquare2Dof999 = 13.82;

TEST("the drawn feature is uniform over the features", "[unit]")
{
    // One feature per node and depth 1: the root splits on its first drawn feature.
    const Dataset d = classificationData(400, 8, 2, 51);
    TreeParams params;
    params.nClasses         = 2;
    params.featuresPerNode  = 1;
    params.maxDepth         = 1;
    constexpr size_t nTrees = 4000;
    const std::vector<std::vector<uint16_t> > noBootstrap;
    std::vector<size_t> hits(d.p, 0);
    for (const auto & t : train<ClsCountRecord, baseCpu>(d, params, nTrees, noBootstrap))
        if (rowsOf(t)[0].featureIndex >= 0) ++hits[size_t(rowsOf(t)[0].featureIndex)];
    REQUIRE(chiSquare(hits) < chiSquare7Dof999);
}

TEST("random thresholds are uniform over the node's range", "[unit]")
{
    Dataset d;
    d.nRows    = 1000;
    d.p        = 1;
    d.nClasses = 2;
    for (size_t i = 0; i < d.nRows; ++i)
    {
        d.x.push_back(double(i) / double(d.nRows - 1));
        d.y.push_back(double(i % 2));
    }
    TreeParams params;
    params.nClasses       = 2;
    params.randomSplitter = true;
    params.maxDepth       = 1;
    const std::vector<std::vector<uint16_t> > noBootstrap;
    std::vector<size_t> hits(8, 0);
    for (const auto & t : train<ClsCountRecord, baseCpu>(d, params, 4000, noBootstrap))
    {
        REQUIRE(rowsOf(t)[0].featureIndex == 0);
        const double u = rowsOf(t)[0].featureValueOrResponse;
        REQUIRE((u >= 0 && u < 1));
        ++hits[std::min<size_t>(7, size_t(u * 8))];
    }
    REQUIRE(chiSquare(hits) < chiSquare7Dof999);
}

TEST("the random categorical split takes each category present with equal chance (defect 7)", "[unit]")
{
    Dataset d;
    d.nRows              = 900;
    d.p                  = 1;
    d.nClasses           = 2;
    d.categorical        = { true };
    const double codes[] = { 0, 1, 100 };
    for (size_t i = 0; i < d.nRows; ++i)
    {
        d.x.push_back(codes[i % 3]);
        d.y.push_back(double((i / 3) % 2));
    }
    TreeParams params;
    params.nClasses       = 2;
    params.randomSplitter = true;
    params.maxDepth       = 1;
    const std::vector<std::vector<uint16_t> > noBootstrap;
    std::vector<size_t> hits(3, 0);
    for (const auto & t : train<ClsCountRecord, baseCpu>(d, params, 3000, noBootstrap))
    {
        const double v = rowsOf(t)[0].featureValueOrResponse;
        ++hits[v == 0 ? 0 : v == 1 ? 1 : 2];
    }
    REQUIRE(chiSquare(hits) < chiSquare2Dof999);
}

TEST("the best categorical split equals brute force when one category dominates (defect 6)", "[unit]")
{
    // Category 0 holds 90% of the rows with mixed labels; small categories 1..5 are pure. The best
    // one-vs-rest split is found by trying each category against the Gini proxy.
    Dataset d;
    d.p           = 1;
    d.nClasses    = 2;
    d.categorical = { true };
    std::mt19937_64 gen(61);
    for (size_t i = 0; i < 900; ++i)
    {
        d.x.push_back(0);
        d.y.push_back(double(gen() % 2));
    }
    for (size_t c = 1; c <= 5; ++c)
        for (size_t i = 0; i < 20; ++i)
        {
            d.x.push_back(double(c));
            d.y.push_back(double(c % 2));
        }
    d.nRows          = d.x.size();
    double bestProxy = -1, bestCat = -1;
    for (double cat = 0; cat <= 5; ++cat)
    {
        double l[2] = { 0, 0 }, r[2] = { 0, 0 };
        for (size_t i = 0; i < d.nRows; ++i) (d.x[i] == cat ? l : r)[size_t(d.y[i])] += 1;
        const double wl = l[0] + l[1], wr = r[0] + r[1];
        const double proxy = (l[0] * l[0] + l[1] * l[1]) / wl + (r[0] * r[0] + r[1] * r[1]) / wr;
        if (proxy > bestProxy)
        {
            bestProxy = proxy;
            bestCat   = cat;
        }
    }
    TreeParams params;
    params.nClasses = 2;
    params.maxDepth = 1;
    const std::vector<std::vector<uint16_t> > noBootstrap;
    const auto trees = train<ClsCountRecord, baseCpu>(d, params, 1, noBootstrap);
    REQUIRE(rowsOf(trees[0])[0].featureIndex == 0);
    REQUIRE(rowsOf(trees[0])[0].featureValueOrResponse == bestCat);
}

TEST("nodes above parallelSortRecords sort in parallel, NaN included, independent of the thread count", "[unit]")
{
    // 300k rows put the top levels above the 2^17-record threshold of the parallel sort.
    Dataset d = regressionData(300000, 4, 81);
    for (size_t i = 0; i < d.nRows; i += 97) d.x[i * d.p + (i % d.p)] = std::numeric_limits<double>::quiet_NaN();
    TreeParams params;
    params.maxDepth = 3;
    const std::vector<std::vector<uint16_t> > noBootstrap;
    std::vector<EmittedTree> one;
    {
        ThreadCountGuard guard(1);
        one = train<RegCountRecord<double>, baseCpu>(d, params, 2, noBootstrap);
    }
    const auto many = train<RegCountRecord<double>, baseCpu>(d, params, 2, noBootstrap);
    for (size_t t = 0; t < 2; ++t)
    {
        requireConsistent(d, many[t], {});
        requireSameTables(one[t], many[t]);
    }
}

TEST("other engines draw from the calling thread's stream", "[unit]")
{
    const Dataset d   = classificationData(1500, 6, 3, 71);
    const auto counts = bootstrapCounts(4, 1500, 72);
    TreeParams params;
    params.nClasses        = 3;
    params.featuresPerNode = 2;
    params.counterDraws    = false;
    ThreadCountGuard one(1);
    const auto a = train<ClsCountRecord, baseCpu>(d, params, 4, counts);
    const auto b = train<ClsCountRecord, baseCpu>(d, params, 4, counts);
    for (size_t i = 0; i < 4; ++i)
    {
        requireConsistent(d, a[i], counts[i]);
        requireSameTables(a[i], b[i]); // one thread: the stream order is fixed
    }
}
// --- Histogram phase (design 4.6) ---

/// Integer features in [0, D), one categorical, NaN in feature 1; integer responses, so every bin and
/// row statistic is exact and the histogram and exact phases score candidates identically.
Dataset lowCardinalityData(size_t nRows, size_t nClasses, uint64_t seed)
{
    std::mt19937_64 gen(seed);
    const size_t p = 5;
    Dataset d;
    d.nRows       = nRows;
    d.p           = p;
    d.nClasses    = nClasses;
    d.categorical = { false, false, false, true, false };
    d.x.resize(nRows * p);
    d.y.resize(nRows);
    for (size_t i = 0; i < nRows; ++i)
    {
        double s = 0;
        for (size_t f = 0; f < p; ++f)
        {
            const size_t levels = 4 + 6 * f;
            d.x[i * p + f]      = double(gen() % levels);
            s += d.x[i * p + f] * (f == 3 ? 0.5 : 1.0);
        }
        if (i % 11 == 0) d.x[i * p + 1] = std::numeric_limits<double>::quiet_NaN();
        d.y[i] = nClasses ? double((size_t(s) + gen() % 3) % nClasses) : double(size_t(s) % 7 + gen() % 3);
    }
    return d;
}

/// Every in-bag row reaches the same leaf of both trees.
void requireSameRouting(const Dataset & d, const EmittedTree & a, const EmittedTree & b, const std::vector<uint16_t> & counts)
{
    for (size_t i = 0; i < d.nRows; ++i)
        if (counts.empty() || counts[i]) REQUIRE(leafOf(a, d.x.data() + i * d.p, d.categorical) == leafOf(b, d.x.data() + i * d.p, d.categorical));
}

/// a and b have the same structure, statistics and leaves; thresholds may differ where the routing of the
/// training rows does not.
void requireSameTrees(const Dataset & d, const EmittedTree & a, const EmittedTree & b, const std::vector<uint16_t> & counts)
{
    const size_t n = a.nodes->getNumberOfRows();
    REQUIRE(n == b.nodes->getNumberOfRows());
    const DecisionTreeNode * ra = rowsOf(a);
    const DecisionTreeNode * rb = rowsOf(b);
    for (size_t i = 0; i < n; ++i)
    {
        REQUIRE(ra[i].featureIndex == rb[i].featureIndex);
        REQUIRE(ra[i].leftIndexOrClass == rb[i].leftIndexOrClass);
        REQUIRE(ra[i].defaultLeft == rb[i].defaultLeft);
        REQUIRE(a.impurity->getArray()[i] == b.impurity->getArray()[i]);
        REQUIRE(a.nodeSamples->getArray()[i] == b.nodeSamples->getArray()[i]);
        if (ra[i].featureIndex < 0) REQUIRE(ra[i].featureValueOrResponse == rb[i].featureValueOrResponse);
    }
    const size_t np = a.probabilities->getNumberOfRows() * a.probabilities->getNumberOfColumns();
    for (size_t i = 0; i < np; ++i) REQUIRE(a.probabilities->getArray()[i] == b.probabilities->getArray()[i]);
    requireSameRouting(d, a, b, counts);
}

BinningParams exactBinning()
{
    BinningParams b;
    b.maxBins    = 64;
    b.minBinSize = 1;
    return b;
}

TEST("hist with one bin per value equals the exact phase on the same draws", "[unit]")
{
    constexpr size_t nTrees     = 6;
    const Dataset c             = lowCardinalityData(4000, 3, 91);
    const Dataset r             = lowCardinalityData(4000, 0, 92);
    const auto counts           = bootstrapCounts(nTrees, 4000, 93);
    const BinningParams binning = exactBinning();
    forEachCpu([&](auto cpu) {
        constexpr CpuType ct = decltype(cpu)::value;
        for (bool random : { false, true })
        {
            TreeParams params;
            params.featuresPerNode = 3;
            params.randomSplitter  = random;
            params.denseThreshold  = 0; // the histogram phase at every node
            params.nClasses        = 3;
            const auto exactC      = train<ClsCountRecord, ct>(c, params, nTrees, counts);
            const auto histC       = train<ClsCountRecord, ct>(c, params, nTrees, counts, 777, &binning);
            params.nClasses        = 0;
            const auto exactR      = train<RegCountRecord<double>, ct>(r, params, nTrees, counts);
            const auto histR       = train<RegCountRecord<double>, ct>(r, params, nTrees, counts, 777, &binning);
            size_t borderSplits    = 0;
            for (size_t t = 0; t < nTrees; ++t)
            {
                requireSameTrees(c, exactC[t], histC[t], counts[t]);
                requireSameTrees(r, exactR[t], histR[t], counts[t]);
                // Histogram thresholds are bin borders, here the integer values themselves.
                for (const EmittedTree * h : { &histC[t], &histR[t] })
                    for (size_t i = 0; i < h->nodes->getNumberOfRows(); ++i)
                    {
                        const DecisionTreeNode & node = rowsOf(*h)[i];
                        if (node.featureIndex < 0) continue;
                        REQUIRE(node.featureValueOrResponse == std::floor(node.featureValueOrResponse));
                        ++borderSplits;
                    }
            }
            REQUIRE(borderSplits > 0);
        }
    });
}

TEST("chunked histograms and partitions do not depend on chunking or the thread count", "[unit]")
{
    constexpr size_t nTrees = 3;
    Dataset c               = classificationData(40000, 6, 3, 101);
    Dataset r               = regressionData(40000, 6, 102, true);
    for (size_t i = 0; i < c.nRows; i += 53) c.x[i * c.p + (i % c.p)] = std::numeric_limits<double>::quiet_NaN();
    const auto counts = bootstrapCounts(nTrees, 40000, 103);
    BinningParams binning;
    binning.maxBins = 32;
    TreeParams params;
    params.featuresPerNode = 3;
    params.maxDepth        = 8;
    params.chunkRows       = 1000;
    std::vector<EmittedTree> refC, refR;
    {
        ThreadCountGuard one(1);
        params.nClasses = 3;
        refC            = train<ClsCountRecord, baseCpu>(c, params, nTrees, counts, 777, &binning);
        params.nClasses = 0;
        refR            = train<RegWeightRecord<double>, baseCpu>(r, params, nTrees, counts, 777, &binning);
    }
    params.nClasses  = 3;
    const auto manyC = train<ClsCountRecord, baseCpu>(c, params, nTrees, counts, 777, &binning);
    params.nClasses  = 0;
    const auto manyR = train<RegWeightRecord<double>, baseCpu>(r, params, nTrees, counts, 777, &binning);
    // Count statistics are integers, so one chunk per node gives the same classification trees.
    params.nClasses   = 3;
    params.chunkRows  = 0;
    const auto wholeC = train<ClsCountRecord, baseCpu>(c, params, nTrees, counts, 777, &binning);
    for (size_t t = 0; t < nTrees; ++t)
    {
        requireConsistent(c, refC[t], counts[t]);
        requireConsistent(r, refR[t], counts[t]);
        requireSameTables(refC[t], manyC[t]);
        requireSameTables(refR[t], manyR[t]);
        requireSameTables(refC[t], wholeC[t]);
    }
}

TEST("a histogram node with every drawn feature constant by bins takes the exact phase", "[unit]")
{
    // Two bins on one continuous feature: below the root, each child holds one bin, yet its values differ.
    Dataset d;
    d.nRows = 4000;
    d.p     = 1;
    std::mt19937_64 gen(111);
    std::normal_distribution<double> normal;
    for (size_t i = 0; i < d.nRows; ++i)
    {
        d.x.push_back(normal(gen));
        d.y.push_back(std::sin(2 * d.x.back()) + 0.1 * normal(gen));
    }
    BinningParams binning;
    binning.maxBins = 2;
    TreeParams params;
    params.maxDepth       = 3;
    params.denseThreshold = 0;
    const std::vector<std::vector<uint16_t> > noBootstrap;
    const auto trees      = train<RegCountRecord<double>, baseCpu>(d, params, 1, noBootstrap, 777, &binning);
    const EmittedTree & t = trees[0];
    requireConsistent(d, t, {});
    const DecisionTreeNode * rows = rowsOf(t);
    REQUIRE(rows[0].featureIndex >= 0);
    REQUIRE(rows[1].featureIndex >= 0);
    REQUIRE(rows[2].featureIndex >= 0);
    REQUIRE(depthOf(t) == 3);
}

TEST("the random splitter on bins takes each border uniformly and never leaves a side empty (defect 1)", "[unit]")
{
    // Values 0..8, one bin each: the draw is uniform in [0, 8) and the split goes after bin floor(t).
    Dataset d;
    d.nRows    = 900;
    d.p        = 1;
    d.nClasses = 2;
    for (size_t i = 0; i < d.nRows; ++i)
    {
        d.x.push_back(double(i % 9));
        d.y.push_back(double((i / 9) % 2));
    }
    TreeParams params;
    params.nClasses             = 2;
    params.randomSplitter       = true;
    params.maxDepth             = 1;
    params.denseThreshold       = 0;
    const BinningParams binning = exactBinning();
    const std::vector<std::vector<uint16_t> > noBootstrap;
    std::vector<size_t> hits(8, 0);
    for (const auto & t : train<ClsCountRecord, baseCpu>(d, params, 4000, noBootstrap, 777, &binning))
    {
        REQUIRE(rowsOf(t)[0].featureIndex == 0);
        const double v = rowsOf(t)[0].featureValueOrResponse;
        REQUIRE((v >= 0 && v <= 7 && v == std::floor(v)));
        ++hits[size_t(v)];
    }
    REQUIRE(chiSquare(hits) < chiSquare7Dof999);
}

/// d with three more features: constant everywhere, constant where feature 0 is below 0, and two-valued.
Dataset withConstantFeatures(Dataset d)
{
    const size_t p = d.p + 3;
    std::vector<double> x(d.nRows * p);
    for (size_t i = 0; i < d.nRows; ++i)
    {
        const double * row = d.x.data() + i * d.p;
        std::copy(row, row + d.p, x.data() + i * p);
        x[i * p + d.p]     = 3;
        x[i * p + d.p + 1] = row[0] < 0 ? 1 : double(i % 13);
        x[i * p + d.p + 2] = double(i % 2);
    }
    d.x = std::move(x);
    d.p = p;
    if (!d.categorical.empty()) d.categorical.resize(p, false);
    return d;
}

TEST("fused partitions and constant lists do not change the trees", "[unit]")
{
    constexpr size_t nTrees = 4;
    Dataset c               = withConstantFeatures(classificationData(20000, 5, 3, 121));
    const Dataset low       = withConstantFeatures(lowCardinalityData(20000, 0, 122));
    Dataset r               = withConstantFeatures(regressionData(20000, 5, 123));
    for (size_t i = 0; i < c.nRows; i += 41) c.x[i * c.p + 1] = std::numeric_limits<double>::quiet_NaN();
    const auto counts = bootstrapCounts(nTrees, 20000, 124);
    BinningParams binning;
    binning.maxBins           = 32;
    const BinningParams exact = exactBinning();
    for (bool random : { false, true })
    {
        for (size_t maxLeaves : { size_t(0), size_t(60) })
        {
            TreeParams params;
            params.featuresPerNode = 2;
            params.randomSplitter  = random;
            params.maxLeafNodes    = maxLeaves;
            params.maxBins         = 32;
            params.chunkRows       = 1000;
            std::vector<EmittedTree> refC, refL, refR;
            for (int mode = 0; mode < 4; ++mode)
            {
                params.fusePartition  = mode & 1;
                params.carryConstants = mode & 2;
                // Integer statistics make chunked sums exact; the continuous regression runs in one chunk.
                params.nClasses  = 3;
                params.chunkRows = 1000;
                auto tc          = train<ClsCountRecord, baseCpu>(c, params, nTrees, counts, 777, &binning);
                params.nClasses  = 0;
                auto tl          = train<RegCountRecord<double>, baseCpu>(low, params, nTrees, counts, 777, &exact);
                params.chunkRows = 0;
                auto tr          = train<RegCountRecord<double>, baseCpu>(r, params, nTrees, counts, 777, &binning);
                if (!mode)
                {
                    refC = std::move(tc);
                    refL = std::move(tl);
                    refR = std::move(tr);
                    for (size_t t = 0; t < nTrees; ++t)
                    {
                        requireConsistent(c, refC[t], counts[t]);
                        requireConsistent(low, refL[t], counts[t]);
                        requireConsistent(r, refR[t], counts[t]);
                    }
                    continue;
                }
                for (size_t t = 0; t < nTrees; ++t)
                {
                    requireSameTables(refC[t], tc[t]);
                    requireSameTables(refL[t], tl[t]);
                    requireSameTables(refR[t], tr[t]);
                }
            }
        }
    }
}

} // namespace daal::algorithms::dtrees::engine::internal::test
