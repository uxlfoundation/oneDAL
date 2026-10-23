/* file: df_forest_trainer.i */
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
//  Implementation of trainForest (df_forest_trainer.h). Included by the
//  classification and regression training kernels.
//--
*/

#ifndef __DF_FOREST_TRAINER_I__
#define __DF_FOREST_TRAINER_I__

#include <limits>

#include "algorithms/engines/philox4x32x10/philox4x32x10.h"
#include "src/algorithms/dtrees/engine/node_rng.h"
#include "src/algorithms/dtrees/engine/numa_alloc.h"
#include "src/algorithms/dtrees/engine/quantile_binner.h"
#include "src/algorithms/dtrees/engine/response_set.h"
#include "src/algorithms/dtrees/engine/tree_builder.h"
#include "src/algorithms/dtrees/forest/df_forest_trainer.h"
#include "src/algorithms/engines/engine_types_internal.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_math.h"
#include "src/externals/service_rng.h"
#include "src/services/service_environment.h"
#include "src/threading/threading.h"

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace training
{
namespace internal
{
namespace trainer
{
namespace eng = daal::algorithms::dtrees::engine::internal;
using services::internal::TArray;
using engines::internal::ParallelizationTechnique;

// Rows per block of the passes over the table rows: bootstrap draws and OOB evaluation.
constexpr size_t blockRows = eng::tableBlockRows;

// Classes a ClsCountRecord addresses (u16 class).
constexpr size_t maxClasses = size_t(1) << 16;

// Exclusive bound of the two 31-bit draws that make a tree's seed.
constexpr int seedDrawRange = 0x7fffffff;

inline bool isPhilox(const engines::EnginePtr & e)
{
    using engines::philox4x32x10::Batch;
    using engines::philox4x32x10::defaultDense;
    return dynamic_cast<const Batch<float, defaultDense> *>(e.get()) || dynamic_cast<const Batch<double, defaultDense> *>(e.get());
}

/**
 *  \brief Engine streams of a fit (design 4.13)
 *
 *  One stream per tree, then, for engines other than philox4x32x10, one per thread for node draws. Tree
 *  streams hold treeBudget draws each (two for the seed, the bootstrap, the MDA permutations) and thread
 *  streams threadBudget, upper bounds that keep the regions disjoint.
 */
template <CpuType cpu>
class ForestStreams
{
public:
    services::Status init(const engines::EnginePtr & base, size_t nTrees, size_t treeBudget, size_t nThreads, size_t threadBudget)
    {
        _base     = base;
        _nTrees   = nTrees;
        _nThreads = nThreads;
        auto impl = dynamic_cast<engines::internal::BatchBaseImpl *>(base.get());
        DAAL_CHECK(impl, services::ErrorEngineNotSupported);
        const ParallelizationTechnique order[] = { engines::internal::family, engines::internal::skipahead, engines::internal::leapfrog };
        bool supported                         = false;
        for (ParallelizationTechnique t : order)
        {
            if (!impl->hasSupport(t)) continue;
            _technique = t;
            supported  = true;
            break;
        }
        DAAL_CHECK(supported, services::ErrorEngineNotSupported);

        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nTrees, treeBudget);
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nThreads, threadBudget);
        const size_t treeRegion = nTrees * treeBudget;
        DAAL_OVERFLOW_CHECK_BY_ADDING(size_t, treeRegion, nThreads * threadBudget);
        _total     = treeRegion + nThreads * threadBudget;
        _maxBudget = treeBudget > threadBudget ? treeBudget : threadBudget;

        const size_t n = nTrees + nThreads;
        engines::internal::Params<cpu> params(n);
        DAAL_CHECK_MALLOC(params.nSkip.get());
        for (size_t i = 0; i < nTrees; ++i) params.nSkip[i] = i * treeBudget;
        for (size_t j = 0; j < nThreads; ++j) params.nSkip[nTrees + j] = treeRegion + j * threadBudget;
        DAAL_CHECK_MALLOC(_engines.reset(n) && _states.reset(n));
        services::Status s;
        _collection.reset(new engines::internal::EnginesCollection<cpu>(base, _technique, params, _engines, &s));
        DAAL_CHECK_MALLOC(_collection.get());
        DAAL_CHECK_STATUS_VAR(s);
        for (size_t i = 0; i < n; ++i)
        {
            auto streamImpl = dynamic_cast<engines::internal::BatchBaseImpl *>(_engines[i].get());
            DAAL_CHECK(streamImpl, services::ErrorEngineNotSupported);
            _states[i] = streamImpl->getState();
        }
        return s;
    }

    void * tree(size_t i) const { return _states[i]; }
    void * thread(size_t j) const { return _states[_nTrees + j]; }
    size_t nThreads() const { return _nThreads; }

    /// The engine after the fit: past every stream region, so a later fit on it reads none of this fit's draws.
    engines::EnginePtr updated()
    {
        if (_technique == engines::internal::family)
        {
            TArray<size_t, cpu> unused(_nTrees + _nThreads);
            return _collection->getUpdatedEngine(_base, _engines, unused);
        }
        // Leapfrog streams interleave, so the regions end at the stream count times the largest budget.
        engines::EnginePtr e = _base->clone();
        const size_t skip    = _technique == engines::internal::skipahead ? _total : (_nTrees + _nThreads) * _maxBudget;
        if (!e || !e->skipAhead(skip)) return engines::EnginePtr();
        return e;
    }

private:
    engines::EnginePtr _base;
    ParallelizationTechnique _technique = engines::internal::family;
    size_t _nTrees                      = 0;
    size_t _nThreads                    = 0;
    size_t _total                       = 0;
    size_t _maxBudget                   = 0;
    TArray<engines::EnginePtr, cpu> _engines;
    TArray<void *, cpu> _states;
    services::SharedPtr<engines::internal::EnginesCollection<cpu> > _collection;
};

/// What the tree engine needs per tree and returns (dtrees/engine/tree_builder.h).
template <typename algorithmFPType, typename Record, CpuType cpu>
class ForestInputs : public eng::TreeInputs<Record, cpu>
{
public:
    struct Setup
    {
        const eng::ResponseSource<algorithmFPType, cpu> * src = nullptr;
        ForestStreams<cpu> * streams                          = nullptr;
        dtrees::internal::ModelImpl * model                   = nullptr;
        size_t nRows                                          = 0;
        size_t nFeatures                                      = 0;
        size_t nSamples                                       = 0; // bootstrap draws, or leading rows without bootstrap
        bool bootstrap                                        = true;
        uint64_t * oobBits                                    = nullptr; // per tree, a bit per row with no draw; null when not needed
        double * treeWeight                                   = nullptr; // per tree, the in-bag weight
        double * mdi                                          = nullptr; // per tree and feature; null when not needed
    };

    explicit ForestInputs(const Setup & setup)
        : _s(setup), _words((setup.nRows + 63) / 64), _threadStream([this]() -> void * { return nextThreadStream(); })
    {
        _nextThread.set(0);
    }

    services::Status prepareTree(size_t iTree, int numaNode, eng::ResponseSet<Record, cpu> & records, uint64_t & treeSeed) override
    {
        void * state = _s.streams->tree(iTree);
        daal::internal::RNGsInst<int, cpu> rng;
        int seed[2] = { 0, 0 };
        DAAL_CHECK(!rng.uniform(2, seed, state, 0, seedDrawRange), services::ErrorIncorrectErrorcodeFromGenerator);
        treeSeed = eng::mix64<cpu>((uint64_t(uint32_t(seed[0])) << 31) | uint64_t(uint32_t(seed[1])));

        TArray<uint16_t, cpu> counts(_s.nRows);
        DAAL_CHECK_MALLOC(counts.get());
        uint16_t * c = counts.get();
        if (_s.bootstrap)
        {
            for (size_t r = 0; r < _s.nRows; ++r) c[r] = 0;
            TArray<int, cpu> draws(blockRows);
            DAAL_CHECK_MALLOC(draws.get());
            for (size_t done = 0; done < _s.nSamples; done += blockRows)
            {
                const size_t m = _s.nSamples - done < blockRows ? _s.nSamples - done : blockRows;
                DAAL_CHECK(!rng.uniform(m, draws.get(), state, 0, int(_s.nRows)), services::ErrorIncorrectErrorcodeFromGenerator);
                for (size_t i = 0; i < m; ++i)
                {
                    uint16_t & v = c[size_t(draws[i])];
                    DAAL_CHECK(v < 0xffff, services::ErrorIncorrectParameter); // a row drawn 65535 times
                    ++v;
                }
            }
        }
        else
        {
            for (size_t r = 0; r < _s.nRows; ++r) c[r] = r < _s.nSamples ? 1 : 0;
        }
        if (_s.oobBits)
        {
            uint64_t * bits = _s.oobBits + iTree * _words;
            for (size_t i = 0; i < _words; ++i) bits[i] = 0;
            for (size_t r = 0; r < _s.nRows; ++r)
                if (!c[r]) bits[r >> 6] |= uint64_t(1) << (r & 63);
        }
        double total         = 0;
        services::Status s   = records.fill(c, *_s.src, numaNode, total);
        _s.treeWeight[iTree] = total;
        return s;
    }

    services::Status emitTree(size_t iTree, eng::EmittedTree & tree) override
    {
        DAAL_CHECK_MALLOC(_s.model->setTree(iTree, tree.nodes, tree.impurity, tree.nodeSamples, tree.probabilities));
        if (!_s.mdi) return services::Status();
        // MDI of the documentation: each split's impurity decrease weighted by its share of the tree's weight.
        const double w = _s.treeWeight[iTree];
        double * dst   = _s.mdi + iTree * _s.nFeatures;
        for (size_t f = 0; f < _s.nFeatures; ++f) dst[f] = w > 0 ? tree.featureImportance[f] / w : 0;
        return services::Status();
    }

    void * nodeStream() override { return _threadStream.local(); }

private:
    void * nextThreadStream()
    {
        const size_t j = _nextThread.inc() - 1;
        return j < _s.streams->nThreads() ? _s.streams->thread(j) : nullptr;
    }

    Setup _s;
    size_t _words;
    services::Atomic<size_t> _nextThread;
    daal::tls<void *> _threadStream;
};

/// The leaf of a tree table that row x reaches, routing as the prediction kernels do; value(f) reads feature f.
template <typename Value>
size_t leafOf(const dtrees::internal::DecisionTreeNode * nodes, const bool * categorical, const Value & value)
{
    size_t r = 0;
    while (nodes[r].featureIndex >= 0)
    {
        const size_t f   = size_t(nodes[r].featureIndex);
        const double v   = value(f);
        const bool right = categorical[f] ? !(v == nodes[r].featureValueOrResponse) : v > nodes[r].featureValueOrResponse;
        r                = nodes[r].leftIndexOrClass + (right ? 1 : 0);
    }
    return r;
}

/**
 *  \brief OOB error and MDA importance, after all trees are built, one tree at a time in tree order
 *
 *  Rows are evaluated in parallel within a tree, each row once per tree, and sums are taken over fixed blocks
 *  in block order, so the results do not depend on the thread count. MDA permutations continue each tree's
 *  stream after its bootstrap draws.
 */
template <typename algorithmFPType, CpuType cpu>
class OutOfBag
{
public:
    OutOfBag(const algorithmFPType * x, const eng::ResponseSource<algorithmFPType, cpu> & src, const bool * categorical, size_t nRows,
             size_t nFeatures, size_t nClasses)
        : _x(x), _src(src), _categorical(categorical), _nRows(nRows), _p(nFeatures), _nClasses(nClasses), _words((nRows + 63) / 64)
    {}

    services::Status allocate(bool predictions, bool mda)
    {
        _predictions = predictions;
        if (predictions)
        {
            if (_nClasses)
            {
                DAAL_CHECK_MALLOC(_votes.reset(_nRows * _nClasses));
            }
            else
            {
                DAAL_CHECK_MALLOC(_sums.reset(_nRows) && _counts.reset(_nRows));
            }
            for (size_t i = 0; i < _votes.size(); ++i) _votes[i] = 0;
            for (size_t i = 0; i < _sums.size(); ++i) _sums[i] = 0;
            for (size_t i = 0; i < _counts.size(); ++i) _counts[i] = 0;
        }
        if (mda) DAAL_CHECK_MALLOC(_mdaMean.reset(_p) && _mdaM2.reset(_p));
        for (size_t i = 0; i < _mdaMean.size(); ++i) _mdaMean[i] = _mdaM2[i] = 0;
        DAAL_CHECK_MALLOC(_rows.reset(_nRows));
        return services::Status();
    }

    services::Status addTree(const dtrees::internal::DecisionTreeNode * nodes, const uint64_t * oobBits, void * stream)
    {
        size_t n = 0;
        for (size_t r = 0; r < _nRows; ++r)
            if ((oobBits[r >> 6] >> (r & 63)) & 1) _rows[n++] = uint32_t(r);
        if (!n) return services::Status();
        const uint32_t * rows = _rows.get();
        const size_t nBlocks  = (n + blockRows - 1) / blockRows;
        auto rowValue         = [&](uint32_t row) { return [=](size_t f) { return double(_x[size_t(row) * _p + f]); }; };

        if (_predictions)
        {
            daal::threader_for(nBlocks, 1, [&](size_t b) {
                const size_t end = (b + 1) * blockRows < n ? (b + 1) * blockRows : n;
                for (size_t i = b * blockRows; i < end; ++i)
                {
                    const size_t leaf = leafOf(nodes, _categorical, rowValue(rows[i]));
                    if (_nClasses)
                    {
                        _votes[size_t(rows[i]) * _nClasses + nodes[leaf].leftIndexOrClass] += 1;
                    }
                    else
                    {
                        _sums[rows[i]] += nodes[leaf].featureValueOrResponse;
                        _counts[rows[i]] += 1;
                    }
                }
            });
        }
        if (!_mdaMean.get()) return services::Status();

        // MDA: the increase of the tree's OOB error when feature j takes the values of a permutation of the OOB rows.
        TArray<double, cpu> partial(nBlocks);
        TArray<double, cpu> u(n);
        TArray<uint32_t, cpu> perm(n);
        DAAL_CHECK_MALLOC(partial.get() && u.get() && perm.get());
        auto meanError = [&](size_t permuted) {
            daal::threader_for(nBlocks, 1, [&](size_t b) {
                const size_t end = (b + 1) * blockRows < n ? (b + 1) * blockRows : n;
                double sum       = 0;
                for (size_t i = b * blockRows; i < end; ++i)
                {
                    const size_t row  = rows[i];
                    const size_t from = rows[perm[i]];
                    const size_t leaf = leafOf(nodes, _categorical, [&](size_t f) { return double(_x[(f == permuted ? from : row) * _p + f]); });
                    sum += error(nodes[leaf], row);
                }
                partial[b] = sum;
            });
            double sum = 0;
            for (size_t b = 0; b < nBlocks; ++b) sum += partial[b];
            return sum / double(n);
        };
        for (size_t i = 0; i < n; ++i) perm[i] = uint32_t(i);
        const double base = meanError(_p); // no feature permuted
        ++_mdaTrees;
        daal::internal::RNGsInst<double, cpu> rng;
        for (size_t j = 0; j < _p; ++j)
        {
            // Fisher-Yates from n uniform draws: position i swaps with one of [0, i].
            DAAL_CHECK(!rng.uniform(n, u.get(), stream, 0.0, 1.0), services::ErrorIncorrectErrorcodeFromGenerator);
            for (size_t i = 0; i < n; ++i) perm[i] = uint32_t(i);
            for (size_t i = n - 1; i > 0; --i)
            {
                size_t k         = size_t(u[i] * double(i + 1));
                k                = k > i ? i : k;
                const uint32_t t = perm[i];
                perm[i]          = perm[k];
                perm[k]          = t;
            }
            const double diff  = meanError(j) - base;
            const double delta = diff - _mdaMean[j];
            _mdaMean[j] += delta / double(_mdaTrees);
            _mdaM2[j] += delta * (diff - _mdaMean[j]);
        }
        return services::Status();
    }

    /// OOB error tables, as legacy computes them: classification by majority vote over the trees whose OOB set
    /// holds the row, regression by the mean of their predictions.
    services::Status finish(const ForestResults & res) const
    {
        const size_t n = _nRows;
        WriteOnlyRows<algorithmFPType, cpu> err(res.oobError, 0, 1);
        WriteOnlyRows<algorithmFPType, cpu> perObs(res.oobErrorPerObservation, 0, n);
        WriteOnlyRows<algorithmFPType, cpu> accuracy(res.oobErrorAccuracy, 0, 1);
        WriteOnlyRows<algorithmFPType, cpu> r2(res.oobErrorR2, 0, 1);
        WriteOnlyRows<algorithmFPType, cpu> decision(res.oobErrorDecisionFunction, 0, n);
        WriteOnlyRows<algorithmFPType, cpu> prediction(res.oobErrorPrediction, 0, n);
        if (res.oobError) DAAL_CHECK_BLOCK_STATUS(err);
        if (res.oobErrorPerObservation) DAAL_CHECK_BLOCK_STATUS(perObs);
        if (res.oobErrorAccuracy) DAAL_CHECK_BLOCK_STATUS(accuracy);
        if (res.oobErrorR2) DAAL_CHECK_BLOCK_STATUS(r2);
        if (res.oobErrorDecisionFunction) DAAL_CHECK_BLOCK_STATUS(decision);
        if (res.oobErrorPrediction) DAAL_CHECK_BLOCK_STATUS(prediction);
        size_t nPredicted = 0;
        double sumError = 0, yMean = 0, sumSquares = 0;
        if (!_nClasses)
        {
            const algorithmFPType * y = _src.responses(0);
            for (size_t i = 0; i < n; ++i) yMean += double(y[i]);
            yMean /= double(n);
        }
        for (size_t i = 0; i < n; ++i)
        {
            if (_nClasses)
            {
                const uint32_t * v = _votes.get() + i * _nClasses;
                size_t best        = 0;
                uint32_t total     = v[0];
                for (size_t c = 1; c < _nClasses; ++c)
                {
                    total += v[c];
                    best = v[c] > v[best] ? c : best;
                }
                if (decision.get())
                    for (size_t c = 0; c < _nClasses; ++c)
                        decision.get()[i * _nClasses + c] = total ? algorithmFPType(double(v[c]) / double(total)) : 0;
                if (!total)
                {
                    if (perObs.get()) perObs.get()[i] = -1; // in no tree's OOB set
                    continue;
                }
                const bool wrong = best != _src.classes(0)[i];
                ++nPredicted;
                sumError += wrong;
                if (perObs.get()) perObs.get()[i] = algorithmFPType(wrong);
            }
            else
            {
                if (!_counts[i])
                {
                    if (perObs.get()) perObs.get()[i] = -1;
                    if (prediction.get()) prediction.get()[i] = 0;
                    continue;
                }
                const double y    = double(_src.responses(0)[i]);
                const double mean = _sums[i] / double(_counts[i]);
                const double e    = (mean - y) * (mean - y);
                ++nPredicted;
                sumError += e;
                sumSquares += (y - yMean) * (y - yMean);
                if (perObs.get()) perObs.get()[i] = algorithmFPType(e);
                if (prediction.get()) prediction.get()[i] = algorithmFPType(mean);
            }
        }
        if (err.get()) err.get()[0] = nPredicted ? algorithmFPType(sumError / double(nPredicted)) : 0;
        if (accuracy.get()) accuracy.get()[0] = nPredicted ? algorithmFPType(1 - sumError / double(nPredicted)) : 1;
        // Undefined without OOB predictions. Constant responses give a zero denominator; as in scikit-learn's
        // r2_score, R2 is then 1 for an exact fit and 0 otherwise.
        if (r2.get())
            r2.get()[0] = !nPredicted    ? std::numeric_limits<algorithmFPType>::quiet_NaN() :
                          sumSquares > 0 ? algorithmFPType(1 - sumError / sumSquares) :
                                           algorithmFPType(sumError > 0 ? 0 : 1);
        return services::Status();
    }

    /// MDA_Raw: the mean over trees; MDA_Scaled: the mean divided by its standard error over trees, as legacy.
    void mda(bool scaled, algorithmFPType * out) const
    {
        const double t = double(_mdaTrees);
        for (size_t j = 0; j < _p; ++j)
        {
            if (!scaled)
            {
                out[j] = algorithmFPType(_mdaMean[j]);
                continue;
            }
            const double se2 = _mdaTrees > 1 ? _mdaM2[j] / t / t : 0;
            out[j]           = se2 > 0 ? algorithmFPType(_mdaMean[j] / daal::internal::MathInst<double, cpu>::sSqrt(se2)) : 0;
        }
    }

private:
    double error(const dtrees::internal::DecisionTreeNode & leaf, size_t row) const
    {
        if (_nClasses) return double(leaf.leftIndexOrClass != _src.classes(0)[row]);
        const double d = leaf.featureValueOrResponse - double(_src.responses(0)[row]);
        return d * d;
    }

    const algorithmFPType * _x;
    const eng::ResponseSource<algorithmFPType, cpu> & _src;
    const bool * _categorical;
    size_t _nRows;
    size_t _p;
    size_t _nClasses;
    size_t _words;
    bool _predictions = false;
    size_t _mdaTrees  = 0;
    TArray<uint32_t, cpu> _votes;
    TArray<double, cpu> _sums;
    TArray<uint32_t, cpu> _counts;
    TArray<double, cpu> _mdaMean;
    TArray<double, cpu> _mdaM2;
    TArray<uint32_t, cpu> _rows;
};

/// Features drawn per node when par.featuresPerNode is 0, as legacy: sqrt(p) for classification, p / 3 for regression.
template <CpuType cpu>
size_t defaultFeaturesPerNode(size_t p, bool classification)
{
    const size_t k = classification ? size_t(daal::internal::MathInst<double, cpu>::sSqrt(double(p))) : p / 3;
    return k < 1 ? 1 : k;
}

// Bytes of the largest record layout (RegWeightRecord<double>), two buffers of which a subtree reads per row.
constexpr size_t maxRecordBytes = 24;
// L2 size taken when the CPU does not report one: SPR's 2 MB per core.
constexpr size_t defaultL2Bytes = size_t(2) << 20;

/// The serial-subtree cutoff: the rows whose records (both buffers), bin row and X row fit a core's L2 (design 8, step 15).
template <typename algorithmFPType, CpuType cpu>
size_t serialSubtreeRows(size_t p, size_t binRowBytes)
{
    const size_t l2     = services::internal::getL2CacheSize();
    const size_t perRow = 2 * maxRecordBytes + binRowBytes + p * sizeof(algorithmFPType);
    return (l2 ? l2 : defaultL2Bytes) / perRow;
}

template <typename algorithmFPType, typename Record, CpuType cpu>
services::Status train(const eng::TreeData<algorithmFPType, cpu> & data, const eng::TreeParams & params, size_t nTrees,
                       const typename ForestInputs<algorithmFPType, Record, cpu>::Setup & setup)
{
    ForestInputs<algorithmFPType, Record, cpu> inputs(setup);
    return eng::trainTrees<algorithmFPType, Record, cpu>(data, params, nTrees, inputs);
}

} // namespace trainer

template <typename algorithmFPType, CpuType cpu>
services::Status trainForest(const data_management::NumericTable & x, const data_management::NumericTable & y,
                             const data_management::NumericTable * w, const Parameter & par, size_t nClasses, bool hist,
                             const decision_forest::internal::EngineHyperparameters & hp, dtrees::internal::ModelImpl & model, ForestResults & res)
{
    namespace eng = daal::algorithms::dtrees::engine::internal;
    using trainer::TArray;
    const size_t nRows  = x.getNumberOfRows();
    const size_t p      = x.getNumberOfColumns();
    const size_t nTrees = par.nTrees;
    DAAL_CHECK(nRows && p && nTrees, services::ErrorIncorrectParameter);
    DAAL_CHECK_STATUS_VAR(eng::checkRowCount(nRows));
    DAAL_CHECK(nClasses <= trainer::maxClasses, services::ErrorIncorrectNumberOfClasses);
    DAAL_CHECK(par.engine, services::ErrorEngineNotSupported);
    DAAL_CHECK_MALLOC(model.resize(nTrees));
    const bool replicate = !par.memorySavingMode;

    // X row-major per NUMA node, responses and normalized weights, feature types.
    eng::NumaReplicated<algorithmFPType, cpu> xr;
    services::Status s = eng::copyTableRows<algorithmFPType, cpu>(x, replicate, xr);
    DAAL_CHECK_STATUS_VAR(s);
    eng::ResponseSource<algorithmFPType, cpu> src;
    s = src.build(y, w, nClasses > 0, replicate);
    DAAL_CHECK_STATUS_VAR(s);
    if (nClasses)
        for (size_t i = 0; i < nRows; ++i) DAAL_CHECK(src.classes(0)[i] < nClasses, services::ErrorIncorrectClassLabels);
    TArray<bool, cpu> categorical(p);
    DAAL_CHECK_MALLOC(categorical.get());
    for (size_t f = 0; f < p; ++f) categorical[f] = x.getFeatureType(f) == data_management::features::DAAL_CATEGORICAL;

    // The binning sample reads the engine first; the streams are derived from it as the sample left it (4.13).
    engines::EnginePtr base = par.engine->clone();
    auto baseImpl           = dynamic_cast<engines::internal::BatchBaseImpl *>(base.get());
    DAAL_CHECK(baseImpl, services::ErrorEngineNotSupported);
    eng::BinMatrix<cpu> bins;
    bool binned = false;
    if (hist)
    {
        eng::BinningParams bp;
        bp.maxBins    = par.maxBins;
        bp.minBinSize = par.minBinSize;
        bp.strategy   = par.binningStrategy;
        bp.replicate  = replicate;
        s             = eng::QuantileBinner<algorithmFPType, cpu>::build(x, bp, baseImpl->getState(), bins, binned);
        DAAL_CHECK_STATUS_VAR(s);
    }

    // Node draws by counter where the engine is philox4x32x10 and service_rng can address it; otherwise per-thread streams.
    eng::CounterRng<cpu> probe;
    const bool counter      = trainer::isPhilox(par.engine) && probe.init(1).ok();
    const bool mda          = par.varImportance == MDA_Raw || par.varImportance == MDA_Scaled;
    const bool mdi          = par.varImportance == MDI;
    const uint64_t oobFlags = computeOutOfBagError | computeOutOfBagErrorPerObservation | computeOutOfBagErrorAccuracy | computeOutOfBagErrorR2
                              | computeOutOfBagErrorDecisionFunction | computeOutOfBagErrorPrediction;
    const bool predictions = (par.resultsToCompute & oobFlags) != 0;
    // Bootstrap draws may exceed the rows (fraction above 1); without bootstrap a tree takes at most every row.
    size_t nSamples = size_t(par.observationsPerTreeFraction * double(nRows));
    DAAL_CHECK(nSamples >= 1, services::ErrorIncorrectParameter);
    if (!par.bootstrap && nSamples > nRows) nSamples = nRows;
    const size_t k = par.featuresPerNode ? par.featuresPerNode : trainer::defaultFeaturesPerNode<cpu>(p, nClasses > 0);

    // Stream budgets, in engine elements, counting two per uniform double: a tree's seed, bootstrap and MDA draws;
    // a thread's p + k draws per node for up to 2n nodes per tree. The spare thread stream serves calls from outside
    // the arenas.
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, 2 * p, nRows);
    const size_t mdaDraws = mda ? 2 * p * nRows : 0;
    size_t treeBudget     = 2 + (par.bootstrap ? nSamples : 0);
    DAAL_OVERFLOW_CHECK_BY_ADDING(size_t, treeBudget, mdaDraws);
    treeBudget += mdaDraws;
    const size_t nThreads = counter ? 0 : size_t(threader_get_max_threads_number()) + 1;
    size_t threadBudget   = 0;
    if (!counter)
    {
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nTrees, 2 * nRows);
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nTrees * 2 * nRows, 2 * (p + k));
        threadBudget = nTrees * 2 * nRows * 2 * (p + k);
    }
    trainer::ForestStreams<cpu> streams;
    s = streams.init(base, nTrees, treeBudget, nThreads, threadBudget);
    DAAL_CHECK_STATUS_VAR(s);

    // Thresholds as legacy: weights are the normalized user weights over all rows, or 1 per row.
    double totalWeight = double(nRows);
    if (src.weighted())
    {
        totalWeight = 0;
        for (size_t i = 0; i < nRows; ++i) totalWeight += double(src.weights(0)[i]);
    }
    const double eps = services::internal::EpsilonVal<double>::get();
    eng::TreeParams tp;
    tp.nClasses            = nClasses;
    tp.featuresPerNode     = k;
    tp.maxDepth            = par.maxTreeDepth;
    tp.maxLeafNodes        = par.maxLeafNodes;
    tp.minLeafRecords      = par.minObservationsInLeafNode > 1 ? par.minObservationsInLeafNode : 1;
    tp.minSplitRecords     = par.minObservationsInSplitNode > 2 ? par.minObservationsInSplitNode : 2;
    tp.minSplitRecords     = tp.minSplitRecords > 2 * tp.minLeafRecords ? tp.minSplitRecords : 2 * tp.minLeafRecords;
    tp.sizesInDraws        = true; // legacy's count; distinct rows (sklearn) is a follow-up (design 8)
    tp.minLeafWeight       = par.minWeightFractionInLeafNode * totalWeight;
    tp.minWeightedDecrease = par.minImpurityDecreaseInSplitNode * totalWeight - eps * totalWeight;
    tp.impurityThreshold   = par.impurityThreshold > eps ? par.impurityThreshold : eps;
    tp.randomSplitter      = par.splitter == decision_forest::training::random;
    tp.counterDraws        = counter;
    tp.treesInFlight       = size_t(hp.treesInFlight);
    tp.serialSubtreeRows =
        hp.serialSubtreeRows >= 0 ? size_t(hp.serialSubtreeRows) : trainer::serialSubtreeRows<algorithmFPType, cpu>(p, binned ? bins.rowStride() : 0);
    tp.denseThreshold = size_t(hp.denseThreshold);
    tp.maxBins        = par.maxBins;
    tp.chunkRows      = size_t(hp.chunkRows);
    tp.maxChunks      = size_t(hp.maxChunks);

    eng::TreeData<algorithmFPType, cpu> data;
    data.x           = &xr;
    data.responses   = &src;
    data.categorical = categorical.get();
    data.bins        = binned ? &bins : nullptr;
    data.nRows       = nRows;
    data.nFeatures   = p;

    const size_t words = (nRows + 63) / 64;
    TArray<uint64_t, cpu> oobBits;
    TArray<double, cpu> treeWeight(nTrees);
    TArray<double, cpu> mdiPerTree;
    DAAL_CHECK_MALLOC(treeWeight.get());
    if (predictions || mda) DAAL_CHECK_MALLOC(oobBits.reset(nTrees * words));
    if (mdi && res.varImportance) DAAL_CHECK_MALLOC(mdiPerTree.reset(nTrees * p));

    // The tree engine, by record layout (design 4.10).
    if (nClasses)
    {
        if (src.weighted())
        {
            using Inputs = trainer::ForestInputs<algorithmFPType, eng::ClsWeightRecord<algorithmFPType>, cpu>;
            typename Inputs::Setup setup { &src,          &streams,      &model,           nRows,           p, nSamples,
                                           par.bootstrap, oobBits.get(), treeWeight.get(), mdiPerTree.get() };
            s = trainer::train<algorithmFPType, eng::ClsWeightRecord<algorithmFPType>, cpu>(data, tp, nTrees, setup);
        }
        else
        {
            using Inputs = trainer::ForestInputs<algorithmFPType, eng::ClsCountRecord, cpu>;
            typename Inputs::Setup setup { &src,          &streams,      &model,           nRows,           p, nSamples,
                                           par.bootstrap, oobBits.get(), treeWeight.get(), mdiPerTree.get() };
            s = trainer::train<algorithmFPType, eng::ClsCountRecord, cpu>(data, tp, nTrees, setup);
        }
    }
    else
    {
        if (src.weighted())
        {
            using Inputs = trainer::ForestInputs<algorithmFPType, eng::RegWeightRecord<algorithmFPType>, cpu>;
            typename Inputs::Setup setup { &src,          &streams,      &model,           nRows,           p, nSamples,
                                           par.bootstrap, oobBits.get(), treeWeight.get(), mdiPerTree.get() };
            s = trainer::train<algorithmFPType, eng::RegWeightRecord<algorithmFPType>, cpu>(data, tp, nTrees, setup);
        }
        else
        {
            using Inputs = trainer::ForestInputs<algorithmFPType, eng::RegCountRecord<algorithmFPType>, cpu>;
            typename Inputs::Setup setup { &src,          &streams,      &model,           nRows,           p, nSamples,
                                           par.bootstrap, oobBits.get(), treeWeight.get(), mdiPerTree.get() };
            s = trainer::train<algorithmFPType, eng::RegCountRecord<algorithmFPType>, cpu>(data, tp, nTrees, setup);
        }
    }
    DAAL_CHECK_STATUS_VAR(s);
    DAAL_CHECK(model.size() == nTrees, services::ErrorMemoryAllocationFailed);

    // OOB error and MDA, in tree order.
    if (predictions || mda)
    {
        trainer::OutOfBag<algorithmFPType, cpu> oob(xr.data(0), src, categorical.get(), nRows, p, nClasses);
        DAAL_CHECK_STATUS_VAR(oob.allocate(predictions, mda && res.varImportance));
        for (size_t t = 0; t < nTrees; ++t)
        {
            const auto * nodes = static_cast<const dtrees::internal::DecisionTreeNode *>(model.at(t)->getArray());
            s                  = oob.addTree(nodes, oobBits.get() + t * words, streams.tree(t));
            DAAL_CHECK_STATUS_VAR(s);
        }
        if (predictions) DAAL_CHECK_STATUS_VAR(oob.finish(res));
        if (mda && res.varImportance)
        {
            WriteOnlyRows<algorithmFPType, cpu> out(res.varImportance, 0, 1);
            DAAL_CHECK_BLOCK_STATUS(out);
            oob.mda(par.varImportance == MDA_Scaled, out.get());
        }
    }
    if (mdi && res.varImportance)
    {
        // The mean over trees, summed in tree order.
        WriteOnlyRows<algorithmFPType, cpu> out(res.varImportance, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(out);
        for (size_t f = 0; f < p; ++f)
        {
            double sum = 0;
            for (size_t t = 0; t < nTrees; ++t) sum += mdiPerTree[t * p + f];
            out.get()[f] = algorithmFPType(sum / double(nTrees));
        }
    }

    res.updatedEngine = streams.updated();
    DAAL_CHECK(res.updatedEngine, services::ErrorEngineNotSupported);
    return s;
}

} // namespace internal
} // namespace training
} // namespace decision_forest
} // namespace algorithms
} // namespace daal

#endif
