/* file: df_train_dense_default_impl.i */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of auxiliary functions for decision forest train algorithms
//  (defaultDense) method.
//--
*/

#ifndef __DF_TRAIN_DENSE_DEFAULT_IMPL_I__
#define __DF_TRAIN_DENSE_DEFAULT_IMPL_I__

#include "src/algorithms/dtrees/dtrees_train_data_helper.i"
#include "src/threading/threading.h"
#include "src/algorithms/dtrees/dtrees_model_impl.h"
#include "src/algorithms/engines/engine_types_internal.h"
#include "src/algorithms/service_heap.h"
#include "src/services/service_defines.h"
#include "src/algorithms/distributions/uniform/uniform_kernel.h"
#include "src/algorithms/dtrees/forest/df_hyperparameter_impl.h"

using namespace daal::algorithms::dtrees::training::internal;
using namespace daal::algorithms::internal;

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

//////////////////////////////////////////////////////////////////////////////////////////
// Service class, it uses to keep information about nodes
//////////////////////////////////////////////////////////////////////////////////////////
template <typename WorkItem, CpuType cpu>
class BinaryHeap
{
public:
    BinaryHeap(services::Status & s)
        : _capacity(1024),
          _count(0),
          _data(new WorkItem[_capacity]) { DAAL_CHECK_COND_ERROR(_data, s, services::ErrorMemoryAllocationFailed) }

          BinaryHeap(const BinaryHeap &) = delete;

    ~BinaryHeap()
    {
        delete[] _data;
        _data = nullptr;
    }

    bool empty() const { return (_count == 0); }

    WorkItem & pop()
    {
        DAAL_ASSERT(!empty());

        popMaxHeap<cpu>(_data, _data + _count, [](const WorkItem & v1, const WorkItem & v2) -> bool { return v1.improvement < v2.improvement; });
        --_count;
        return _data[_count];
    }

    services::Status push(WorkItem & value)
    {
        if (_count == _capacity)
        {
            services::Status status = grow();
            DAAL_CHECK_STATUS_VAR(status)
        }
        DAAL_ASSERT(_count < _capacity);

        _data[_count++] = value;
        makeMaxHeap<cpu>(_data, _data + _count, [](const WorkItem & v1, const WorkItem & v2) -> bool { return v1.improvement < v2.improvement; });
        return services::Status();
    }

private:
    services::Status grow()
    {
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, _capacity, 2);
        const size_t newCapacity = _capacity * 2;
        DAAL_ASSERT(_count < newCapacity);
        WorkItem * const newData = new WorkItem[newCapacity];
        DAAL_CHECK_MALLOC(newData)
        for (size_t i = 0; i < _count; ++i)
        {
            newData[i] = _data[i];
        }
        delete[] _data;
        _data     = newData;
        _capacity = newCapacity;

        return services::Status();
    }

    size_t _capacity; // capacity of the heap
    size_t _count;    // counter of heap elements, the heap grows from left to right
    WorkItem * _data; // array of heap elements, max element is on the left
};

//////////////////////////////////////////////////////////////////////////////////////////
// Service structure, node split error & splitting status
//////////////////////////////////////////////////////////////////////////////////////////
struct NodeSplitResult
{
    services::Status status;
    bool bSplitSucceeded;
};

//////////////////////////////////////////////////////////////////////////////////////////
// Service structure, contains numeric tables to be calculated as result
//////////////////////////////////////////////////////////////////////////////////////////
struct ResultData
{
public:
    ResultData(const Parameter & par, NumericTable * _varImp, NumericTable * _oobError, NumericTable * _oobErrorPerObs,
               NumericTable * _oobErrorAccuracy, NumericTable * _oobErrorR2, NumericTable * _oobErrorDecisionFunction,
               NumericTable * _oobErrorPrediction)
    {
        if (par.varImportance != decision_forest::training::none) varImp = _varImp;
        if (par.resultsToCompute & decision_forest::training::computeOutOfBagError) oobError = _oobError;
        if (par.resultsToCompute & decision_forest::training::computeOutOfBagErrorAccuracy) oobErrorAccuracy = _oobErrorAccuracy;
        if (par.resultsToCompute & decision_forest::training::computeOutOfBagErrorR2) oobErrorR2 = _oobErrorR2;
        if (par.resultsToCompute & decision_forest::training::computeOutOfBagErrorPerObservation) oobErrorPerObs = _oobErrorPerObs;
        if (par.resultsToCompute & decision_forest::training::computeOutOfBagErrorDecisionFunction)
            oobErrorDecisionFunction = _oobErrorDecisionFunction;
        if (par.resultsToCompute & decision_forest::training::computeOutOfBagErrorPrediction) oobErrorPrediction = _oobErrorPrediction;
    }
    NumericTable * varImp                   = nullptr; //if needed then allocated outside kernel
    NumericTable * oobError                 = nullptr; //if needed then allocated outside kernel
    NumericTable * oobErrorAccuracy         = nullptr; //if needed then allocated outside kernel
    NumericTable * oobErrorR2               = nullptr; //if needed then allocated outside kernel
    NumericTable * oobErrorPerObs           = nullptr; //if needed then allocated outside kernel
    NumericTable * oobErrorDecisionFunction = nullptr; //if needed then allocated outside kernel
    NumericTable * oobErrorPrediction       = nullptr; //if needed then allocated outside kernel
    NumericTablePtr oobIndices;                        //if needed then allocated in kernel
    engines::EnginePtr updatedEngine;                  // engine updated after simulations
};

//////////////////////////////////////////////////////////////////////////////////////////
// Service structure, contains workset required for tree calculation in one thread
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class TreeThreadCtxBase
{
public:
    TreeThreadCtxBase(algorithmFPType * _varImp = nullptr) : varImp(_varImp), varImpVariance(nullptr), nTrees(0), oobBuf(nullptr) {}
    ~TreeThreadCtxBase()
    {
        if (varImpVariance) service_free<algorithmFPType, cpu>(varImpVariance);
        if (oobBuf) service_free<byte, cpu>(oobBuf);
    }
    void finalizeVarImp(training::VariableImportanceMode mode, size_t nVars);

protected:
    bool init(const decision_forest::training::Parameter & par, const NumericTable * x)
    {
        if (par.varImportance == training::MDA_Scaled)
        {
            varImpVariance = service_calloc<algorithmFPType, cpu>(x->getNumberOfColumns());
            DAAL_CHECK_STATUS_VAR(varImpVariance);
        }
        return true;
    }

    void reduceTo(training::VariableImportanceMode mode, TreeThreadCtxBase & other, size_t nVars, size_t nSamples) const
    {
        // Reduces tls variable importance results
        if (varImp)
        {
            if (mode == training::MDI)
            {
                for (size_t i = 0; i < nVars; ++i) other.varImp[i] += varImp[i];
            }
            else
            {
                const algorithmFPType div  = algorithmFPType(1) / algorithmFPType(other.nTrees + nTrees);
                const algorithmFPType n1   = algorithmFPType(nTrees) * div;
                const algorithmFPType n2   = algorithmFPType(other.nTrees) * div;
                const algorithmFPType div2 = algorithmFPType(nTrees * other.nTrees) * div;
                for (size_t i = 0; i < nVars; ++i)
                {
                    const algorithmFPType newM = varImp[i] * n1 + other.varImp[i] * n2;
                    if (varImpVariance)
                    {
                        const algorithmFPType deltaM = varImp[i] - other.varImp[i];
                        const algorithmFPType v1     = varImpVariance[i];
                        const algorithmFPType v2     = other.varImpVariance[i];
                        const algorithmFPType v      = v1 + v2 + deltaM * deltaM * div2;
                        other.varImpVariance[i]      = v;
                    }
                    other.varImp[i] = newM;
                }
            }
        }
        other.nTrees += nTrees;
    }

public:
    algorithmFPType * varImp;
    algorithmFPType * varImpVariance;
    size_t nTrees;
    byte * oobBuf;
};

//////////////////////////////////////////////////////////////////////////////////////////
// Finalizes calculation of variable importance results
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
void TreeThreadCtxBase<algorithmFPType, cpu>::finalizeVarImp(training::VariableImportanceMode mode, size_t nVars)
{
    if (mode == training::MDA_Scaled)
    {
        //average over all trees and scale by its variance
        if (nTrees > 1)
        {
            const algorithmFPType div = 1. / algorithmFPType(nTrees);
            for (size_t i = 0; i < nVars; ++i)
            {
                varImpVariance[i] *= div;
                if (isPositive<algorithmFPType, cpu>(varImpVariance[i]))
                    varImp[i] /= daal::internal::MathInst<algorithmFPType, cpu>::sSqrt(varImpVariance[i] * div);
            }
        }
        else
        {
            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t i = 0; i < nVars; ++i)
            {
                varImp[i] = 0;
            }
        }
    }
    else if (mode == training::MDI)
    {
        const algorithmFPType div = 1. / algorithmFPType(nTrees);
        //average over all trees
        for (size_t i = 0; i < nVars; ++i) varImp[i] *= div;
    }

#if ENABLE_VAR_IMP_NORMALIZATION
    algorithmFPType sum = 0;
    for (size_t i = 0; i < nVars; ++i) sum += varImp[i];
    //normalize by division to the sum of all values
    if (!isPositive<algorithmFPType, cpu>(sum))
    {
        algorithmFPType maxVal = 0;
        for (size_t i = 0; i < nVars; ++i)
        {
            const algorithmFPType val = daal::internal::MathInst<algorithmFPType, cpu>::sFabs(varImp[i]);
            maxVal                    = daal::internal::MathInst<algorithmFPType, cpu>::sMax(maxVal, val);
        }
        if (!isPositive<algorithmFPType, cpu>(maxVal)) return;
        const algorithmFPType div = 1. / maxVal;
        for (size_t i = 0; i < nVars; varImp[i++] *= div)
            ;
    }
    else
    {
        sum = 1. / sum;
        for (size_t i = 0; i < nVars; varImp[i++] *= sum)
            ;
    }
#endif
}

template <typename algorithmFPType, CpuType cpu, class Ctx>
Ctx * createTlsContext(const NumericTable * x, const Parameter & par, size_t nClasses)
{
    const size_t szVarImp = (par.varImportance == decision_forest::training::none ? 0 : x->getNumberOfColumns() * sizeof(algorithmFPType));
    size_t szCtx          = sizeof(Ctx);
    if (szCtx % sizeof(algorithmFPType)) szCtx = sizeof(algorithmFPType) * (1 + (szCtx / sizeof(algorithmFPType)));
    const size_t sz = szCtx + szVarImp;
    byte * ptr      = service_scalable_calloc<byte, cpu>(sz);
    Ctx * ctx       = new (ptr) Ctx();
    if (ctx)
    {
        if (szVarImp) ctx->varImp = (algorithmFPType *)(ptr + szCtx);
        if (!ctx->init(par, x, nClasses))
        {
            //allocation of extra data hasn't succeeded
            ctx->~Ctx();
            service_scalable_free<byte, cpu>(ptr);
            return nullptr;
        }
    }
    return ctx;
}

template <CpuType cpu>
services::Status selectParallelizationTechnique(const Parameter & par, engines::internal::ParallelizationTechnique & technique)
{
    auto engineImpl = dynamic_cast<engines::internal::BatchBaseImpl *>(par.engine.get());

    if (engineImpl == NULL)
    {
        return services::Status(services::ErrorNullResult);
    }

    engines::internal::ParallelizationTechnique techniques[] = { engines::internal::family, engines::internal::leapfrog,
                                                                 engines::internal::skipahead };

    for (auto & t : techniques)
    {
        if (engineImpl->hasSupport(t))
        {
            technique = t;
            return services::Status();
        }
    }
    return services::Status(ErrorEngineNotSupported);
}

//////////////////////////////////////////////////////////////////////////////////////////
// compute() implementation
//////////////////////////////////////////////////////////////////////////////////////////
template <CpuType cpu, typename IndexType, typename BinIndexType>
services::Status copyBinIndex(const size_t nRows, const size_t nCols, const IndexType * featureIndex,
                              TVector<BinIndexType, cpu, ScalableAllocator<cpu> > & binIndexVector, BinIndexType ** binIndex);

template <CpuType cpu, dtrees::internal::IndexedFeatures::IndexType, dtrees::internal::IndexedFeatures::IndexType>
services::Status copyBinIndex(const size_t nRows, const size_t nCols, const dtrees::internal::IndexedFeatures::IndexType * featureIndex,
                              TVector<dtrees::internal::IndexedFeatures::IndexType, cpu, ScalableAllocator<cpu> > & binIndexVector,
                              dtrees::internal::IndexedFeatures::IndexType ** binIndex)
{
    *binIndex = const_cast<dtrees::internal::IndexedFeatures::IndexType *>(featureIndex);
    return services::Status();
}

template <CpuType cpu, typename IndexType, typename BinIndexType>
services::Status copyBinIndex(const size_t nRows, const size_t nCols, const IndexType * featureIndex,
                              TVector<BinIndexType, cpu, ScalableAllocator<cpu> > & binIndexVector, BinIndexType ** binIndex)
{
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nRows, nCols);
    binIndexVector.resize(nRows * nCols);
    DAAL_CHECK(binIndexVector.get(), ErrorMemoryAllocationFailed);
    *binIndex = binIndexVector.get();

    const size_t nThreads    = threader_get_threads_number();
    const size_t nBlocks     = ((nThreads < nRows) ? nThreads : 1);
    const size_t sizeOfBlock = nRows / nBlocks + !!(nRows % nBlocks);

    daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock) {
        const size_t iStart = iBlock * sizeOfBlock;
        const size_t iEnd   = (((iBlock + 1) * sizeOfBlock > nRows) ? nRows : iStart + sizeOfBlock);

        for (size_t i = iStart; i < iEnd; ++i)
        {
            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t j = 0; j < nCols; ++j)
            {
                (*binIndex)[nRows * j + i] = static_cast<BinIndexType>(featureIndex[nRows * j + i]);
            }
        }
    });
    return services::Status();
}

template <typename algorithmFPType, typename BinIndexType, CpuType cpu, typename ModelType, typename TaskType, typename HyperparameterType>
services::Status computeImpl(HostAppIface * pHostApp, const NumericTable * x, const NumericTable * y, const NumericTable * w, ModelType & md,
                             ResultData & res, const Parameter & par, size_t nClasses, const dtrees::internal::FeatureTypes & featTypes,
                             const dtrees::internal::IndexedFeatures * indexedFeatures, const HyperparameterType * hyperparameter = nullptr)
{
    services::Status s;
    DAAL_CHECK(md.resize(par.nTrees), ErrorMemoryAllocationFailed);

    const size_t nRows = x->getNumberOfRows();
    const size_t nCols = x->getNumberOfColumns();
    TVector<BinIndexType, cpu, ScalableAllocator<cpu> > binIndexVector;
    BinIndexType * binIndex = nullptr;

    if (indexedFeatures)
    {
        s = copyBinIndex<cpu>(nRows, nCols, indexedFeatures->data(0), binIndexVector, &binIndex);
        DAAL_CHECK_STATUS_VAR(s);
    }

    const auto nFeatures = x->getNumberOfColumns();
    WriteOnlyRows<algorithmFPType, cpu> varImpBD(res.varImp, 0, 1);
    if (res.varImp) DAAL_CHECK_BLOCK_STATUS(varImpBD);

    //set of auxiliary data used by single thread or a target for reduction by multiple threads
    typedef typename TaskType::ThreadCtxType Ctx;
    Ctx mainCtx(varImpBD.get());
    DAAL_CHECK(mainCtx.init(par, x, nClasses), ErrorMemoryAllocationFailed);

    if (mainCtx.varImp)
        //initialize its data
        daal::services::internal::service_memset<algorithmFPType, cpu>(mainCtx.varImp, 0, nFeatures);

    //use tls in case of multiple threads
    const bool bThreaded = (threader_get_max_threads_number() > 1) && (par.nTrees > 1);
    daal::tls<Ctx *> tlsCtx([&]() -> Ctx * {
        //in case of single thread no need to allocate
        return (bThreaded ? createTlsContext<algorithmFPType, cpu, Ctx>(x, par, nClasses) : &mainCtx);
    });
    daal::tls<TaskType *> tlsTask([&]() -> TaskType * {
        //in case of single thread no need to allocate
        Ctx * ctx = tlsCtx.local();
        return ctx ? new TaskType(pHostApp, x, y, w, par, featTypes, indexedFeatures, binIndex, *ctx, nClasses, hyperparameter) : nullptr;
    });

    engines::internal::ParallelizationTechnique technique = engines::internal::family;
    selectParallelizationTechnique<cpu>(par, technique);
    engines::internal::Params<cpu> params(par.nTrees);
    for (size_t i = 0; i < par.nTrees; i++)
    {
        params.nSkip[i] = i * par.nTrees * x->getNumberOfRows() * (par.featuresPerNode + 1);
    }
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, par.nTrees, sizeof(engines::EnginePtr));
    TArray<engines::EnginePtr, cpu> engines(par.nTrees);
    engines::internal::EnginesCollection<cpu> enginesCollection(par.engine, technique, params, engines, &s);
    DAAL_CHECK_STATUS_VAR(s);

    services::internal::TArray<size_t, cpu> numElems(par.nTrees);

    daal::SafeStatus safeStat;
    daal::threader_for(par.nTrees, par.nTrees, [&](size_t i) {
        if (!safeStat.ok()) return;
        TaskType * task = tlsTask.local();
        DAAL_CHECK_MALLOC_THR(task);
        dtrees::internal::Tree * pTree = nullptr;
        numElems[i]                    = 0;
        auto engineImpl                = dynamic_cast<engines::internal::BatchBaseImpl *>(engines[i].get());
        DAAL_CHECK_THR(engineImpl, ErrorEngineNotSupported);
        services::Status s = task->run(engineImpl, pTree, numElems[i]);
        DAAL_CHECK_STATUS_THR(s);
        if (pTree)
        {
            md.add((typename ModelType::TreeType &)*pTree, nClasses, i);
        }
    });
    s = safeStat.detach();
    tlsCtx.reduce([&](Ctx * ctx) -> void {
        if (ctx && bThreaded)
        {
            ctx->reduceTo(par.varImportance, mainCtx, nFeatures, nRows);
            ctx->~Ctx();
            service_scalable_free<byte, cpu>((byte *)ctx);
        }
    });
    tlsTask.parallel_reduce([&](TaskType * task) -> void {
        delete task;
        task = nullptr;
    });
    DAAL_CHECK_STATUS_VAR(s);
    DAAL_CHECK_MALLOC(md.size() == par.nTrees);

    res.updatedEngine = enginesCollection.getUpdatedEngine(par.engine, engines, numElems);

    //finalize results computation
    //variable importance
    if (varImpBD.get()) mainCtx.finalizeVarImp(par.varImportance, nFeatures);

    //OOB error
    if (par.resultsToCompute
        & (computeOutOfBagError | computeOutOfBagErrorPerObservation | computeOutOfBagErrorAccuracy | computeOutOfBagErrorR2
           | computeOutOfBagErrorDecisionFunction | computeOutOfBagErrorPrediction))
    {
        WriteOnlyRows<algorithmFPType, cpu> oobErr(res.oobError, 0, 1);
        if (par.resultsToCompute & computeOutOfBagError) DAAL_CHECK_BLOCK_STATUS(oobErr);

        WriteOnlyRows<algorithmFPType, cpu> oobErrAccuracy(res.oobErrorAccuracy, 0, 1);
        if (par.resultsToCompute & computeOutOfBagErrorAccuracy) DAAL_CHECK_BLOCK_STATUS(oobErrAccuracy);

        WriteOnlyRows<algorithmFPType, cpu> oobErrR2(res.oobErrorR2, 0, 1);
        if (par.resultsToCompute & computeOutOfBagErrorR2) DAAL_CHECK_BLOCK_STATUS(oobErrR2);

        WriteOnlyRows<algorithmFPType, cpu> oobErrPerObs(res.oobErrorPerObs, 0, nRows);
        if (par.resultsToCompute & computeOutOfBagErrorPerObservation) DAAL_CHECK_BLOCK_STATUS(oobErrPerObs);

        WriteOnlyRows<algorithmFPType, cpu> oobErrDecisionFunction(res.oobErrorDecisionFunction, 0, nRows);
        if (par.resultsToCompute & computeOutOfBagErrorDecisionFunction) DAAL_CHECK_BLOCK_STATUS(oobErrDecisionFunction);

        WriteOnlyRows<algorithmFPType, cpu> oobErrPrediction(res.oobErrorPrediction, 0, nRows);
        if (par.resultsToCompute & computeOutOfBagErrorPrediction) DAAL_CHECK_BLOCK_STATUS(oobErrPrediction);

        s = mainCtx.finalizeOOBError(y, oobErr.get(), oobErrPerObs.get(), oobErrAccuracy.get(), oobErrR2.get(), oobErrDecisionFunction.get(),
                                     oobErrPrediction.get());
    }
    return s;
}

//////////////////////////////////////////////////////////////////////////////////////////
// Base task class. Implements general pipeline of tree building
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, typename BinIndexType, typename DataHelper, typename HyperparameterType, CpuType cpu>
class TrainBatchTaskBase
{
public:
    typedef TreeThreadCtxBase<algorithmFPType, cpu> ThreadCtxType;

    services::Status run(engines::internal::BatchBaseImpl * engineImpl, dtrees::internal::Tree *& pTree, size_t & numElems);

protected:
    typedef dtrees::internal::TVector<algorithmFPType, cpu> algorithmFPTypeArray;
    typedef dtrees::internal::TVector<IndexType, cpu> IndexTypeArray;
    TrainBatchTaskBase(HostAppIface * hostApp, const NumericTable * x, const NumericTable * y, const NumericTable * w, const Parameter & par,
                       const dtrees::internal::FeatureTypes & featTypes, const dtrees::internal::IndexedFeatures * indexedFeatures,
                       const BinIndexType * binIndex, ThreadCtxType & threadCtx, size_t nClasses, const HyperparameterType * hyperparameter = nullptr)
        : _hostApp(hostApp, 0), //set granularity later
          _data(x),
          _resp(y),
          _weights(w),
          _par(par),
          _nClasses(nClasses),
          _nSamples(par.observationsPerTreeFraction * x->getNumberOfRows()),
          _nFeaturesPerNode(par.featuresPerNode),
          _helper(indexedFeatures, nClasses, hyperparameter),
          _binIndex(binIndex),
          _impurityThreshold(_par.impurityThreshold),
          _nFeatureBufs(1), //for sequential processing
          _featHelper(featTypes),
          _threadCtx(threadCtx),
          _accuracy(daal::services::internal::EpsilonVal<algorithmFPType>::get()),
          _minSamplesSplit(2),
          _minWeightLeaf(0.),
          _minImpurityDecrease(-daal::services::internal::EpsilonVal<algorithmFPType>::get() * x->getNumberOfRows()),
          _maxLeafNodes(0),
          _useConstFeatures(false),
          _memorySavingMode(false),
          _hyperparameter(hyperparameter)
    {
        if (_impurityThreshold < _accuracy) _impurityThreshold = _accuracy;

        _memorySavingMode = indexedFeatures == nullptr;
        const daal::algorithms::decision_forest::training::interface2::Parameter * algParameter =
            dynamic_cast<const daal::algorithms::decision_forest::training::interface2::Parameter *>(&par);
        if (algParameter != NULL)
        {
            _minSamplesSplit = 2 > par.minObservationsInSplitNode ? 2 : par.minObservationsInSplitNode;
            _minSamplesSplit = _minSamplesSplit > 2 * par.minObservationsInLeafNode ? _minSamplesSplit : 2 * par.minObservationsInLeafNode;
            if (_weights)
            {
                const size_t firstRow = 0;
                const size_t lastRow  = x->getNumberOfRows();
                ReadRows<algorithmFPType, cpu> bd(const_cast<NumericTable *>(_weights), firstRow, lastRow - firstRow + 1);
                const auto pbd               = bd.get();
                algorithmFPType totalWeights = 0.0;
                PRAGMA_VECTOR_ALWAYS
                for (size_t i = 0; i < lastRow; ++i)
                {
                    totalWeights += pbd[i];
                }
                _minWeightLeaf = par.minWeightFractionInLeafNode * totalWeights;
                _minImpurityDecrease =
                    par.minImpurityDecreaseInSplitNode * totalWeights - daal::services::internal::EpsilonVal<algorithmFPType>::get() * totalWeights;
            }
            else
            {
                _minWeightLeaf       = par.minWeightFractionInLeafNode * x->getNumberOfRows();
                _minImpurityDecrease = par.minImpurityDecreaseInSplitNode * x->getNumberOfRows()
                                       - daal::services::internal::EpsilonVal<algorithmFPType>::get() * x->getNumberOfRows();
            }
            _maxLeafNodes = par.maxLeafNodes;
        }
    }

    size_t nFeatures() const { return _data->getNumberOfColumns(); }
    typename DataHelper::NodeType::Base * buildDepthFirst(services::Status & s, size_t iStart, size_t n, size_t level,
                                                          typename DataHelper::ImpurityData & curImpurity, bool & bUnorderedFeaturesUsed,
                                                          size_t nClasses, algorithmFPType totalWeights);
    typename DataHelper::NodeType::Base * buildBestFirst(services::Status & s, size_t iStart, size_t n, size_t level,
                                                         typename DataHelper::ImpurityData & curImpurity, bool & bUnorderedFeaturesUsed,
                                                         size_t nClasses, algorithmFPType totalWeights);
    template <typename WorkItem>
    typename DataHelper::NodeType::Base * buildNode(const size_t level, const size_t nClasses, size_t & remainingSplitNodes, WorkItem & item,
                                                    typename DataHelper::ImpurityData & impurity);

    algorithmFPType * featureBuf(size_t iBuf) const
    {
        DAAL_ASSERT(iBuf < _nFeatureBufs);
        return _aFeatureBuf[iBuf].get();
    }
    IndexType * featureIndexBuf(size_t iBuf) const
    {
        DAAL_ASSERT(iBuf < _nFeatureBufs);
        return _aFeatureIndexBuf[iBuf].get();
    }
    bool terminateCriteria(size_t nSamples, size_t level, typename DataHelper::ImpurityData & imp, algorithmFPType totalWeights) const
    {
        const daal::algorithms::decision_forest::training::interface2::Parameter * algParameter =
            dynamic_cast<const daal::algorithms::decision_forest::training::interface2::Parameter *>(&_par);
        if (algParameter != NULL)
        {
            return ((nSamples < 2 * _par.minObservationsInLeafNode) || (nSamples < _minSamplesSplit) || (totalWeights < 2 * _minWeightLeaf)
                    || _helper.terminateCriteria(imp, _impurityThreshold, nSamples) || ((_par.maxTreeDepth > 0) && (level >= _par.maxTreeDepth)));
        }
        else
            return (nSamples < 2 * _par.minObservationsInLeafNode || _helper.terminateCriteria(imp, _impurityThreshold, nSamples)
                    || ((_par.maxTreeDepth > 0) && (level >= _par.maxTreeDepth)));
    }
    ThreadCtxType & threadCtx() { return _threadCtx; }
    typename DataHelper::NodeType::Split * makeSplit(size_t iFeature, algorithmFPType featureValue, bool bUnordered,
                                                     typename DataHelper::NodeType::Base * left, typename DataHelper::NodeType::Base * right,
                                                     algorithmFPType imp);
    typename DataHelper::NodeType::Leaf * makeLeaf(const IndexType * idx, size_t n, typename DataHelper::ImpurityData & imp, size_t makeLeaf);

    NodeSplitResult findBestSplit(size_t level, size_t iStart, size_t n, const typename DataHelper::ImpurityData & curImpurity,
                                  IndexType & iBestFeature, typename DataHelper::TSplitData & split, algorithmFPType totalWeights);
    NodeSplitResult findBestSplitSerial(size_t level, size_t iStart, size_t n, const typename DataHelper::ImpurityData & curImpurity,
                                        IndexType & iBestFeature, typename DataHelper::TSplitData & split, algorithmFPType totalWeights);
    NodeSplitResult simpleSplit(size_t iStart, const typename DataHelper::ImpurityData & curImpurity, IndexType & iFeatureBest,
                                typename DataHelper::TSplitData & split);
    void addImpurityDecrease(IndexType iFeature, size_t n, const typename DataHelper::ImpurityData & curImpurity,
                             const typename DataHelper::TSplitData & split);

    void featureValuesToBuf(size_t iFeature, algorithmFPType * featureVal, IndexType * aIdx, size_t n)
    {
        _helper.getColumnValues(iFeature, aIdx, n, featureVal);
        daal::algorithms::internal::qSort<algorithmFPType, int, cpu>(n, featureVal, aIdx);
    }

    services::Status computeResults(const dtrees::internal::Tree & t);

    algorithmFPType computeOOBError(const dtrees::internal::Tree & t, size_t n, const IndexType * aInd);

    algorithmFPType computeOOBErrorPerm(const dtrees::internal::Tree & t, size_t n, const IndexType * aInd, const IndexType * aPerm,
                                        size_t iPermutedFeature);

    void setupHostApp()
    {
        DAAL_INT64 minPartCoeff = 4l;
        DAAL_INT64 minSizeCoeff = 24000l;
        if (this->_hyperparameter != nullptr)
        {
            if (std::is_same<HyperparameterType, decision_forest::classification::training::internal::Hyperparameter>::value)
            {
                this->_hyperparameter->find(classification::training::internal::minPartCoefficient, minPartCoeff);
                this->_hyperparameter->find(classification::training::internal::minSizeCoefficient, minSizeCoeff);
            }
            else if (std::is_same<HyperparameterType, decision_forest::regression::training::internal::Hyperparameter>::value)
            {
                this->_hyperparameter->find(regression::training::internal::minPartCoefficient, minPartCoeff);
                this->_hyperparameter->find(regression::training::internal::minSizeCoefficient, minSizeCoeff);
            }
        }
        const size_t minPart = minPartCoeff * _helper.size();    // corresponds to the 4 topmost levels, if minPartCoeff == 4
        const size_t minSize = minSizeCoeff / _nFeaturesPerNode; // at least that many, corresponds to the tree 1000 obs/10 features/8 levels,
                                                                 // if minSizeCoeff == 24000
        _hostApp.setup(minPart < minSize ? minSize : minPart);
    }

protected:
    TArray<IndexType, cpu> _aFeatureIdx;      //indices of features to be used for the split at the current level
    TArray<IndexType, cpu> _aConstFeatureIdx; //indices of found constant features
    DataHelper _helper;
    services::internal::HostAppHelper _hostApp;
    typename DataHelper::TreeType _tree;
    mutable TVector<IndexType, cpu> _aSample;
    mutable TArray<algorithmFPTypeArray, cpu> _aFeatureBuf;
    mutable TArray<IndexTypeArray, cpu> _aFeatureIndexBuf;

    const NumericTable * _data;
    const NumericTable * _resp;
    const NumericTable * _weights;
    const Parameter & _par;
    const size_t _nSamples;
    const size_t _nFeaturesPerNode;
    const size_t _nFeatureBufs;   //number of buffers to get feature values (to process features independently in parallel)
    const bool _useConstFeatures; //including constant features in number of features per node
    mutable size_t _nConstFeature;

    const BinIndexType * _binIndex;
    const FeatureTypes & _featHelper;
    algorithmFPType _accuracy;
    algorithmFPType _impurityThreshold;
    ThreadCtxType & _threadCtx;
    size_t _nClasses;
    size_t * _numElems;
    size_t _minSamplesSplit;
    algorithmFPType _minWeightLeaf;
    algorithmFPType _minImpurityDecrease;
    size_t _maxLeafNodes;
    bool _memorySavingMode;

    const HyperparameterType * _hyperparameter;
};

template <typename algorithmFPType, typename BinIndexType, typename DataHelper, typename HyperparameterType, CpuType cpu>
services::Status TrainBatchTaskBase<algorithmFPType, BinIndexType, DataHelper, HyperparameterType, cpu>::run(
    engines::internal::BatchBaseImpl * engineImpl, dtrees::internal::Tree *& pTree, size_t & numElems)
{
    const size_t maxFeatures = nFeatures();
    _nConstFeature           = 0;
    _numElems                = &numElems;
    _helper.engineImpl       = engineImpl;
    pTree                    = nullptr;
    _tree.destroy();
    _aSample.reset(_nSamples);
    _aFeatureBuf.reset(_nFeatureBufs);
    _aFeatureIndexBuf.reset(_nFeatureBufs);

    /* first maxFeatures entries serve as a buffer of drawn samples for node splitting */
    /* second maxFeatures entries contains [0, ..., maxFeatures - 1] and is used to randomly draw indices */
    _aFeatureIdx.reset(maxFeatures * 2);
    _aConstFeatureIdx.reset(maxFeatures * 2);

    DAAL_CHECK_MALLOC(_aConstFeatureIdx.get());
    services::internal::service_memset_seq<IndexType, cpu>(_aConstFeatureIdx.get(), IndexType(0), 2 * maxFeatures);
    // in order to use drawKFromBufferWithoutReplacement we need to initialize
    // the buffer to contain all indices from [0, 1, ..., maxFeatures - 1]
    DAAL_CHECK_MALLOC(_aFeatureIdx.get());
    services::internal::service_memset_seq<IndexType, cpu>(_aFeatureIdx.get(), IndexType(0), maxFeatures);
    services::internal::service_memset_incrementing<IndexType, cpu>(_aFeatureIdx.get() + maxFeatures, IndexType(0), maxFeatures);

    DAAL_CHECK_MALLOC(_aSample.get() && _helper.reset(_nSamples) && _helper.resetWeights(_nSamples) && _aFeatureBuf.get() && _aFeatureIndexBuf.get()
                      && _aFeatureIdx.get());

    for (size_t i = 0; i < _nFeatureBufs; ++i)
    {
        _aFeatureBuf[i].reset(_data->getNumberOfRows());
        DAAL_CHECK_MALLOC(_aFeatureBuf[i].get());
        _aFeatureIndexBuf[i].reset(_nSamples);
        DAAL_CHECK_MALLOC(_aFeatureIndexBuf[i].get());
    }

    if (_par.bootstrap)
    {
        *_numElems += _nSamples;
        RNGsInst<int, cpu> rng;
        rng.uniform(_nSamples, _aSample.get(), _helper.engineImpl->getState(), 0, _data->getNumberOfRows());
        daal::algorithms::internal::qSort<int, cpu>(_nSamples, _aSample.get());
    }
    else
    {
        auto aSample = _aSample.get();
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < _nSamples; ++i)
        {
            aSample[i] = i;
        }
    }
    //init responses buffer, keep _aSample values in it
    DAAL_CHECK_MALLOC(_helper.init(_data, _resp, _aSample.get(), _weights));

    //use _aSample as an array of response indices stored by helper from now on
    PRAGMA_FORCE_SIMD
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < _aSample.size(); ++i)
    {
        _aSample[i] = i;
    }

    setupHostApp();

    double totalWeights = double(0);
    typename DataHelper::ImpurityData initialImpurity;
    const bool noWeights = !_helper.providedWeights();
    if (noWeights)
    {
        _helper.template calcImpurity<true>(_aSample.get(), _nSamples, initialImpurity, totalWeights);
    }
    else
    {
        _helper.template calcImpurity<false>(_aSample.get(), _nSamples, initialImpurity, totalWeights);
    }
    bool bUnorderedFeaturesUsed = false;

    Status s;
    typename DataHelper::NodeType::Base * nd =
        _maxLeafNodes ? buildBestFirst(s, 0, _nSamples, 0, initialImpurity, bUnorderedFeaturesUsed, _nClasses, totalWeights) :
                        buildDepthFirst(s, 0, _nSamples, 0, initialImpurity, bUnorderedFeaturesUsed, _nClasses, totalWeights);
    if (nd)
    {
        //to prevent memory leak in case of general allocator
        _tree.reset(nd, bUnorderedFeaturesUsed);
        _threadCtx.nTrees++;
    }

    if (s
        && ((_par.resultsToCompute
             & (computeOutOfBagError | computeOutOfBagErrorPerObservation | computeOutOfBagErrorAccuracy | computeOutOfBagErrorR2
                | computeOutOfBagErrorDecisionFunction | computeOutOfBagErrorPrediction))
            || (_par.varImportance > MDI)))
        s = computeResults(_tree);
    if (s) pTree = &_tree;
    return s;
}

template <typename algorithmFPType, typename BinIndexType, typename DataHelper, typename HyperparameterType, CpuType cpu>
typename DataHelper::NodeType::Split * TrainBatchTaskBase<algorithmFPType, BinIndexType, DataHelper, HyperparameterType, cpu>::makeSplit(
    size_t iFeature, algorithmFPType featureValue, bool bUnordered, typename DataHelper::NodeType::Base * left,
    typename DataHelper::NodeType::Base * right, algorithmFPType imp)
{
    typename DataHelper::NodeType::Split * pNode = _tree.allocator().allocSplit();
    pNode->set(iFeature, featureValue, bUnordered);
    pNode->kid[0]   = left;
    pNode->kid[1]   = right;
    pNode->impurity = imp;
    return pNode;
}

template <typename algorithmFPType, typename BinIndexType, typename DataHelper, typename HyperparameterType, CpuType cpu>
typename DataHelper::NodeType::Leaf * TrainBatchTaskBase<algorithmFPType, BinIndexType, DataHelper, HyperparameterType, cpu>::makeLeaf(
    const IndexType * idx, size_t n, typename DataHelper::ImpurityData & imp, size_t nClasses)
{
    typename DataHelper::NodeType::Leaf * pNode = _tree.allocator().allocLeaf(_nClasses);
    _helper.setLeafData(*pNode, idx, n, imp);
    return pNode;
}

template <typename algorithmFPType, typename BinIndexType, typename DataHelper, typename HyperparameterType, CpuType cpu>
typename DataHelper::NodeType::Base * TrainBatchTaskBase<algorithmFPType, BinIndexType, DataHelper, HyperparameterType, cpu>::buildDepthFirst(
    services::Status & s, size_t iStart, size_t n, size_t level, typename DataHelper::ImpurityData & curImpurity, bool & bUnorderedFeaturesUsed,
    size_t nClasses, algorithmFPType totalWeights)
{
    const size_t maxFeatures = nFeatures();
    if (_hostApp.isCancelled(s, n)) return nullptr;

    if (terminateCriteria(n, level, curImpurity, totalWeights)) return makeLeaf(_aSample.get() + iStart, n, curImpurity, nClasses);

    typename DataHelper::TSplitData split;
    IndexType iFeature;

    NodeSplitResult split_result = findBestSplit(level, iStart, n, curImpurity, iFeature, split, totalWeights);
    DAAL_ASSERT(split_result.status.ok());
    if (split_result.bSplitSucceeded)
    {
        const size_t nLeft   = split.nLeft;
        const double imp     = curImpurity.var;
        const double impLeft = split.left.var;

        // check impurity decrease
        if (split.totalWeights * split.impurityDecrease < _minImpurityDecrease) return makeLeaf(_aSample.get() + iStart, n, curImpurity, nClasses);
        if (_par.varImportance == training::MDI) addImpurityDecrease(iFeature, n, curImpurity, split);
        typename DataHelper::NodeType::Base * left =
            buildDepthFirst(s, iStart, split.nLeft, level + 1, split.left, bUnorderedFeaturesUsed, nClasses, split.leftWeights);

        /* The split information can be converted from the left-side of the split
           to the right-side without accessing the raw data by using the cost function
           values, weights, and number of elements from the parent node and new left
           node. This is faster but accumulates floating point arithmetic error.
           When the right-side is a single value, the raw value can be directly
           used and the cost function can be immediately calculated. This removes
           the error for single-element right nodes and is important in unlimited-
           depth decision forest fits. The if statement separates these two
           methods and checks for single element right splits.*/
        if (n - split.nLeft != 1)
        {
            _helper.convertLeftImpToRight(n, curImpurity, split);
        }
        else
        {
            _helper.singleSwap(_aSample.get()[iStart + nLeft], split);
        }

        if (!_memorySavingMode && !_useConstFeatures)
        {
            for (size_t i = _nConstFeature; i > 0; --i)
            {
                if (level + 1 < _aConstFeatureIdx[maxFeatures + _aConstFeatureIdx[i - 1]])
                {
                    DAAL_ASSERT(_nConstFeature > 0);
                    --_nConstFeature;
                    _aConstFeatureIdx[maxFeatures + _aConstFeatureIdx[i - 1]] = 0; //clean level
                    _aConstFeatureIdx[i - 1]                                  = 0; //clean index
                }
                else
                {
                    break;
                }
            }
        }
        typename DataHelper::NodeType::Base * right =
            s.ok() ? buildDepthFirst(s, iStart + nLeft, split.nLeft, level + 1, split.left, bUnorderedFeaturesUsed, nClasses, split.leftWeights) :
                     nullptr;
        typename DataHelper::NodeType::Base * res = nullptr;
        if (!left || !right || !(res = makeSplit(iFeature, split.featureValue, split.featureUnordered, left, right, curImpurity.var)))
        {
            if (left) dtrees::internal::deleteNode<typename DataHelper::NodeType, typename DataHelper::TreeType::Allocator>(left, _tree.allocator());
            if (right)
                dtrees::internal::deleteNode<typename DataHelper::NodeType, typename DataHelper::TreeType::Allocator>(right, _tree.allocator());
            return nullptr;
        }
        bUnorderedFeaturesUsed |= bool(split.featureUnordered);
        res->count = n;
        DAAL_ASSERT(nLeft == left->count);
        DAAL_ASSERT(split.nLeft == right->count);
        return res;
    }

    return makeLeaf(_aSample.get() + iStart, n, curImpurity, nClasses);
}

template <typename algorithmFPType, typename BinIndexType, typename DataHelper, typename HyperparameterType, CpuType cpu>
template <typename WorkItem>
typename DataHelper::NodeType::Base * TrainBatchTaskBase<algorithmFPType, BinIndexType, DataHelper, HyperparameterType, cpu>::buildNode(
    const size_t level, const size_t nClasses, size_t & remainingSplitNodes, WorkItem & item, typename DataHelper::ImpurityData & impurity)
{
    typename DataHelper::TSplitData split;
    IndexType iFeature;

    if (!remainingSplitNodes || terminateCriteria(item.n, item.level, impurity, item.totalWeights))
    {
        return makeLeaf(_aSample.get() + item.start, item.n, impurity, nClasses);
    }

    NodeSplitResult split_result = findBestSplit(level, item.start, item.n, impurity, iFeature, split, item.totalWeights);
    DAAL_ASSERT(split_result.status.ok());
    if (split_result.bSplitSucceeded)
    {
        const double imp     = impurity.var;
        const double impLeft = split.left.var;

        // check impurity decrease
        double improve = imp * item.totalWeights - impLeft * item.leftWeights - (item.totalWeights - item.leftWeights) * (imp - impLeft);
        if (improve < _minImpurityDecrease)
        {
            return makeLeaf(_aSample.get() + item.start, item.n, impurity, nClasses);
        }
        else
        {
            if (_par.varImportance == training::MDI)
            {
                addImpurityDecrease(iFeature, item.n, impurity, split);
            }

            item.nLeft        = split.nLeft;
            item.leftWeights  = split.leftWeights;
            item.improvement  = improve;
            item.impurityLeft = split.left;
            _helper.convertLeftImpToRight(item.n, impurity, split);
            item.impurityRight = split.left;

            if (!(item.node = makeSplit(iFeature, split.featureValue, split.featureUnordered, nullptr, nullptr, impurity.var)))
            {
                return nullptr;
            }

            item.isLeaf = false;
            item.featureUnordered |= bool(split.featureUnordered);
            item.node->count = item.n;
            --remainingSplitNodes;
            return item.node;
        }
    }

    return makeLeaf(_aSample.get() + item.start, item.n, impurity, nClasses);
}

template <typename algorithmFPType, typename BinIndexType, typename DataHelper, typename HyperparameterType, CpuType cpu>
typename DataHelper::NodeType::Base * TrainBatchTaskBase<algorithmFPType, BinIndexType, DataHelper, HyperparameterType, cpu>::buildBestFirst(
    services::Status & s, size_t iStart, size_t n, size_t level, typename DataHelper::ImpurityData & curImpurity, bool & bUnorderedFeaturesUsed,
    size_t nClasses, algorithmFPType totalWeights)
{
    struct WorkItem
    {
        bool isLeaf;
        bool featureUnordered;
        size_t start;
        size_t n;
        size_t nLeft;
        size_t level;
        double improvement;
        algorithmFPType leftWeights;
        algorithmFPType totalWeights;
        typename DataHelper::ImpurityData impurityLeft {};
        typename DataHelper::ImpurityData impurityRight {};
        typename DataHelper::NodeType::Split * node;

        WorkItem()
            : isLeaf(true),
              featureUnordered(false),
              start(0),
              n(0),
              nLeft(0),
              level(0),
              improvement(0.0),
              leftWeights(0.),
              totalWeights(0.),
              node(nullptr)
        {}

        WorkItem(bool featureUnordered, size_t start, size_t n, size_t level, algorithmFPType totalWeights)
            : isLeaf(true),
              featureUnordered(featureUnordered),
              start(start),
              n(n),
              nLeft(0),
              level(level),
              improvement(0.0),
              leftWeights(0.),
              totalWeights(totalWeights),
              node(nullptr)
        {}

        WorkItem & operator=(const WorkItem & src)
        {
            if (src.isLeaf)
            {
                improvement = 0.0;
                isLeaf      = true;
                return *this;
            }

            isLeaf           = src.isLeaf;
            featureUnordered = src.featureUnordered;
            start            = src.start;
            n                = src.n;
            nLeft            = src.nLeft;
            level            = src.level;
            improvement      = src.improvement;
            impurityLeft     = src.impurityLeft;
            impurityRight    = src.impurityRight;
            node             = src.node;
            leftWeights      = src.leftWeights;
            totalWeights     = src.totalWeights;

            return *this;
        }
    };

    if (_hostApp.isCancelled(s, n))
    {
        return nullptr;
    }
    BinaryHeap<WorkItem, cpu> binaryHeap(s);
    if (!s.ok())
    {
        return nullptr;
    }
    size_t remainingSplitNodes = _maxLeafNodes - 1;

    // Create base
    WorkItem base(bUnorderedFeaturesUsed, iStart, n, level, totalWeights);
    typename DataHelper::NodeType::Base * baseNode =
        TrainBatchTaskBase<algorithmFPType, BinIndexType, DataHelper, HyperparameterType, cpu>::buildNode(level, nClasses, remainingSplitNodes, base,
                                                                                                          curImpurity);

    DAAL_ASSERT(baseNode);
    s = binaryHeap.push(base);
    if (!s.ok())
    {
        return nullptr;
    }

    while (!binaryHeap.empty())
    {
        WorkItem & src = binaryHeap.pop();
        if (src.isLeaf)
        {
            continue;
        }

        // create leftChild
        WorkItem leftChild(src.featureUnordered, src.start, src.nLeft, src.level + 1, src.leftWeights);
        src.node->kid[0] = TrainBatchTaskBase<algorithmFPType, BinIndexType, DataHelper, HyperparameterType, cpu>::buildNode(
            src.level + 1, nClasses, remainingSplitNodes, leftChild, src.impurityLeft);

        // create rightChild
        WorkItem rightChild(src.featureUnordered, src.start + src.nLeft, src.n - src.nLeft, src.level + 1, src.totalWeights - src.leftWeights);
        src.node->kid[1] = TrainBatchTaskBase<algorithmFPType, BinIndexType, DataHelper, HyperparameterType, cpu>::buildNode(
            src.level + 1, nClasses, remainingSplitNodes, rightChild, src.impurityRight);

        DAAL_ASSERT(src.node->kid[0]);
        DAAL_ASSERT(src.node->kid[1]);
        s = binaryHeap.push(leftChild);
        if (!s.ok())
        {
            return nullptr;
        }
        s = binaryHeap.push(rightChild);
        if (!s.ok())
        {
            return nullptr;
        }
    }
    return baseNode;
}

template <typename algorithmFPType, typename BinIndexType, typename DataHelper, typename HyperparameterType, CpuType cpu>
NodeSplitResult TrainBatchTaskBase<algorithmFPType, BinIndexType, DataHelper, HyperparameterType, cpu>::simpleSplit(
    size_t iStart, const typename DataHelper::ImpurityData & curImpurity, IndexType & iFeatureBest, typename DataHelper::TSplitData & split)
{
    services::Status st;
    RNGsInst<IndexType, cpu> rng;
    algorithmFPType featBuf[2];
    IndexType * aIdx = _aSample.get() + iStart;
    for (size_t i = 0; i < _nFeaturesPerNode; ++i)
    {
        IndexType iFeature;
        *_numElems += 1;

        int errorcode = rng.uniform(1, &iFeature, _helper.engineImpl->getState(), 0, _data->getNumberOfColumns());
        if (errorcode)
        {
            st = services::Status(services::ErrorNullResult);
        }

        featureValuesToBuf(iFeature, featBuf, aIdx, 2);
        if (featBuf[1] - featBuf[0] <= _accuracy) //all values of the feature are the same
            continue;
        _helper.simpleSplit(featBuf, aIdx, split);
        split.featureUnordered = _featHelper.isUnordered(iFeature);
        split.impurityDecrease = curImpurity.var;
        iFeatureBest           = iFeature;
        return { st, true };
    }
    return { st, false };
}

template <typename algorithmFPType, typename BinIndexType, typename DataHelper, typename HyperparameterType, CpuType cpu>
NodeSplitResult TrainBatchTaskBase<algorithmFPType, BinIndexType, DataHelper, HyperparameterType, cpu>::findBestSplit(
    size_t level, size_t iStart, size_t n, const typename DataHelper::ImpurityData & curImpurity, IndexType & iFeatureBest,
    typename DataHelper::TSplitData & split, algorithmFPType totalWeights)
{
    if (n == 2)
    {
        DAAL_ASSERT(_par.minObservationsInLeafNode == 1);
#ifdef DEBUG_CHECK_IMPURITY
        _helper.checkImpurity(_aSample.get() + iStart, n, curImpurity);
#endif
        return simpleSplit(iStart, curImpurity, iFeatureBest, split);
    }
    return findBestSplitSerial(level, iStart, n, curImpurity, iFeatureBest, split, totalWeights);
}

//find best split and put it to featureIndexBuf
template <typename algorithmFPType, typename BinIndexType, typename DataHelper, typename HyperparameterType, CpuType cpu>
NodeSplitResult TrainBatchTaskBase<algorithmFPType, BinIndexType, DataHelper, HyperparameterType, cpu>::findBestSplitSerial(
    size_t level, size_t iStart, size_t n, const typename DataHelper::ImpurityData & curImpurity, IndexType & iBestFeature,
    typename DataHelper::TSplitData & bestSplit, algorithmFPType totalWeights)
{
    services::Status st;

    /* counter of the number of visited features, we visit _nFeaturesPerNode
    *  depending on _useConstFeatures, constant features can be skipped
    */
    size_t nVisitedFeature = 0;
    /* total number of features */
    const size_t maxFeatures = nFeatures();
    /* minimum fraction of all samples per bin */
    const algorithmFPType qMax = 0.02;
    /* index of the best split, initialized to first index we investigate */
    IndexType * bestSplitIdx = featureIndexBuf(0) + iStart;
    /* sample index */
    IndexType * aIdx = _aSample.get() + iStart;
    /* zero-based index of best split */
    int64_t iBestSplit               = -1;
    int64_t idxFeatureValueBestSplit = -1;
    typename DataHelper::TSplitData split {};
    /* RNG for sample drawing */
    RNGsInst<IndexType, cpu> rng;
    /* index for swapping samples in Fisher-Yates sampling */
    IndexType swapIdx;

    for (size_t i = 0; i < maxFeatures && nVisitedFeature < _nFeaturesPerNode; ++i)
    {
        /* draw a random sample without replacement */
        // based on Fisher Yates sampling
        // _aFeatureIdx has length of 2 * _maxFeatures
        // first maxFeatures contain the currently selected features
        // at iteration i, we have drawn i features and written them to
        // _aFeatureIdx[0, 1, ..., i-1]
        //
        // the second half of the buffer contains all numbers from
        // [0, 1, ..., maxFeatures-1] and we randomly select one without
        // replacement based on Fisher Yates sampling
        // drawing uniformly from [0, maxFeatures-i] and swapping the indices
        // assures uniform probability of all drawn numbers

        /* draw the i-th index of the sample */
        int errorcode = rng.uniform(1, &swapIdx, _helper.engineImpl->getState(), 0, maxFeatures - i);
        if (errorcode)
        {
            st = services::Status(services::ErrorNullResult);
        }

        /* account for buffer offset from 0 */
        swapIdx += maxFeatures;
        /* _aFeatureIdx[swapIdx] was drawn */
        _aFeatureIdx[i] = _aFeatureIdx[swapIdx];
        /* swap in number at [2 * maxFeatures - 1 - i] for next draw */
        _aFeatureIdx[swapIdx] = _aFeatureIdx[2 * maxFeatures - 1 - i];
        /* store drawn number at end of number buffer so that no number is lost */
        _aFeatureIdx[2 * maxFeatures - 1 - i] = _aFeatureIdx[i];

        const auto iFeature = _aFeatureIdx[i];
        const bool bUseIndexedFeatures =
            (!_memorySavingMode) && (algorithmFPType(n) > qMax * algorithmFPType(_helper.indexedFeatures().numIndices(iFeature)));

        if (!_maxLeafNodes && !_useConstFeatures && !_memorySavingMode)
        {
            if (_aConstFeatureIdx[maxFeatures + iFeature] > 0) continue; //selected feature is known constant feature
            if (!_helper.hasDiffFeatureValues(iFeature, aIdx, n))
            {
                _aConstFeatureIdx[maxFeatures + iFeature] = level + 1;
                _aConstFeatureIdx[_nConstFeature]         = iFeature;
                ++_nConstFeature;
                continue; //all values of the feature are the same, selected feature is new constant feature
            }
            else
                ++nVisitedFeature;
        }
        else
        {
            ++nVisitedFeature;
            if (!_memorySavingMode && !_helper.hasDiffFeatureValues(iFeature, aIdx, n)) continue;
        }

        if (bUseIndexedFeatures)
        {
            split.featureUnordered = _featHelper.isUnordered(iFeature);
            //index of best feature value in the array of sorted feature values
            const int idxFeatureValue =
                _helper.findSplitForFeatureSorted(featureBuf(0), iFeature, aIdx, n, _par.minObservationsInLeafNode, curImpurity, split,
                                                  _minWeightLeaf, totalWeights, _binIndex + _data->getNumberOfRows() * iFeature);
            if (idxFeatureValue < 0) continue;
            iBestSplit = i;
            split.copyTo(bestSplit);
            idxFeatureValueBestSplit = idxFeatureValue;
        }
        else
        {
            algorithmFPType * featBuf = featureBuf(0) + iStart; //single thread
            featureValuesToBuf(iFeature, featBuf, aIdx, n);
            if (featBuf[n - 1] - featBuf[0] <= _accuracy) //all values of the feature are the same
                continue;
#ifdef DEBUG_CHECK_IMPURITY
            _helper.checkImpurity(aIdx, n, curImpurity);
#endif
            split.featureUnordered = _featHelper.isUnordered(iFeature);
            if (!_helper.findSplitForFeature(featBuf, aIdx, n, _par.minObservationsInLeafNode, _accuracy, curImpurity, split, _minWeightLeaf,
                                             totalWeights))
                continue;
            idxFeatureValueBestSplit = -1;
            iBestSplit               = i;
            split.copyTo(bestSplit);
            DAAL_ASSERT(bestSplit.iStart < n);
            DAAL_ASSERT(bestSplit.iStart + bestSplit.nLeft <= n);
            if (i + 1 < _nFeaturesPerNode || split.featureUnordered) services::internal::tmemcpy<IndexType, cpu>(bestSplitIdx, aIdx, n);
#ifdef DEBUG_CHECK_IMPURITY
            _helper.checkImpurity(aIdx, bestSplit.nLeft, bestSplit.left);
#endif
        }
    }

    if (!st.ok() || iBestSplit < 0)
    {
        // either:
        // error during splitting -> failure
        // or no split found -> not a failure but still have to return
        return { st, false };
    }

    iBestFeature    = _aFeatureIdx[iBestSplit];
    bool bCopyToIdx = true;
    if (idxFeatureValueBestSplit >= 0)
    {
        //sorted feature was used
        //calculate impurity and get split to bestSplitIdx
        const bool noWeights = !_helper.providedWeights();
        if (noWeights)
        {
            _helper.template finalizeBestSplit<true>(aIdx, _binIndex + _data->getNumberOfRows() * iBestFeature, n, iBestFeature,
                                                     idxFeatureValueBestSplit, bestSplit, bestSplitIdx);
        }
        else
        {
            _helper.template finalizeBestSplit<false>(aIdx, _binIndex + _data->getNumberOfRows() * iBestFeature, n, iBestFeature,
                                                      idxFeatureValueBestSplit, bestSplit, bestSplitIdx);
        }
    }
    else if (bestSplit.featureUnordered)
    {
        if (bestSplit.iStart)
        {
            DAAL_ASSERT(bestSplit.iStart + bestSplit.nLeft <= n);
            services::internal::tmemcpy<IndexType, cpu>(aIdx, bestSplitIdx + bestSplit.iStart, bestSplit.nLeft);
            aIdx += bestSplit.nLeft;
            services::internal::tmemcpy<IndexType, cpu>(aIdx, bestSplitIdx, bestSplit.iStart);
            aIdx += bestSplit.iStart;
            bestSplitIdx += bestSplit.iStart + bestSplit.nLeft;
            if (n > (bestSplit.iStart + bestSplit.nLeft))
                services::internal::tmemcpy<IndexType, cpu>(aIdx, bestSplitIdx, n - bestSplit.iStart - bestSplit.nLeft);
            bCopyToIdx = false; //done
        }
    }
    else
        bCopyToIdx = (iBestSplit + 1 < _nFeaturesPerNode); //if iBestSplit is the last considered feature
                                                           //then aIdx already contains the best split, no need to copy
    if (bCopyToIdx) services::internal::tmemcpy<IndexType, cpu>(aIdx, bestSplitIdx, n);
    return { st, true };
}

template <typename algorithmFPType, typename BinIndexType, typename DataHelper, typename HyperparameterType, CpuType cpu>
void TrainBatchTaskBase<algorithmFPType, BinIndexType, DataHelper, HyperparameterType, cpu>::addImpurityDecrease(
    IndexType iFeature, size_t n, const typename DataHelper::ImpurityData & curImpurity, const typename DataHelper::TSplitData & split)
{
    DAAL_ASSERT(_threadCtx.varImp);
    if (!isZero<algorithmFPType, cpu>(split.impurityDecrease)) _threadCtx.varImp[iFeature] += split.impurityDecrease;
}

template <typename algorithmFPType, typename BinIndexType, typename DataHelper, typename HyperparameterType, CpuType cpu>
services::Status TrainBatchTaskBase<algorithmFPType, BinIndexType, DataHelper, HyperparameterType, cpu>::computeResults(
    const dtrees::internal::Tree & t)
{
    const size_t nOOB = _helper.getNumOOBIndices();
    if (!nOOB) return services::Status();
    TArray<IndexType, cpu> oobIndices(nOOB);
    DAAL_CHECK_MALLOC(oobIndices.get());
    _helper.getOOBIndices(oobIndices.get());
    const bool bMDA(_par.varImportance == training::MDA_Raw || _par.varImportance == training::MDA_Scaled);
    if (_par.resultsToCompute
            & (computeOutOfBagError | computeOutOfBagErrorPerObservation | computeOutOfBagErrorAccuracy | computeOutOfBagErrorR2
               | computeOutOfBagErrorDecisionFunction | computeOutOfBagErrorPrediction)
        || bMDA)
    {
        const algorithmFPType oobError = computeOOBError(t, nOOB, oobIndices.get());
        if (bMDA)
        {
            TArray<IndexType, cpu> permutation(nOOB);
            DAAL_CHECK_MALLOC(permutation.get());
            for (size_t i = 0; i < nOOB; permutation[i] = i, ++i)
                ;
            const size_t nTrees        = _threadCtx.nTrees;
            const algorithmFPType div1 = algorithmFPType(1) / algorithmFPType(nTrees);
            for (size_t i = 0, n = nFeatures(); i < n; ++i)
            {
                shuffle<cpu>(_helper.engineImpl->getState(), nOOB, permutation.get());
                const algorithmFPType permOOBError = computeOOBErrorPerm(t, nOOB, oobIndices.get(), permutation.get(), i);
                const algorithmFPType diff         = (permOOBError - oobError);
                //_threadCtx.varImp[i] is a mean of diff among all the trees
                const algorithmFPType delta = diff - _threadCtx.varImp[i]; //old mean
                _threadCtx.varImp[i] += div1 * delta;
                if (_threadCtx.varImpVariance) _threadCtx.varImpVariance[i] += delta * (diff - _threadCtx.varImp[i]); //new mean
            }
        }
    }
    return services::Status();
}

template <typename algorithmFPType, typename DataHelper, CpuType cpu>
algorithmFPType predictionError(const DataHelper & h, const dtrees::internal::Tree & t, const algorithmFPType * x, const NumericTable * resp,
                                size_t iRow)
{
    ReadRows<algorithmFPType, cpu> y(const_cast<NumericTable *>(resp), iRow, 1);
    return h.predictionError(h.predict(t, x), *y.get());
}

template <typename algorithmFPType, typename BinIndexType, typename DataHelper, typename HyperparameterType, CpuType cpu>
algorithmFPType TrainBatchTaskBase<algorithmFPType, BinIndexType, DataHelper, HyperparameterType, cpu>::computeOOBErrorPerm(
    const dtrees::internal::Tree & t, size_t n, const IndexType * aInd, const IndexType * aPerm, size_t iPermutedFeature)
{
    DAAL_ASSERT(n);

    const auto dim = nFeatures();

    //compute prediction error on each OOB row and get its mean using online formulae (Welford)
    //TODO: can be threader_for() block
    TArray<algorithmFPType, cpu> buf(dim);
    ReadRows<algorithmFPType, cpu> x(const_cast<NumericTable *>(_data), aInd[0], 1);
    services::internal::tmemcpy<algorithmFPType, cpu>(buf.get(), x.get(), dim);
    ReadRows<algorithmFPType, cpu> p(const_cast<NumericTable *>(_data), aInd[aPerm[0]], 1);
    buf[iPermutedFeature] = p.get()[iPermutedFeature];
    algorithmFPType mean  = predictionError<algorithmFPType, DataHelper, cpu>(_helper, t, buf.get(), _resp, aInd[0]);

    for (size_t i = 1; i < n; ++i)
    {
        services::internal::tmemcpy<algorithmFPType, cpu>(buf.get(), x.set(const_cast<NumericTable *>(_data), aInd[i], 1), dim);
        buf[iPermutedFeature] = p.set(const_cast<NumericTable *>(_data), aInd[aPerm[i]], 1)[iPermutedFeature];
        algorithmFPType val   = predictionError<algorithmFPType, DataHelper, cpu>(_helper, t, buf.get(), _resp, aInd[i]);
        mean += (val - mean) / algorithmFPType(i + 1);
    }
    return mean;
}

template <typename algorithmFPType, typename BinIndexType, typename DataHelper, typename HyperparameterType, CpuType cpu>
algorithmFPType TrainBatchTaskBase<algorithmFPType, BinIndexType, DataHelper, HyperparameterType, cpu>::computeOOBError(
    const dtrees::internal::Tree & t, size_t n, const IndexType * aInd)
{
    DAAL_ASSERT(n);
    //compute prediction error on each OOB row and get its mean online formulae (Welford)
    //TODO: can be threader_for() block
    ReadRows<algorithmFPType, cpu> x(const_cast<NumericTable *>(_data), aInd[0], 1);
    algorithmFPType mean = _helper.predictionError(t, x.get(), _resp, aInd[0], _threadCtx.oobBuf);
    for (size_t i = 1; i < n; ++i)
    {
        algorithmFPType val = _helper.predictionError(t, x.set(const_cast<NumericTable *>(_data), aInd[i], 1), _resp, aInd[i], _threadCtx.oobBuf);
        mean += (val - mean) / algorithmFPType(i + 1);
    }
    return mean;
}

} /* namespace internal */
} /* namespace training */
} /* namespace decision_forest */
} /* namespace algorithms */
} /* namespace daal */

#endif
