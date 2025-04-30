/* file: gbt_train_dense_default_impl.i */
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
//  Implementation of auxiliary functions for gradient boosted trees training
//  (defaultDense) method.
//--
*/

#ifndef __GBT_TRAIN_DENSE_DEFAULT_IMPL_I__
#define __GBT_TRAIN_DENSE_DEFAULT_IMPL_I__

#include "src/algorithms/dtrees/dtrees_model_impl.h"
#include "src/algorithms/dtrees/dtrees_train_data_helper.i"
#include "src/algorithms/dtrees/dtrees_predict_dense_default_impl.i"
#include "src/algorithms/dtrees/gbt/gbt_internal.h"
#include "src/algorithms/dtrees/gbt/gbt_train_aux.i"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace training
{
namespace internal
{
using namespace daal::algorithms::dtrees::training::internal;
using namespace daal::algorithms::gbt::internal;

typedef int RowIndexType;

//////////////////////////////////////////////////////////////////////////////////////////
// Base task class. Implements general pipeline of tree building
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, typename BinIndexType, CpuType cpu>
class TrainBatchTaskBase
{
public:
    typedef gbt::internal::TreeImpRegression<> TreeType;
    typedef typename TreeType::NodeType NodeType;
    typedef ImpurityData<algorithmFPType, cpu> ImpurityType;
    typedef LossFunction<algorithmFPType, cpu> LossFunctionType;
    typedef OrderedRespHelper<algorithmFPType, cpu> DataHelperType;

    const LossFunctionType * lossFunc() const
    {
        DAAL_ASSERT(_loss);
        return _loss;
    }
    LossFunctionType * lossFunc()
    {
        DAAL_ASSERT(_loss);
        return _loss;
    }
    const Parameter & par() const { return _par; }
    const DataHelperType & dataHelper() const { return _dataHelper; }
    const FeatureTypes & featTypes() const { return _featHelper; }
    RowIndexType nFeaturesPerNode() const { return _nFeaturesPerNode; }
    bool isParallelFeatures() const { return _bParallelFeatures; }
    bool isParallelNodes() const { return _bParallelNodes; }
    bool isParallelTrees() const { return _bParallelTrees; }
    RowIndexType nSamples() const { return _nSamples; }
    bool isBagging() const { return !!_aSampleToF.get(); }
    const RowIndexType * aSampleToF() const { return _aSampleToF.get(); }
    bool isThreaded() const { return _bThreaded; }
    bool isIndexedMode() const { return !par().memorySavingMode; }
    RowIndexType numAvailableThreads() const
    {
        auto n = _nParallelNodes.get();
        return _nThreadsMax > n ? _nThreadsMax - n : 0;
    }
    size_t nFeatures() const { return _data->getNumberOfColumns(); }
    algorithmFPType accuracy() const { return _accuracy; }
    size_t nTrees() const { return _nTrees; }

    services::Status run(gbt::internal::GbtDecisionTree ** aTbl, HomogenNumericTable<double> ** aTblImp, HomogenNumericTable<int> ** aTblSmplCnt,
                         size_t iIteration, GlobalStorages<algorithmFPType, BinIndexType, cpu> & GH_SUMS_BUF);
    virtual services::Status init();
    bool isIndirect() const { return _bIndirect; }
    double computeLeafWeightUpdateF(const int * idx, size_t n, const ImpurityType & imp, size_t iTree);
    void updateOOB(size_t iTree, TreeType & t);
    bool terminateCriteria(size_t nSamples, size_t level, const ImpurityType & imp) const
    {
        return ((nSamples < 2 * _par.minObservationsInLeafNode) || ((_par.maxTreeDepth > 0) && (level >= _par.maxTreeDepth)));
    }

    void featureValuesToBuf(size_t iFeature, algorithmFPType * featureVal, int * aIdx, size_t n)
    {
        _dataHelper.getColumnValues(iFeature, aIdx, n, featureVal);
        daal::algorithms::internal::qSort<algorithmFPType, int, cpu>(n, featureVal, aIdx);
    }

    void chooseFeatures(RowIndexType * featureSample)
    {
        const RowIndexType nFeat(nFeatures());
        {
            AUTOLOCK(_mtEngine);
            if (nFeaturesPerNode() * nFeaturesPerNode() < 2 * nFeat)
            {
                RNGsInst<int, cpu>().uniformWithoutReplacement(nFeaturesPerNode(), featureSample, featureSample + nFeaturesPerNode(),
                                                               _engine.getState(), 0, nFeat);
            }
            else
            {
                for (RowIndexType i = 0; i < nFeat; ++i) featureSample[i] = i;
                dtrees::training::internal::shuffle<cpu>(_engine.getState(), nFeat, featureSample);
            }
        }
    }

protected:
    typedef dtrees::internal::TVector<algorithmFPType, cpu> algorithmFPTypeArray;

    TrainBatchTaskBase(const NumericTable * x, const NumericTable * y, const Parameter & par, const dtrees::internal::FeatureTypes & featTypes,
                       const dtrees::internal::IndexedFeatures * indexedFeatures, engines::internal::BatchBaseImpl & engine, size_t nClasses)
        : _data(x),
          _resp(y),
          _par(par),
          _engine(engine),
          _nClasses(nClasses),
          _nSamples(par.observationsPerTreeFraction * x->getNumberOfRows()),
          _nFeaturesPerNode(par.featuresPerNode ? par.featuresPerNode : x->getNumberOfColumns()),
          _dataHelper(indexedFeatures),
          _featHelper(featTypes),
          _accuracy(daal::services::internal::EpsilonVal<algorithmFPType>::get()),
          _nTrees(nClasses > 2 ? nClasses : 1),
          _nThreadsMax(threader_get_threads_number()),
          _nParallelNodes(0)
    {
        int internalOptions = par.internalOptions;
        if (_nTrees < 2 || par.memorySavingMode) internalOptions &= ~parallelTrees; //clear parallelTrees flag
        _bThreaded = ((_nThreadsMax > 1) && ((internalOptions & parallelAll) != 0));
        if (_bThreaded)
        {
            _bParallelFeatures = !!(internalOptions & parallelFeatures);
            _bParallelNodes    = !!(internalOptions & parallelNodes);
            _bParallelTrees    = !!(internalOptions & parallelTrees);
        }
    }
    ~TrainBatchTaskBase()
    {
        delete _loss;
        _loss = nullptr;
    }

    virtual void initLossFunc()                                                                           = 0;
    virtual services::Status buildTrees(gbt::internal::GbtDecisionTree ** aTbl, HomogenNumericTable<double> ** aTblImp,
                                        HomogenNumericTable<int> ** aTblSmplCnt,
                                        GlobalStorages<algorithmFPType, BinIndexType, cpu> & GH_SUMS_BUF) = 0;
    virtual void step(const algorithmFPType * y)                                                          = 0;
    virtual bool getInitialF(algorithmFPType & val) { return false; }

    //loss function arguments (current estimation of y)
    algorithmFPType * f() { return _aF.get(); }
    const algorithmFPType * f() const { return _aF.get(); }

    void initializeF(algorithmFPType initValue)
    {
        const auto nRows = _data->getNumberOfRows();
        const auto nF    = nRows * _nTrees;
        //initialize f. TODO: input argument
        algorithmFPType * pf = f();
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < nF; ++i) pf[i] = initValue;
    }

public:
    daal::services::AtomicInt _nParallelNodes;

protected:
    DataHelperType _dataHelper;
    const FeatureTypes & _featHelper;
    TVector<algorithmFPType, cpu> _aF; //loss function arguments (f)
    //bagging, first _nSamples indices are the mapping of sample to row indices, the rest is OOB indices
    TVector<int, cpu> _aSampleToF;

    daal::Mutex _mtEngine;
    engines::internal::BatchBaseImpl & _engine;

    const NumericTable * _data;
    const NumericTable * _resp;
    const Parameter & _par;
    const RowIndexType _nSamples;
    const RowIndexType _nFeaturesPerNode;
    const int _nThreadsMax;

    algorithmFPType _accuracy;
    algorithmFPType _initialF = 0.0;
    size_t _nClasses;
    size_t _nTrees; //per iteration
    LossFunctionType * _loss = nullptr;

    bool _bThreaded         = false;
    bool _bParallelFeatures = false;
    bool _bParallelNodes    = false;
    bool _bParallelTrees    = false;
    bool _bIndirect         = true;
};

template <typename algorithmFPType, typename BinIndexType, CpuType cpu>
services::Status TrainBatchTaskBase<algorithmFPType, BinIndexType, cpu>::init()
{
    delete _loss;
    _loss = nullptr;
    initLossFunc();
    const auto nRows = _data->getNumberOfRows();
    if (_nSamples < nRows)
    {
        _aSampleToF.reset(nRows);
        DAAL_CHECK_MALLOC(_aSampleToF.get());
    }
    const auto nF = nRows * _nTrees;
    _aF.reset(nF);
    DAAL_CHECK_MALLOC(_aF.get());

    _bIndirect = true;

    return _dataHelper.init(_data, _resp, isIndirect() ? _aSampleToF.get() : (const int *)nullptr);
}

template <typename algorithmFPType, typename BinIndexType, CpuType cpu>
double TrainBatchTaskBase<algorithmFPType, BinIndexType, cpu>::computeLeafWeightUpdateF(const int * idx, size_t n, const ImpurityType & imp,
                                                                                        size_t iTree)
{
    double res          = _initialF;
    algorithmFPType val = imp.h + _par.lambda;
    if (isZero<algorithmFPType, cpu>(val)) return res;

    algorithmFPType * pf      = f();
    val                       = -imp.g / val;
    const algorithmFPType inc = val * _par.shrinkage;
    const size_t nThreads     = numAvailableThreads();
    const size_t nBlocks      = getNBlocksForOpt<cpu>(nThreads, n);
    const bool inParallel     = nBlocks > 1;
    const size_t nPerBlock    = n / nBlocks;
    const size_t nSurplus     = n % nBlocks;
    LoopHelper<cpu>::run(inParallel, nBlocks, [&](size_t iBlock) {
        const size_t start = iBlock + 1 > nSurplus ? nPerBlock * iBlock + nSurplus : (nPerBlock + 1) * iBlock;
        const size_t end   = iBlock + 1 > nSurplus ? start + nPerBlock : start + (nPerBlock + 1);
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = start; i < end; i++) pf[idx[i] * this->_nTrees + iTree] += inc;
    });

    return res + inc;
}

template <typename algorithmFPType, typename BinIndexType, CpuType cpu>
services::Status TrainBatchTaskBase<algorithmFPType, BinIndexType, cpu>::run(gbt::internal::GbtDecisionTree ** aTbl,
                                                                             HomogenNumericTable<double> ** aTblImp,
                                                                             HomogenNumericTable<int> ** aTblSmplCnt, size_t iIteration,
                                                                             GlobalStorages<algorithmFPType, BinIndexType, cpu> & GH_SUMS_BUF)
{
    for (size_t i = 0; i < _nTrees; ++i)
    {
        aTbl[i]        = nullptr;
        aTblImp[i]     = nullptr;
        aTblSmplCnt[i] = nullptr;
    }

    if (iIteration)
    {
        _initialF = 0;
    }
    else
    {
        if (!getInitialF(_initialF)) _initialF = algorithmFPType(0);
        initializeF(_initialF);
    }

    const size_t nRows = _data->getNumberOfRows();
    if (isBagging())
    {
        auto aSampleToF = _aSampleToF.get();
        for (size_t i = 0; i < nRows; ++i) aSampleToF[i] = i;
        {
            TVector<int, cpu> auxBuf(nRows);
            DAAL_CHECK_MALLOC(auxBuf.get());
            //no need to lock mutex here
            dtrees::training::internal::shuffle<cpu>(_engine.getState(), nRows, aSampleToF, auxBuf.get());
        }
        daal::algorithms::internal::qSort<RowIndexType, cpu>(nSamples(), aSampleToF);
    }
    step(this->_dataHelper.y());
    _nParallelNodes.set(0);
    return buildTrees(aTbl, aTblImp, aTblSmplCnt, GH_SUMS_BUF);
}

template <typename algorithmFPType, typename BinIndexType, CpuType cpu>
void TrainBatchTaskBase<algorithmFPType, BinIndexType, cpu>::updateOOB(size_t iTree, TreeType & t)
{
    const double res      = _initialF;
    const auto aSampleToF = _aSampleToF.get();
    auto pf               = f();
    const size_t n        = _aSampleToF.size();
    const size_t nIt      = n - _nSamples;
    daal::threader_for(nIt, nIt, [&](size_t i) {
        RowIndexType iRow = aSampleToF[i + _nSamples];
        ReadRows<algorithmFPType, cpu> x(const_cast<NumericTable *>(_dataHelper.data()), iRow, 1);
        auto pNode = dtrees::prediction::internal::findNode<algorithmFPType, TreeType, cpu>(t, x.get());
        DAAL_ASSERT(pNode);
        algorithmFPType inc = TreeType::NodeType::castLeaf(pNode)->response;
        // pf buffer was already initialized by _initialF before first iteration
        pf[iRow * _nTrees + iTree] += inc - res;
    });
}

//////////////////////////////////////////////////////////////////////////////////////////
// Base task class. Implements general pipeline of tree building
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, typename BinIndexType, CpuType cpu>
class TrainBatchTaskBaseXBoost : public TrainBatchTaskBase<algorithmFPType, BinIndexType, cpu>
{
public:
    typedef TrainBatchTaskBase<algorithmFPType, BinIndexType, cpu> super;
    typedef typename super::DataHelperType DataHelperType;
    typedef gh<algorithmFPType, cpu> ghType;

    TrainBatchTaskBaseXBoost(HostAppIface * hostApp, const NumericTable * x, const NumericTable * y, const Parameter & par,
                             const dtrees::internal::FeatureTypes & featTypes, const dtrees::internal::IndexedFeatures * indexedFeatures,
                             engines::internal::BatchBaseImpl & engine, size_t nClasses)
        : super(x, y, par, featTypes, indexedFeatures, engine, nClasses), _hostApp(hostApp)
    {}

    //loss function gradient and hessian values calculated in f() points
    ghType * grad(size_t iTree) { return _aGH.get() + iTree * this->_data->getNumberOfRows(); }
    void step(const algorithmFPType * y) DAAL_C11_OVERRIDE
    {
        this->lossFunc()->getGradients(this->_nSamples, this->_data->getNumberOfRows(), y, this->f(), this->aSampleToF(),
                                       (algorithmFPType *)_aGH.get());
    }
    virtual services::Status init() DAAL_C11_OVERRIDE
    {
        auto s = super::init();
        if (s)
        {
            _aGH.reset(this->_data->getNumberOfRows() * this->_nTrees);
            DAAL_CHECK_MALLOC(_aGH.get());
        }
        return s;
    }

protected:
    TVector<ghType, cpu> _aGH; //loss function first and second order derivatives
    HostAppIface * _hostApp;
};

template <typename algorithmFPType, typename RowIndexType, typename BinIndexType, CpuType cpu, typename TaskType, typename ResultType>
services::Status computeTypeDisp(HostAppIface * pHostApp, const NumericTable * x, const NumericTable * y, gbt::internal::ModelImpl & md,
                                 const gbt::training::Parameter & par, engines::internal::BatchBaseImpl & engine, size_t nClasses,
                                 dtrees::internal::IndexedFeatures & indexedFeatures, dtrees::internal::FeatureTypes & featTypes, ResultType * res,
                                 algorithmFPType * ptrWeight, algorithmFPType * ptrCover, algorithmFPType * ptrTotalCover, algorithmFPType * ptrGain,
                                 algorithmFPType * ptrTotalGain)
{
    services::Status s;

    const size_t nFeaturesPerNode = par.featuresPerNode ? par.featuresPerNode : x->getNumberOfColumns();
    const bool inexactWithHistMethod =
        !par.memorySavingMode && par.splitMethod == gbt::training::inexact && x->getNumberOfColumns() == nFeaturesPerNode;

    TaskType task(pHostApp, x, y, par, featTypes, par.memorySavingMode ? nullptr : &indexedFeatures, engine, nClasses);
    DAAL_CHECK_STATUS(s, task.init());

    const size_t nTrees = task.nTrees();
    DAAL_CHECK_MALLOC(md.reserve(par.maxIterations * nTrees));

    TVector<gbt::internal::GbtDecisionTree *, cpu> aTables;
    TVector<HomogenNumericTable<double> *, cpu> impTables;
    TVector<HomogenNumericTable<int> *, cpu> nodeSampleCountTables;

    typename gbt::internal::GbtDecisionTree * pTbl = nullptr;
    HomogenNumericTable<double> * pTblImp          = nullptr;
    HomogenNumericTable<int> * pTblSmplCnt         = nullptr;

    gbt::internal::GbtDecisionTree ** aTbl  = &pTbl;
    HomogenNumericTable<double> ** aTblImp  = &pTblImp;
    HomogenNumericTable<int> ** aTblSmplCnt = &pTblSmplCnt;

    if (nTrees > 1)
    {
        aTables.reset(nTrees);
        impTables.reset(nTrees);
        nodeSampleCountTables.reset(nTrees);

        DAAL_CHECK_MALLOC(aTables.get());
        DAAL_CHECK_MALLOC(impTables.get());
        DAAL_CHECK_MALLOC(nodeSampleCountTables.get());

        aTbl        = aTables.get();
        aTblImp     = impTables.get();
        aTblSmplCnt = nodeSampleCountTables.get();
    }

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, x->getNumberOfColumns(), sizeof(size_t));

    TVector<size_t, cpu, ScalableAllocator<cpu> > nUniquesArr(x->getNumberOfColumns());
    size_t * UniquesArr = nUniquesArr.get();
    DAAL_CHECK_MALLOC(UniquesArr);
    size_t nDiffFeatMax;
    if (!par.memorySavingMode)
    {
        nUniquesArr[0] = 0;
        nDiffFeatMax   = indexedFeatures.numIndices(0);
        for (size_t i = 1; i < x->getNumberOfColumns(); ++i)
        {
            nUniquesArr[i] = nUniquesArr[i - 1] + indexedFeatures.numIndices(i - 1);
            nDiffFeatMax += indexedFeatures.numIndices(i);
        }
    }

    const size_t initValue = (inexactWithHistMethod) ? 2 : 0;
    const size_t nStor     = x->getNumberOfColumns();

    GlobalStorages<algorithmFPType, BinIndexType, cpu> storage(x->getNumberOfColumns(), nStor, nDiffFeatMax, initValue);
    storage.nUniquesArr  = nUniquesArr;
    storage.nDiffFeatMax = nDiffFeatMax;

    if (!par.memorySavingMode)
    {
        for (size_t i = 0; i < x->getNumberOfColumns(); ++i)
        {
            storage.singleGHSums.add(i, indexedFeatures.numIndices(i), 2);
        }
    }

    TVector<BinIndexType, cpu, ScalableAllocator<cpu> > newFIArr;

    if (inexactWithHistMethod)
    {
        size_t nThreads    = threader_get_threads_number();
        size_t nRows       = x->getNumberOfRows();
        size_t nCols       = x->getNumberOfColumns();
        size_t nBlocks     = ((nThreads < nRows) ? nThreads : 1);
        size_t sizeOfBlock = nRows / nBlocks + !!(nRows % nBlocks);

        newFIArr.resize(nRows * nCols);
        BinIndexType * newFI = newFIArr.get();
        DAAL_CHECK_MALLOC(newFI);

        const dtrees::internal::IndexedFeatures::IndexType * fi = indexedFeatures.data(0);

        daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock) {
            const size_t iStart = iBlock * sizeOfBlock;
            const size_t iEnd   = (((iBlock + 1) * sizeOfBlock > nRows) ? nRows : iStart + sizeOfBlock);

            for (size_t i = iStart; i < iEnd; ++i)
            {
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j < nCols; ++j)
                {
                    newFI[nCols * i + j] = fi[nRows * j + i];
                }
            }
        });
        storage.newFI = newFI;
    }

    size_t * totalCoverFeature  = nullptr;
    size_t * weightFeature      = nullptr;
    double * totalGainFeature   = nullptr;
    algorithmFPType * allWeight = nullptr;

    TVector<algorithmFPType, cpu> allWeightVec(nStor, static_cast<algorithmFPType>(0));
    DAAL_CHECK_MALLOC(allWeightVec.get());
    allWeight = allWeightVec.get();

    for (size_t i = 0; (i < par.maxIterations) && !algorithms::internal::isCancelled(s, pHostApp); ++i)
    {
        s = task.run(aTbl, aTblImp, aTblSmplCnt, i, storage);
        if (!s)
        {
            deleteTables<cpu>(aTbl, aTblImp, aTblSmplCnt, nTrees);
            break;
        }
        size_t iTree = 0;
        for (; (iTree < nTrees) && (aTbl[iTree]->getNumberOfNodes() < 2); ++iTree)
            ;
        if (iTree == nTrees) //all are one level (constant response) trees
        {
            deleteTables<cpu>(aTbl, aTblImp, aTblSmplCnt, nTrees);
            break;
        }

        for (iTree = 0; iTree < nTrees; ++iTree)
        {
            if ((ptrTotalCover != nullptr) || (ptrCover != nullptr))
            {
                totalCoverFeature = (aTbl[iTree]->getArrayCoverFeature());
            }
            if ((ptrWeight != nullptr) || (ptrCover != nullptr) || (ptrGain != nullptr))
            {
                weightFeature = aTbl[iTree]->getArrayNumSplitFeature();
            }
            if ((ptrTotalGain != nullptr) || (ptrGain != nullptr))
            {
                totalGainFeature = (aTbl[iTree]->getArrayGainFeature());
            }

            if (ptrWeight != nullptr)
                for (size_t kFeature = 0; kFeature < nStor; ++kFeature) ptrWeight[kFeature] += static_cast<algorithmFPType>(weightFeature[kFeature]);

            if (ptrTotalCover != nullptr)
                for (size_t kFeature = 0; kFeature < nStor; ++kFeature)
                    ptrTotalCover[kFeature] += static_cast<algorithmFPType>(totalCoverFeature[kFeature]);

            if (ptrTotalGain != nullptr)
                for (size_t kFeature = 0; kFeature < nStor; ++kFeature)
                    ptrTotalGain[kFeature] += static_cast<algorithmFPType>(totalGainFeature[kFeature]);

            if (ptrCover != nullptr)
                for (size_t kFeature = 0; kFeature < nStor; ++kFeature)
                    ptrCover[kFeature] += static_cast<algorithmFPType>(totalCoverFeature[kFeature]);

            if (ptrGain != nullptr)
                for (size_t kFeature = 0; kFeature < nStor; ++kFeature) ptrGain[kFeature] += static_cast<algorithmFPType>(totalGainFeature[kFeature]);

            if ((ptrWeight != nullptr) || (ptrCover != nullptr) || (ptrGain != nullptr))
                for (size_t kFeature = 0; kFeature < nStor; ++kFeature) allWeight[kFeature] += static_cast<algorithmFPType>(weightFeature[kFeature]);

            md.add(aTbl[iTree], aTblImp[iTree], aTblSmplCnt[iTree]);
        }

        if ((i + 1 < par.maxIterations) && task.done()) break;
    }

    if (ptrCover != nullptr)
        for (size_t i = 0; i < nStor; ++i)
            if (allWeight[i] != 0) ptrCover[i] = ptrCover[i] / allWeight[i];

    if (ptrGain != nullptr)
        for (size_t i = 0; i < nStor; ++i)
            if (allWeight[i] != 0) ptrGain[i] = ptrGain[i] / allWeight[i];

    return s;
}

//////////////////////////////////////////////////////////////////////////////////////////
// compute() implementation
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu, typename BinIndexType, typename TaskType, typename ResultType>
services::Status computeImpl(HostAppIface * pHostApp, const NumericTable * x, const NumericTable * y, gbt::internal::ModelImpl & md,
                             const gbt::training::Parameter & par, engines::internal::BatchBaseImpl & engine, size_t nClasses,
                             dtrees::internal::IndexedFeatures & indexedFeatures, dtrees::internal::FeatureTypes & featTypes, ResultType * res,
                             algorithmFPType * ptrWeight, algorithmFPType * ptrCover, algorithmFPType * ptrTotalCover, algorithmFPType * ptrGain,
                             algorithmFPType * ptrTotalGain)

{
    return computeTypeDisp<algorithmFPType, int, BinIndexType, cpu, TaskType>(pHostApp, x, y, md, par, engine, nClasses, indexedFeatures, featTypes,
                                                                              res, ptrWeight, ptrCover, ptrTotalCover, ptrGain,
                                                                              ptrTotalGain); // TODO: remove int
}

} /* namespace internal */
} /* namespace training */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif
