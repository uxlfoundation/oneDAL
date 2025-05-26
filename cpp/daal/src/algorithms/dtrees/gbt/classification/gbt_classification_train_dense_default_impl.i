/* file: gbt_classification_train_dense_default_impl.i */
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
//  Implementation of auxiliary functions for gradient boosted trees classification
//  (defaultDense) method.
//--
*/

#ifndef __GBT_CLASSIFICATION_TRAIN_DENSE_DEFAULT_IMPL_I__
#define __GBT_CLASSIFICATION_TRAIN_DENSE_DEFAULT_IMPL_I__

#include "src/algorithms/dtrees/gbt/classification/gbt_classification_train_kernel.h"
#include "src/algorithms/dtrees/gbt/classification/gbt_classification_model_impl.h"
#include "src/algorithms/dtrees/gbt/gbt_train_dense_default_impl.i"
#include "src/algorithms/dtrees/gbt/gbt_train_tree_builder.i"
#include "src/algorithms/service_error_handling.h"
#include "src/services/service_algo_utils.h"

using namespace daal::algorithms::dtrees::training::internal;
using namespace daal::algorithms::gbt::training::internal;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace classification
{
namespace training
{
namespace internal
{
//////////////////////////////////////////////////////////////////////////////////////////
// Logistic loss function, L(y,f) = -[y*ln(sigmoid(f)) + (1 - y)*ln(1-sigmoid(f))]
// where sigmoid(f) = 1/(1 + exp(-f)
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class LogisticLoss : public LossFunction<algorithmFPType, cpu>
{
public:
    virtual void getGradients(size_t n, size_t nRows, const algorithmFPType * y, const algorithmFPType * f, const IndexType * sampleInd,
                              algorithmFPType * gh) DAAL_C11_OVERRIDE
    {
        TVector<algorithmFPType, cpu, ScalableAllocator<cpu> > aExp(n);
        auto exp                           = aExp.get();
        const algorithmFPType expThreshold = daal::internal::MathInst<algorithmFPType, cpu>::vExpThreshold();
        const size_t nThreads              = daal::threader_get_threads_number();
        const size_t nBlocks               = getNBlocksForOpt<cpu>(nThreads, n);
        const size_t nPerBlock             = n / nBlocks;
        const size_t nSurplus              = n % nBlocks;
        const bool inParallel              = nBlocks > 1;
        LoopHelper<cpu>::run(inParallel, nBlocks, [&](size_t iBlock) {
            const size_t start = iBlock + 1 > nSurplus ? nPerBlock * iBlock + nSurplus : (nPerBlock + 1) * iBlock;
            const size_t end   = iBlock + 1 > nSurplus ? start + nPerBlock : start + (nPerBlock + 1);
            if (sampleInd)
            {
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t i = start; i < end; i++)
                {
                    exp[i] = -f[sampleInd[i]];
                    /* make all values less than threshold as threshold value
                    to fix slow work on vExp on large negative inputs */
                    if (exp[i] < expThreshold) exp[i] = expThreshold;
                }
            }
            else
            {
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t i = start; i < end; i++)
                {
                    exp[i] = -f[i];
                    /* make all values less than threshold as threshold value
                    to fix slow work on vExp on large negative inputs */
                    if (exp[i] < expThreshold) exp[i] = expThreshold;
                }
            }
            daal::internal::MathInst<algorithmFPType, cpu>::vExp(end - start, exp + start, exp + start);
            if (sampleInd)
            {
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t i = start; i < end; i++)
                {
                    const algorithmFPType sigm = algorithmFPType(1.0) / (algorithmFPType(1.0) + exp[i]);
                    gh[2 * sampleInd[i]]       = sigm - y[sampleInd[i]];               //gradient
                    gh[2 * sampleInd[i] + 1]   = sigm * (algorithmFPType(1.0) - sigm); //hessian
                }
            }
            else
            {
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t i = start; i < end; i++)
                {
                    const auto sigm = algorithmFPType(1.0) / (algorithmFPType(1.0) + exp[i]);
                    gh[2 * i]       = sigm - y[i];                          //gradient
                    gh[2 * i + 1]   = sigm * (algorithmFPType(1.0) - sigm); //hessian
                }
            }
        });
    }
};

//////////////////////////////////////////////////////////////////////////////////////////
// Cross entropy loss function, L(y,f)=-sum(I(y=k)*ln(pk)) where pk = exp(fk)/sum(exp(f))
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, CpuType cpu>
class CrossEntropyLoss : public LossFunction<algorithmFPType, cpu>
{
public:
    CrossEntropyLoss(size_t numClasses) : _nClasses(numClasses) {}
    virtual void getGradients(size_t n, size_t nRows, const algorithmFPType * y, const algorithmFPType * f, const IndexType * sampleInd,
                              algorithmFPType * gh) DAAL_C11_OVERRIDE
    {
        static const size_t s_cMaxClassesBufSize = 12;
        const bool bUseTLS(_nClasses > s_cMaxClassesBufSize);
        daal::TlsMem<algorithmFPType, cpu> lsData(_nClasses);
        daal::threader_for(n, n, [&](size_t i) {
            algorithmFPType buf[s_cMaxClassesBufSize];
            algorithmFPType * p  = bUseTLS ? lsData.local() : buf;
            const size_t iSample = (sampleInd ? sampleInd[i] : i);
            getSoftmax(f + _nClasses * iSample, p);
            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t k = 0; k < _nClasses; ++k)
            {
                const algorithmFPType pk = p[k];
                const algorithmFPType h  = algorithmFPType(2.) * pk * (algorithmFPType(1.) - pk);
                algorithmFPType * gh_ik  = gh + 2 * (k * nRows + iSample);
                gh_ik[1]                 = h;
                if (size_t(y[iSample]) == k)
                    gh_ik[0] = (pk - algorithmFPType(1.));
                else
                    gh_ik[0] = pk;
            }
        });
    }

protected:
    void getSoftmax(const algorithmFPType * arg, algorithmFPType * res) const
    {
        const algorithmFPType expThreshold = daal::internal::MathInst<algorithmFPType, cpu>::vExpThreshold();
        algorithmFPType maxArg             = arg[0];
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 1; i < _nClasses; ++i)
        {
            if (maxArg < arg[i]) maxArg = arg[i];
        }
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < _nClasses; ++i)
        {
            res[i] = arg[i] - maxArg;
            /* make all values less than threshold as threshold value
            to fix slow work on vExp on large negative inputs */
            if (res[i] < expThreshold) res[i] = expThreshold;
        }
        daal::internal::MathInst<algorithmFPType, cpu>::vExp(_nClasses, res, res);
        algorithmFPType sum(0.);
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < _nClasses; ++i) sum += res[i];

        sum = algorithmFPType(1.) / sum;
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < _nClasses; ++i) res[i] *= sum;
    }

protected:
    size_t _nClasses;
};

//////////////////////////////////////////////////////////////////////////////////////////
// TrainBatchTask for classification
//////////////////////////////////////////////////////////////////////////////////////////
template <typename algorithmFPType, typename BinIndexType, gbt::classification::training::Method method, CpuType cpu>
class TrainBatchTask : public TrainBatchTaskBaseXBoost<algorithmFPType, BinIndexType, cpu>
{
    typedef TrainBatchTaskBaseXBoost<algorithmFPType, BinIndexType, cpu> super;
    typedef TreeBuilder<algorithmFPType, int, BinIndexType, cpu> TreeBuilderType; // TODO: fix int
    typedef ls<TreeBuilderType *> lsType;

public:
    TrainBatchTask(HostAppIface * pHostApp, const NumericTable * x, const NumericTable * y, const gbt::training::Parameter & par,
                   const dtrees::internal::FeatureTypes & featTypes, const dtrees::internal::IndexedFeatures * indexedFeatures,
                   engines::internal::BatchBaseImpl & engine, size_t nClasses)
        : super(pHostApp, x, y, par, featTypes, indexedFeatures, engine, nClasses), _builder(nullptr), _ls(nullptr)
    {}

    ~TrainBatchTask()
    {
        delete _builder;
        _builder = nullptr;
        if (_ls)
        {
            _ls->reduce([](TreeBuilderType * ptr) -> void {
                delete ptr;
                ptr = nullptr;
            });
            delete _ls;
            _ls = nullptr;
        }
    }

    bool done() { return false; }
    virtual services::Status init() DAAL_C11_OVERRIDE
    {
        auto s = super::init();
        if (!s) return s;
        if (this->isParallelTrees())
        {
            _ls = new lsType([=]() -> TreeBuilderType * { return TreeBuilderType::create(*this); });
            DAAL_CHECK_MALLOC(_ls);
            return s;
        }
        _builder = TreeBuilderType::create(*this);
        DAAL_CHECK_MALLOC(_builder);
        return _builder->init();
    }

protected:
    virtual void initLossFunc() DAAL_C11_OVERRIDE
    {
        switch (static_cast<const gbt::classification::training::Parameter &>(this->_par).loss)
        {
        case crossEntropy:
            if (this->_nClasses == 2)
                this->_loss = new LogisticLoss<algorithmFPType, cpu>();
            else
                this->_loss = new CrossEntropyLoss<algorithmFPType, cpu>(this->_nClasses);
            break;
        default: DAAL_ASSERT(false);
        }
    }

    virtual services::Status buildTrees(gbt::internal::GbtDecisionTree ** aTbl, HomogenNumericTable<double> ** aTblImp,
                                        HomogenNumericTable<int> ** aTblSmplCnt,
                                        GlobalStorages<algorithmFPType, BinIndexType, cpu> & GH_SUMS_BUF) DAAL_C11_OVERRIDE
    {
        if (this->isParallelTrees())
        {
            this->_nParallelNodes.set(this->_nTrees); //highest level parallelization first
            daal::SafeStatus safeStat;
            daal::threader_for(this->_nTrees, this->_nTrees, [&](size_t i) {
                if (safeStat)
                    safeStat |= buildTreeThreadLocal(aTbl[i], aTblImp[i], aTblSmplCnt[i], i, GH_SUMS_BUF);
                else
                    return;
                this->_nParallelNodes.dec(); //allow lower levels of parallelization
            });

            return safeStat.detach();
        }

        services::Status s;
        for (size_t i = 0; s.ok() && (i < this->_nTrees) && !daal::algorithms::internal::isCancelled(s, this->_hostApp); ++i)
        {
            DAAL_ASSERT(this->_nParallelNodes.get() == 0);
            this->_nParallelNodes.inc();
            s |= _builder->run(aTbl[i], aTblImp[i], aTblSmplCnt[i], i, GH_SUMS_BUF);
            this->_nParallelNodes.dec();
            DAAL_ASSERT(this->_nParallelNodes.get() == 0);
        }
        return s;
    }

    services::Status buildTreeThreadLocal(gbt::internal::GbtDecisionTree *& tbl, HomogenNumericTable<double> *& pTblImp,
                                          HomogenNumericTable<int> *& pTblSmplCnt, size_t iTree,
                                          GlobalStorages<algorithmFPType, BinIndexType, cpu> & GH_SUMS_BUF)
    {
        auto pBuilder = _ls->local();
        DAAL_CHECK_MALLOC(pBuilder);
        services::Status s;
        if ((pBuilder->isInitialized() || (s = pBuilder->init()).ok()) && !algorithms::internal::isCancelled(s, this->_hostApp))
            s = pBuilder->run(tbl, pTblImp, pTblSmplCnt, iTree, GH_SUMS_BUF);
        _ls->release(pBuilder);
        if (s) algorithms::internal::isCancelled(s, this->_hostApp);
        return s;
    }

protected:
    TreeBuilder<algorithmFPType, int, BinIndexType, cpu> * _builder; // TODO: replace int
    lsType * _ls;
};

//////////////////////////////////////////////////////////////////////////////////////////
// ClassificationTrainBatchKernel
//////////////////////////////////////////////////////////////////////////////////////////

template <typename algorithmFPType, gbt::classification::training::Method method, CpuType cpu>
services::Status ClassificationTrainBatchKernel<algorithmFPType, method, cpu>::compute(HostAppIface * pHost, const NumericTable * x,
                                                                                       const NumericTable * y, gbt::classification::Model & m,
                                                                                       Result & res, const Parameter & par,
                                                                                       engines::internal::BatchBaseImpl & engine)
{
    const size_t nFeaturesPerNode = par.featuresPerNode ? par.featuresPerNode : x->getNumberOfColumns();
    const bool inexactWithHistMethod =
        !par.memorySavingMode && par.splitMethod == gbt::training::inexact && x->getNumberOfColumns() == nFeaturesPerNode;

    services::Status s;
    dtrees::internal::IndexedFeatures indexedFeatures;
    dtrees::internal::FeatureTypes featTypes;
    DAAL_CHECK_MALLOC(featTypes.init(*x));

    if (!par.memorySavingMode)
    {
        BinParams prm(par.maxBins, par.minBinSize);
        DAAL_CHECK_STATUS(s,
                          (indexedFeatures.init<algorithmFPType, cpu>(*x, &featTypes, par.splitMethod == gbt::training::inexact ? &prm : nullptr)));
    }

    WriteOnlyRows<algorithmFPType, cpu> weightsRows, totalCoverRows, coverRows, totalGainRows, gainRows;
    const gbt::classification::training::interface2::Parameter * parPtr =
        dynamic_cast<const gbt::classification::training::interface2::Parameter *>(&par);

    if (parPtr != nullptr)
    {
        if (par.varImportance & gbt::training::weight)
        {
            weightsRows.set(res.get(variableImportanceByWeight).get(), 0, 1);
        }
        if (par.varImportance & gbt::training::totalCover)
        {
            totalCoverRows.set(res.get(variableImportanceByTotalCover).get(), 0, 1);
        }
        if (par.varImportance & gbt::training::cover)
        {
            coverRows.set(res.get(variableImportanceByCover).get(), 0, 1);
        }
        if (par.varImportance & gbt::training::totalGain)
        {
            totalGainRows.set(res.get(variableImportanceByTotalGain).get(), 0, 1);
        }
        if (par.varImportance & gbt::training::gain)
        {
            gainRows.set(res.get(variableImportanceByGain).get(), 0, 1);
        }
    }

    algorithmFPType * ptrWeight     = weightsRows.get();
    algorithmFPType * ptrTotalCover = totalCoverRows.get();
    algorithmFPType * ptrCover      = coverRows.get();
    algorithmFPType * ptrTotalGain  = totalGainRows.get();
    algorithmFPType * ptrGain       = gainRows.get();

    if (inexactWithHistMethod)
    {
        if (indexedFeatures.maxNumIndices() <= 256)
            return computeImpl<algorithmFPType, cpu, uint8_t, TrainBatchTask<algorithmFPType, uint8_t, method, cpu>, Result>(
                pHost, x, y, *static_cast<daal::algorithms::gbt::classification::internal::ModelImpl *>(&m), par, engine, par.nClasses,
                indexedFeatures, featTypes, &res, ptrWeight, ptrCover, ptrTotalCover, ptrGain, ptrTotalGain);
        else if (indexedFeatures.maxNumIndices() <= 65536)
            return computeImpl<algorithmFPType, cpu, uint16_t, TrainBatchTask<algorithmFPType, uint16_t, method, cpu>, Result>(
                pHost, x, y, *static_cast<daal::algorithms::gbt::classification::internal::ModelImpl *>(&m), par, engine, par.nClasses,
                indexedFeatures, featTypes, &res, ptrWeight, ptrCover, ptrTotalCover, ptrGain, ptrTotalGain);
        else
            return computeImpl<algorithmFPType, cpu, uint32_t, TrainBatchTask<algorithmFPType, uint32_t, method, cpu>, Result>(
                pHost, x, y, *static_cast<daal::algorithms::gbt::classification::internal::ModelImpl *>(&m), par, engine, par.nClasses,
                indexedFeatures, featTypes, &res, ptrWeight, ptrCover, ptrTotalCover, ptrGain, ptrTotalGain);
    }
    else
    {
        return computeImpl<algorithmFPType, cpu, uint32_t, TrainBatchTask<algorithmFPType, uint32_t, method, cpu>, Result>(
            pHost, x, y, *static_cast<daal::algorithms::gbt::classification::internal::ModelImpl *>(&m), par, engine, par.nClasses, indexedFeatures,
            featTypes, &res, ptrWeight, ptrCover, ptrTotalCover, ptrGain, ptrTotalGain);
    }
}

} /* namespace internal */
} /* namespace training */
} /* namespace classification */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif
