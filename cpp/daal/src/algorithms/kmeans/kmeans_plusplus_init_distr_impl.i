/* file: kmeans_plusplus_init_distr_impl.i */
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
//  Implementation of Lloyd method for K-means algorithm.
//--
*/

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "src/threading/threading.h"
#include "services/daal_defines.h"
#include "src/externals/service_memory.h"
#include "src/algorithms/service_error_handling.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_sort.h"
#include "src/algorithms/kmeans/kmeans_plusplus_init_impl.i"
#include "src/algorithms/distributions/uniform/uniform_kernel.h"
#include "src/services/service_data_utils.h"

using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::algorithms::distributions::uniform::internal;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu, typename DataHelper>
class TaskStep2Local
{
public:
    TaskStep2Local(NumericTable * ntData) : _data(ntData) {}
    Status run(size_t iFirstOfNewCandidates, const NumericTable * pNewCenters, NumericTable * pMinDist, algorithmFPType & overallError)
    {
        auto nBlocks = _data.nRows / _nRowsInBlock;
        nBlocks += (nBlocks * _nRowsInBlock != _data.nRows);
        TArray<algorithmFPType, cpu> aMinDistAcc(nBlocks); //accumulated minimal distance per every block
        DAAL_CHECK(aMinDistAcc.get(), ErrorMemoryAllocationFailed);

        ReadRows<algorithmFPType, cpu> newCenterBD(const_cast<NumericTable *>(pNewCenters), 0, pNewCenters->getNumberOfRows());
        DAAL_CHECK_BLOCK_STATUS(newCenterBD);
        WriteOnlyRows<algorithmFPType, cpu> minDistBD(pMinDist, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(minDistBD);

        return updateMinDist(iFirstOfNewCandidates, newCenterBD.get(), pNewCenters->getNumberOfRows(), minDistBD.get(), aMinDistAcc.get(),
                             overallError, nBlocks);
    }

protected:
    virtual Status updateMinDist(size_t iFirstOfNewCandidates, const algorithmFPType * pNewCenters, size_t nNewCandidates, algorithmFPType * aMinDist,
                                 algorithmFPType * aMinDistAcc, algorithmFPType & overallError, size_t nBlocks)
    {
        DAAL_ASSERT(nNewCandidates == 1);
        SafeStatus safeStat;
        daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock) {
            safeStat |= _data.updateMinDistInBlock(aMinDistAcc, nBlocks, iBlock, 1, 0, nullptr, pNewCenters, aMinDist);
        });
        if (!safeStat) return safeStat.detach();
        overallError = aMinDistAcc[0];
        for (size_t iBlock = 1; iBlock < nBlocks; ++iBlock) overallError += aMinDistAcc[iBlock];
        return Status();
    }

protected:
    DataHelper _data;
};

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
class TaskStep2LocalParallelPlus : public TaskStep2Local<algorithmFPType, cpu, DataHelper>
{
public:
    TaskStep2LocalParallelPlus(NumericTable * ntData, NumericTable ** aLocalData)
        : TaskStep2Local<algorithmFPType, cpu, DataHelper>(ntData), _aLocalData(aLocalData)
    {}

protected:
    virtual Status updateMinDist(size_t iFirstOfNewCandidates, const algorithmFPType * pNewCenters, size_t nNewCandidates, algorithmFPType * aMinDist,
                                 algorithmFPType * aMinDistAcc, algorithmFPType & overallError, size_t nBlocks) override
    {
        WriteRows<int, cpu> closestClusterBD(_aLocalData[internal::closestCluster], 0, 1);
        DAAL_CHECK_BLOCK_STATUS(closestClusterBD);
        WriteRows<int, cpu> candidateRatingBD(_aLocalData[internal::candidateRating], 0, 1);
        DAAL_CHECK_BLOCK_STATUS(candidateRatingBD);
        TArray<algorithmFPType, cpu> lastAddedCenterNorm2(nNewCandidates);
        DAAL_CHECK(lastAddedCenterNorm2.get(), ErrorMemoryAllocationFailed);

        //calculate
        daal::threader_for(nNewCandidates, nNewCandidates, [&](size_t iPt) {
            const algorithmFPType * pCenter = pNewCenters + this->_data.dim * iPt;
            algorithmFPType val             = 0;
            for (size_t i = 0; i < this->_data.dim; ++i) val += pCenter[i] * pCenter[i];
            lastAddedCenterNorm2.get()[iPt] = algorithmFPType(0.5) * val;
        });

        overallError = algorithmFPType(0.);
        TaskParallelPlusUpdateDist<algorithmFPType, cpu, DataHelper> impl(nBlocks, candidateRatingBD.get(), closestClusterBD.get(), overallError,
                                                                          this->_data, pNewCenters, lastAddedCenterNorm2.get(), aMinDist,
                                                                          aMinDistAcc);
        return impl.updateMinDist(iFirstOfNewCandidates, nNewCandidates);
    }

protected:
    NumericTable ** _aLocalData;
};

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status KMeansInitStep2LocalKernel<method, algorithmFPType, cpu>::compute(const DistributedStep2LocalPlusPlusParameter * par,
                                                                                   const NumericTable * ntData, const NumericTable * pNewCenters,
                                                                                   NumericTable ** aLocalData, NumericTable * pRes,
                                                                                   NumericTable * pOutputForStep5)
{
    WriteRows<int, cpu> numClustersBD(aLocalData[internal::numberOfClusters], 0, 1);
    DAAL_CHECK_BLOCK_STATUS(numClustersBD);
    const auto nRows                   = ntData->getNumberOfRows();
    bool bInitial                      = false;
    const size_t iFirstOfNewCandidates = numClustersBD.get()[0];
    int result                         = 0;

    if (iFirstOfNewCandidates == 0)
    {
        WriteOnlyRows<algorithmFPType, cpu> minDistBD(aLocalData[internal::closestClusterDistance], 0, 1);
        DAAL_CHECK_BLOCK_STATUS(minDistBD);
        daal::services::internal::service_memset<algorithmFPType, cpu>(minDistBD.get(), daal::services::internal::MaxVal<algorithmFPType>::get(),
                                                                       nRows);
        if (isParallelPlusMethod(method))
        {
            WriteOnlyRows<int, cpu> closestClusterBD(aLocalData[internal::closestCluster], 0, 1);
            DAAL_CHECK_BLOCK_STATUS(closestClusterBD);
            daal::services::internal::service_memset<int, cpu>(closestClusterBD.get(), 0, nRows);
            WriteOnlyRows<int, cpu> ratingBD(aLocalData[internal::candidateRating], 0, 1);
            DAAL_CHECK_BLOCK_STATUS(ratingBD);
            daal::services::internal::service_memset<int, cpu>(ratingBD.get(), 0, aLocalData[internal::candidateRating]->getNumberOfColumns());
            *ratingBD.get() = nRows;
        }
        bInitial = true;
    }
    algorithmFPType overallError(0.);
    NumericTable * pMinDist = aLocalData[internal::closestClusterDistance];
    NumericTable * pData    = const_cast<NumericTable *>(ntData);
    Status s;
    if ((method == plusPlusDense) || (bInitial && (method == parallelPlusDense)))
    {
        s = TaskStep2Local<algorithmFPType, cpu, DataHelperDense<algorithmFPType, cpu> >(pData).run(iFirstOfNewCandidates, pNewCenters, pMinDist,
                                                                                                    overallError);
    }
    else if (method == plusPlusCSR || (bInitial && (method == parallelPlusCSR)))
    {
        s = TaskStep2Local<algorithmFPType, cpu, DataHelperCSR<algorithmFPType, cpu> >(pData).run(iFirstOfNewCandidates, pNewCenters, pMinDist,
                                                                                                  overallError);
    }
    else if (method == parallelPlusDense)
    {
        s = TaskStep2LocalParallelPlus<algorithmFPType, cpu, DataHelperDense<algorithmFPType, cpu> >(pData, aLocalData)
                .run(iFirstOfNewCandidates, pNewCenters, pMinDist, overallError);
    }
    else if (method == parallelPlusCSR)
    {
        s = TaskStep2LocalParallelPlus<algorithmFPType, cpu, DataHelperCSR<algorithmFPType, cpu> >(pData, aLocalData)
                .run(iFirstOfNewCandidates, pNewCenters, pMinDist, overallError);
    }
    else
    {
        DAAL_ASSERT(false); //should never happen
        return Status();
    }
    if (!s) return s;
    WriteRows<algorithmFPType, cpu> resBD(pRes, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(resBD);
    resBD.get()[0] = overallError;
    numClustersBD.get()[0] += pNewCenters->getNumberOfRows();
    if (pOutputForStep5)
    {
        const auto nCols = aLocalData[internal::candidateRating]->getNumberOfColumns();
        ReadRows<int, cpu> candidateRatingBD(aLocalData[internal::candidateRating], 0, 1);
        DAAL_CHECK_BLOCK_STATUS(candidateRatingBD);
        WriteRows<int, cpu> outputForStep5BD(pOutputForStep5, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(outputForStep5BD);
        result = daal::services::internal::daal_memcpy_s(outputForStep5BD.get(), sizeof(int) * nCols, candidateRatingBD.get(), sizeof(int) * nCols);
    }
    return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
}

template <typename algorithmFPType, CpuType cpu>
struct KeyValue
{
    size_t key;
    algorithmFPType value;

    KeyValue() {}
    void set(size_t k, algorithmFPType v)
    {
        key   = k;
        value = v;
    }
    bool operator<=(const KeyValue & o) const { return (key < o.key) || ((key == o.key) && (value <= o.value)); }
    bool operator<(const KeyValue & o) const { return (key < o.key) || ((key == o.key) && (value < o.value)); }
    bool operator>(const KeyValue & o) const { return (key > o.key) || ((key == o.key) && (value > o.value)); }
};

template <typename algorithmFPType, CpuType cpu>
size_t findSampleValue(algorithmFPType & sample, size_t iStart, size_t nInput, const KeyValue<algorithmFPType, cpu> * aInput)
{
    //find the block this sample belongs to
    size_t i = iStart;
    for (; (i + 1 < nInput) && (sample >= aInput[i].value); ++i) sample -= aInput[i].value;
    return i;
}

template <typename algorithmFPType, CpuType cpu>
Status createTableSingleRow(SerializationIfacePtr & pRes, size_t nCols, const algorithmFPType * aVal)
{
    Status st;
    int result = 0;

    NumericTablePtr pTbl = HomogenNumericTableCPU<algorithmFPType, cpu>::create(nCols, 1, &st);
    DAAL_CHECK_STATUS_VAR(st);
    WriteOnlyRows<algorithmFPType, cpu> tblBD(pTbl.get(), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(tblBD);
    result = daal::services::internal::daal_memcpy_s(tblBD.get(), sizeof(algorithmFPType) * nCols, aVal, sizeof(algorithmFPType) * nCols);
    pRes   = staticPointerCast<SerializationIface, NumericTable>(pTbl);

    return (!result) ? st : services::Status(services::ErrorMemoryCopyFailedInternal);
};

template <Method method, typename algorithmFPType, CpuType cpu>
Status KMeansInitStep3MasterKernel<method, algorithmFPType, cpu>::init(const Parameter * par, MemoryBlock * pRngState, engines::BatchBase & engine)
{
    _rngState = pRngState;
    if (!_rngState) return Status();
    if (_isFirstIteration) // first iteration?
    {
        if (!_rngState->get())
        {
            _isFirstIteration = false;
            algorithmFPType dummy;
            auto engineImpl = dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl *>(&engine);
            Status s;
            DAAL_CHECK(engineImpl, ErrorIncorrectEngineParameter);
            DAAL_CHECK_STATUS(
                s, (UniformKernelDefault<algorithmFPType, cpu>::compute(algorithmFPType(0.), algorithmFPType(1.), *engineImpl, 1, &dummy)));
            _rngState->reserve(engineImpl->getStateSize());
            return engine.saveState(_rngState->get());
        }
    }
    return engine.loadState(_rngState->get());
}

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status KMeansInitStep3MasterKernel<method, algorithmFPType, cpu>::compute(const Parameter * par, const KeyValueDataCollection * pInputColl,
                                                                                    MemoryBlock * pRngState, KeyValueDataCollection * pOutputColl,
                                                                                    engines::BatchBase & engine)
{
    pOutputColl->clear();
    Status s = init(par, pRngState, engine);
    if (!s) return s;
    typedef KeyValue<algorithmFPType, cpu> TKeyValue;
    TArray<TKeyValue, cpu> aInput(pInputColl->size());
    const size_t outputSize =
        ((method == kmeans::init::plusPlusDense) || (method == kmeans::init::plusPlusCSR)) ? 1 : size_t(par->oversamplingFactor * par->nClusters);
    TNArray<TKeyValue, 1, cpu> aOutput(outputSize);
    TNArray<algorithmFPType, 1, cpu> aRngValues(outputSize);
    DAAL_CHECK(aInput.get() && aOutput.get() && aRngValues.get(), ErrorMemoryAllocationFailed);

    //fill in input structures and calculate overall error value
    algorithmFPType overallError = 0;
    for (size_t i = 0; i < pInputColl->size(); ++i)
    {
        const size_t key     = pInputColl->getKeyByIndex(i);
        NumericTablePtr pTbl = NumericTable::cast(pInputColl->getValueByIndex(i));
        ReadRows<algorithmFPType, cpu> tblBD(pTbl.get(), 0, 1);
        const algorithmFPType value = *tblBD.get();
        DAAL_CHECK(value >= 0, ErrorIncorrectValueInTheNumericTable);
        aInput[i].key   = key;
        aInput[i].value = value;
        overallError += value;
    }

    DAAL_CHECK_STATUS(s,
                      (UniformKernelDefault<algorithmFPType, cpu>::compute(algorithmFPType(0.), overallError, engine, outputSize, aRngValues.get())));
    DAAL_CHECK_STATUS(s, engine.saveState(_rngState->get()));
    if (outputSize > 1) daal::algorithms::internal::qSort<algorithmFPType, cpu>(outputSize, aRngValues.get());

    algorithmFPType sample = *aRngValues.get();
    size_t iInputStart     = 0;
    for (size_t i = 0; i < outputSize; ++i)
    {
        size_t iInput = findSampleValue<algorithmFPType, cpu>(sample, iInputStart, pInputColl->size(), aInput.get());
        aOutput[i].set(aInput[iInput].key, sample);
        if ((i + 1) == outputSize) break;
        sample += aRngValues[i + 1] - aRngValues[i];
        iInputStart = iInput;
    }
    const TKeyValue * pOutput = aOutput.get();
    if (outputSize == 1)
    {
        s = createTableSingleRow<algorithmFPType, cpu>((*pOutputColl)[pOutput->key], 1, &(pOutput->value));
        if (!s) return s;
    }
    for (size_t i = 0; i < outputSize;)
    {
        const auto key         = pOutput[i].key;
        algorithmFPType * pTmp = aRngValues.get();
        size_t n               = 1;
        pTmp[0]                = pOutput[i].value;
        for (size_t j = i + n; (j < outputSize) && (pOutput[j].key == key); pTmp[n++] = pOutput[j++].value)
            ;
        s = createTableSingleRow<algorithmFPType, cpu>((*pOutputColl)[key], n, pTmp);
        if (!s) return s;
        i += n;
    }
    return s;
}

template <typename algorithmFPType, CpuType cpu>
size_t findSample(algorithmFPType & sample, size_t iStart, size_t nInput, const algorithmFPType * aInput)
{
    //find the block this sample belongs to
    size_t i = iStart;
    for (; (i + 1 < nInput) && (sample >= aInput[i]); ++i) sample -= aInput[i];
    return i;
}

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status KMeansInitStep4LocalKernel<method, algorithmFPType, cpu>::compute(const NumericTable * pData, const NumericTable * pInput,
                                                                                   NumericTable ** aLocalData, NumericTable * pOutput)
{
    WriteRows<algorithmFPType, cpu> inputBD(const_cast<NumericTable *>(pInput), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(inputBD);
    const auto outputSize = pInput->getNumberOfColumns();
    int result            = 0;
    if (outputSize > 1) daal::algorithms::internal::qSort<algorithmFPType, cpu>(outputSize, inputBD.get());

    const algorithmFPType * aSamples = inputBD.get();
    const auto dataSize              = aLocalData[internal::closestClusterDistance]->getNumberOfColumns();
    ReadRows<algorithmFPType, cpu> minDistBD(aLocalData[internal::closestClusterDistance], 0, 1);
    DAAL_CHECK_BLOCK_STATUS(minDistBD);

    const auto nFeatures = pData->getNumberOfColumns();
    WriteOnlyRows<algorithmFPType, cpu> outputBD;
    ReadRows<algorithmFPType, cpu> dataBD;

    algorithmFPType sample = *aSamples;
    size_t iInputStart     = 0;
    for (size_t i = 0; i < outputSize; ++i)
    {
        size_t iRow = findSample<algorithmFPType, cpu>(sample, iInputStart, dataSize, minDistBD.get());
        outputBD.set(pOutput, i, 1);
        DAAL_CHECK_BLOCK_STATUS(outputBD);
        dataBD.set(const_cast<NumericTable *>(pData), iRow, 1);
        DAAL_CHECK_BLOCK_STATUS(dataBD);
        result |= daal::services::internal::daal_memcpy_s(outputBD.get(), sizeof(algorithmFPType) * nFeatures, dataBD.get(),
                                                          sizeof(algorithmFPType) * nFeatures);
        if ((i + 1) == outputSize) break;
        sample += aSamples[i + 1] - aSamples[i];
        iInputStart = iRow;
    }
    return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
}

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status KMeansInitStep5MasterKernel<method, algorithmFPType, cpu>::compute(const data_management::DataCollection * pCandidates,
                                                                                    const data_management::DataCollection * pRating,
                                                                                    NumericTable * pResCand, NumericTable * pResWeights)
{
    size_t iRow = 0;
    int result  = 0;

    for (size_t i = 0; i < pCandidates->size(); ++i)
    {
        NumericTable * pTbl = static_cast<NumericTable *>((*pCandidates)[i].get());
        const size_t nRows  = pTbl->getNumberOfRows();
        ReadRows<algorithmFPType, cpu> tblBD(pTbl, 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(tblBD);
        WriteRows<algorithmFPType, cpu> resCandBD(pResCand, iRow, nRows);
        DAAL_CHECK_BLOCK_STATUS(resCandBD);
        const size_t sz = sizeof(algorithmFPType) * pTbl->getNumberOfColumns() * nRows;
        result |= daal::services::internal::daal_memcpy_s(resCandBD.get(), sz, tblBD.get(), sz);
        iRow += nRows;
    }
    const size_t nCols = pResWeights->getNumberOfColumns();
    WriteRows<algorithmFPType, cpu> weightBD(pResWeights, 0, 1);
    DAAL_CHECK_BLOCK_STATUS(weightBD);
    algorithmFPType * aWt = weightBD.get();
    for (size_t i = 0; i < pRating->size(); ++i)
    {
        NumericTable * pTbl = static_cast<NumericTable *>((*pRating)[i].get());
        ReadRows<int, cpu> tblBD(pTbl, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(tblBD);
        if (i == 0)
        {
            for (size_t j = 0; j < nCols; ++j) aWt[j] = algorithmFPType(tblBD.get()[j]);
        }
        else
        {
            for (size_t j = 0; j < nCols; ++j) aWt[j] += algorithmFPType(tblBD.get()[j]);
        }
    }
    algorithmFPType total = 0;
    for (size_t j = 0; j < nCols; ++j) total += aWt[j];
    total = algorithmFPType(1.) / total;
    for (size_t j = 0; j < nCols; ++j) aWt[j] *= total;
    return (!result) ? services::Status() : services::Status(services::ErrorMemoryCopyFailedInternal);
}

template <Method method, typename algorithmFPType, CpuType cpu>
services::Status KMeansInitStep5MasterKernel<method, algorithmFPType, cpu>::finalizeCompute(const Parameter * par, const NumericTable * ntCand,
                                                                                            const NumericTable * ntWeights,
                                                                                            const MemoryBlock * pRngState, NumericTable * pCentroids,
                                                                                            engines::BatchBase & engine)
{
    ReadRows<algorithmFPType, cpu> weightsBD(const_cast<NumericTable *>(ntWeights), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(weightsBD);
    Status s;
    DAAL_CHECK_STATUS(s, engine.loadState(pRngState->get()));

    const size_t nLocalTrials = 1u;
    TaskPlusPlusBatch<algorithmFPType, cpu, DataHelperDense<algorithmFPType, cpu> > task(const_cast<NumericTable *>(ntCand), weightsBD.get(),
                                                                                         pCentroids, par->nClusters, nLocalTrials, engine);
    return task.run();
}

} // namespace internal
} // namespace init
} // namespace kmeans
} // namespace algorithms
} // namespace daal
