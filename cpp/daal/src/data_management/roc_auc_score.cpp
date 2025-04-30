/** file roc_auc_score.cpp */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "data_management/data/internal/roc_auc_score.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "src/algorithms/service_kernel_math.h"
#include "src/algorithms/service_sort.h"
#include "src/externals/service_dispatch.h"
#include "src/externals/service_memory.h"
#include "src/services/service_data_utils.h"
#include "src/threading/threading.h"

namespace daal
{
namespace data_management
{
namespace internal
{
using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;

template <typename DataType, daal::CpuType cpu>
services::Status rocAucScoreImpl(const NumericTablePtr & truePrediction, const NumericTablePtr & testPrediction, double & score)
{
    services::Status s;
    SafeStatus safeStat;
    const size_t nElements = truePrediction->getNumberOfRows();
    TArrayScalable<IdxValType<DataType>, cpu> predict(nElements);
    DAAL_CHECK_MALLOC(predict.get());

    const size_t blockSizeDefault = 256;
    const size_t nBlocks          = nElements / blockSizeDefault + !!(nElements % blockSizeDefault);

    ReadColumns<DataType, cpu> testPredictionBlock(testPrediction.get(), 0, 0, nElements);
    DAAL_CHECK_BLOCK_STATUS(testPredictionBlock);
    const DataType * const testPredictionPtr = testPredictionBlock.get();

    daal::threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
        const size_t blockBegin = iBlock * blockSizeDefault;
        const size_t blockSize  = (iBlock == nBlocks - 1) ? nElements - blockBegin : blockSizeDefault;

        for (size_t i = 0; i < blockSize; ++i)
        {
            const size_t idx   = blockBegin + i;
            predict[idx].value = testPredictionPtr[idx];
            predict[idx].index = idx;
        }
    });

    daal::parallel_sort<DataType>(predict.get(), predict.get() + nElements);

    size_t rank            = 1;
    size_t elementsInBlock = 1;
    size_t i               = 0;
    TArray<double, cpu> predictedRank(nElements);
    DAAL_CHECK_MALLOC(predictedRank.get());
    while (i < nElements)
    {
        size_t j = i;
        while ((j < (nElements - 1)) && (predict[j].value == predict[j + 1].value))
        {
            j++;
        }
        elementsInBlock = j - i + 1;

        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < elementsInBlock; ++j)
        {
            const size_t idx   = predict[i + j].index;
            predictedRank[idx] = static_cast<double>(rank) + ((static_cast<double>(elementsInBlock) - double(1.0)) * double(0.5));
        }
        rank += elementsInBlock;
        i += elementsInBlock;
    }

    double nPos            = double(0);
    double filteredRankSum = double(0);
    for (size_t iBlock = 0; iBlock < nBlocks; ++iBlock)
    {
        const size_t blockBegin = iBlock * blockSizeDefault;
        const size_t blockSize  = (iBlock == nBlocks - 1) ? nElements - blockBegin : blockSizeDefault;
        ReadColumns<int, cpu> truePredictionBlock(truePrediction.get(), 0, blockBegin, blockSize);
        DAAL_CHECK_BLOCK_STATUS(truePredictionBlock);
        const int * const truePredictionPtr = truePredictionBlock.get();
        for (size_t i = 0; i < blockSize; ++i)
        {
            nPos += truePredictionPtr[i];
            if (truePredictionPtr[i] == 1)
            {
                filteredRankSum += predictedRank[i + blockBegin];
            }
        }
    }
    const double nNeg = static_cast<double>(nElements) - nPos;
    score             = (filteredRankSum - (nPos * (nPos + double(1.0)) * double(0.5))) / (nPos * nNeg);
    return s;
}

template <typename DataType>
DAAL_EXPORT double rocAucScore(const NumericTablePtr & truePrediction, const NumericTablePtr & testPrediction)
{
    double score = double(0);
#define DAAL_ROC_AUC_SCORE(cpuId, ...) rocAucScoreImpl<DataType, cpuId>(__VA_ARGS__);

    DAAL_DISPATCH_FUNCTION_BY_CPU(DAAL_ROC_AUC_SCORE, truePrediction, testPrediction, score);

#undef DAAL_ROC_AUC_SCORE
    return score;
}

template DAAL_EXPORT double rocAucScore<float>(const NumericTablePtr & truePrediction, const NumericTablePtr & testPrediction);
template DAAL_EXPORT double rocAucScore<double>(const NumericTablePtr & truePrediction, const NumericTablePtr & testPrediction);

// necessary for maintaining ABI
DAAL_EXPORT double rocAucScore(const NumericTablePtr & truePrediction, const NumericTablePtr & testPrediction)
{
    return rocAucScore<double>(truePrediction, testPrediction);
}

} // namespace internal
} // namespace data_management
} // namespace daal
