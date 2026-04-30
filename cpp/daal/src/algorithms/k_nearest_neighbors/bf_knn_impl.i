/* file: bf_knn_impl.i */
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

#ifndef __BF_KNN_IMPL_I__
#define __BF_KNN_IMPL_I__

#include "algorithms/engines/engine.h"
#include "services/daal_defines.h"
#include "algorithms/classifier/classifier_model.h"
#include "algorithms/k_nearest_neighbors/bf_knn_classification_model.h"
#include "src/algorithms/k_nearest_neighbors/bf_knn_classification_train_kernel.h"
#include "src/algorithms/k_nearest_neighbors/bf_knn_classification_predict_kernel.h"
#include "src/algorithms/k_nearest_neighbors/bf_knn_classification_model_impl.h"
#include "data_management/data/numeric_table.h"
#include "services/env_detect.h"
#include "src/services/service_data_utils.h"
#include "src/services/service_utils.h"
#include "src/services/service_defines.h"
#include "src/threading/threading.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_sort.h"
#include "src/externals/service_math.h"
#include "src/algorithms/k_nearest_neighbors/knn_heap.h"

namespace daal
{
namespace algorithms
{
namespace bf_knn_classification
{
namespace internal
{
using namespace algorithms::internal;

template <typename FPType, CpuType cpu>
class BruteForceNearestNeighbors
{
public:
    BruteForceNearestNeighbors() {}

    ~BruteForceNearestNeighbors() {}

    typedef GlobalNeighbors<FPType, cpu> Neighbors;
    typedef Heap<Neighbors, cpu> HeapType;

    services::Status kNeighbors(const size_t k, const size_t nClasses, VoteWeights voteWeights, DAAL_UINT64 resultsToCompute,
                                DAAL_UINT64 resultsToEvaluate, const NumericTable * trainTable, const NumericTable * testTable,
                                const NumericTable * trainLabelTable, NumericTable * testLabelTable, NumericTable * indicesTable,
                                NumericTable * distancesTable, bf_knn_classification::prediction::internal::PairwiseDistanceType pairwiseDistance,
                                const double minkowskiDegree)
    {
        using bf_knn_classification::prediction::internal::PairwiseDistanceType;

        const size_t nDims  = trainTable->getNumberOfColumns();
        const size_t nTrain = trainTable->getNumberOfRows();
        const size_t nTest  = testTable->getNumberOfRows();

        FPType * trainLabel = nullptr;
        BlockDescriptor<FPType> trainLabelBlock;

        NumericTable * newTrainLabelTable = const_cast<NumericTable *>(trainLabelTable);
        if (resultsToEvaluate & daal::algorithms::classifier::computeClassLabels)
        {
            newTrainLabelTable->getBlockOfRows(0, nTrain, readOnly, trainLabelBlock);
            trainLabel = trainLabelBlock.getBlockPtr();
            DAAL_CHECK_MALLOC(trainLabel);
        }

        services::SharedPtr<PairwiseDistances<FPType, cpu> > dist;

        if (pairwiseDistance == PairwiseDistanceType::minkowski && minkowskiDegree == 2.0)
        {
            dist.reset(new EuclideanDistances<FPType, cpu>(*testTable, *trainTable, true));
        }
        else if (pairwiseDistance == PairwiseDistanceType::minkowski)
        {
            dist.reset(new MinkowskiDistances<FPType, cpu>(*testTable, *trainTable, true, minkowskiDegree));
        }
        else if (pairwiseDistance == PairwiseDistanceType::chebyshev)
        {
            dist.reset(new ChebyshevDistances<FPType, cpu>(*testTable, *trainTable));
        }
        else if (pairwiseDistance == PairwiseDistanceType::cosine)
        {
            dist.reset(new CosineDistances<FPType, cpu>(*testTable, *trainTable));
        }
        else
        {
            dist.reset(new EuclideanDistances<FPType, cpu>(*testTable, *trainTable, true));
        }

        dist->init();

        const size_t outBlockSize = 128;
        const size_t inBlockSize  = 128;
        const size_t nOuterBlocks = nTest / outBlockSize + !!(nTest % outBlockSize);

        TlsMem<FPType, cpu> tlsDistances(inBlockSize * outBlockSize);
        TlsMem<int, cpu> tlsIdx(outBlockSize);
        TlsMem<FPType, cpu> tlsKDistances(inBlockSize * k);
        TlsMem<int, cpu> tlsKIndexes(inBlockSize * k);
        TlsMem<FPType, cpu> tlsVoting(nClasses);

        SafeStatus safeStat;

        daal::threader_for(nOuterBlocks, 1, [&](size_t outerBlock) {
            const size_t outerStart = outerBlock * outBlockSize;
            const size_t outerEnd   = outerBlock + 1 == nOuterBlocks ? nTest : outerStart + outBlockSize;
            const size_t outerSize  = outerEnd - outerStart;

            DAAL_CHECK_STATUS_THR(computeKNearestBlock(dist.get(), outerSize, inBlockSize, outerStart, nTrain, resultsToEvaluate, resultsToCompute,
                                                       nClasses, k, voteWeights, trainLabel, trainTable, testTable, testLabelTable, indicesTable,
                                                       distancesTable, tlsDistances, tlsIdx, tlsKDistances, tlsKIndexes, tlsVoting, nOuterBlocks));
        });

        if (resultsToEvaluate & daal::algorithms::classifier::computeClassLabels)
        {
            newTrainLabelTable->releaseBlockOfRows(trainLabelBlock);
        }

        return safeStat.detach();
    }

protected:
    struct BruteForceTask
    {
    public:
        DAAL_NEW_DELETE();
        FPType * maxs;
        HeapType * heapsData;

        static BruteForceTask * create(const size_t inBlockSize, const size_t outBlockSize, const size_t k)
        {
            auto object = new BruteForceTask(inBlockSize, outBlockSize, k);
            if (object && object->isValid()) return object;
            delete object;
            return nullptr;
        }

        bool isValid() const { return _buff.get() && _heaps.get(); }

    private:
        BruteForceTask(size_t inBlockSize, size_t outBlockSize, size_t k)
        {
            _buff.reset(outBlockSize);
            maxs = _buff.get();
            service_memset_seq<FPType, cpu>(maxs, MaxVal<FPType>::get(), outBlockSize);

            _heaps.reset(outBlockSize);

            for (size_t i = 0; i < outBlockSize; ++i)
            {
                _heaps[i].init(k);
            }
            heapsData = _heaps.get();
        }

        TArrayScalable<FPType, cpu> _buff;
        TArrayScalable<HeapType, cpu> _heaps;
    };

    services::Status computeKNearestBlock(PairwiseDistances<FPType, cpu> * distancesInstance, const size_t blockSize, const size_t trainBlockSize,
                                          const size_t startTestIdx, const size_t nTrain, DAAL_UINT64 resultsToEvaluate, DAAL_UINT64 resultsToCompute,
                                          const size_t nClasses, const size_t k, VoteWeights voteWeights, FPType * trainLabel,
                                          const NumericTable * trainTable, const NumericTable * testTable, NumericTable * testLabelTable,
                                          NumericTable * indicesTable, NumericTable * distancesTable, TlsMem<FPType, cpu> & tlsDistances,
                                          TlsMem<int, cpu> & tlsIdx, TlsMem<FPType, cpu> & tlsKDistances, TlsMem<int, cpu> & tlsKIndexes,
                                          TlsMem<FPType, cpu> & tlsVoting, size_t nOuterBlocks)
    {
        const size_t inBlockSize = trainBlockSize;
        const size_t inRows      = nTrain;
        const size_t nInBlocks   = inRows / inBlockSize + (inRows % inBlockSize > 0);

        const size_t i1    = startTestIdx;
        const size_t i2    = startTestIdx + blockSize;
        const size_t iSize = blockSize;

        ReadRows<FPType, cpu> inDataRows(const_cast<NumericTable *>(testTable), i1, i2 - i1);
        DAAL_CHECK_BLOCK_STATUS(inDataRows);
        const FPType * const testData = inDataRows.get();

        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, blockSize, k);
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, inBlockSize, k);
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, inBlockSize * sizeof(int), k);
        DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, inBlockSize * sizeof(FPType), k);

        SafeStatus safeStat;

        daal::static_tls<BruteForceTask *> tlsTask([=, &safeStat]() {
            auto tlsData = BruteForceTask::create(inBlockSize, iSize, k);
            if (!tlsData)
            {
                safeStat.add(services::ErrorMemoryAllocationFailed);
            }
            return tlsData;
        });

        const size_t nThreads = _daal_threader_get_max_threads();
        daal::conditional_static_threader_for(nOuterBlocks < 2 * nThreads, nInBlocks, [&](size_t inBlock, size_t tid) {
            const size_t j1    = inBlock * inBlockSize;
            const size_t j2    = (inBlock + 1 == nInBlocks ? inRows : j1 + inBlockSize);
            const size_t jSize = j2 - j1;

            const BruteForceTask * tls = tlsTask.local(tid);
            DAAL_CHECK_MALLOC_THR(tls);

            FPType * distancesBuff = tlsDistances.local();
            DAAL_CHECK_MALLOC_THR(distancesBuff);

            int * idx = tlsIdx.local();
            DAAL_CHECK_MALLOC_THR(idx);

            FPType * maxs         = tls->maxs;
            HeapType * heapsLocal = tls->heapsData;

            ReadRows<FPType, cpu> outDataRows(const_cast<NumericTable *>(trainTable), j1, j2 - j1);
            DAAL_CHECK_BLOCK_STATUS_THR(outDataRows);
            const FPType * const trainData = outDataRows.get();

            DAAL_CHECK_STATUS_THR(distancesInstance->computeBatch(testData, trainData, i1, iSize, j1, jSize, distancesBuff));

            for (size_t i = 0; i < iSize; i++)
            {
                const size_t indexes = getIndexesWithLessDistances(idx, distancesBuff + i * jSize, jSize, maxs[i]);

                if (indexes)
                {
                    DAAL_ASSERT(inRows + j1 <= static_cast<size_t>(services::internal::MaxVal<int>::get()));
                    DAAL_ASSERT(inRows + i * jSize <= static_cast<size_t>(services::internal::MaxVal<int>::get()));
                    updateLocalNeighbours(indexes, idx, jSize, i, k, maxs, distancesBuff, j1, heapsLocal[i]);
                }
            }
        });

        int * kIndexes = tlsKIndexes.local();
        DAAL_CHECK_MALLOC(kIndexes);

        FPType * kDistances = tlsKDistances.local();
        DAAL_CHECK_MALLOC(kDistances);

        TArrayScalable<HeapType, cpu> heaps(iSize);

        for (size_t i = 0; i < iSize; ++i)
        {
            heaps[i].init(k);
        }

        tlsTask.reduce([&](BruteForceTask * tls) {
            HeapType * heapsLocal = tls->heapsData;
            for (size_t i = 0; i < iSize; i++)
            {
                const size_t size = heapsLocal[i].size();
                for (size_t j = 0; j < size; ++j)
                {
                    heaps[i].replaceMaxIfNeeded(heapsLocal[i][j], k);
                }
            }

            delete tls;
        });

        for (size_t i = 0; i < iSize; i++)
        {
            for (size_t kk = 0; kk < k; ++kk)
            {
                kDistances[i * k + kk] = heaps[i][kk].distance;
                kIndexes[i * k + kk]   = heaps[i][kk].index;
            }
        }
        distancesInstance->finalize(iSize * k, kDistances);

        // sort by distances
        for (size_t i = 0; i < iSize; ++i)
        {
            qSort<FPType, int, cpu>(k, kDistances + i * k, kIndexes + i * k);
        }

        if (resultsToCompute & computeIndicesOfNeighbors)
        {
            daal::internal::WriteOnlyRows<int, cpu> indexesBlock(indicesTable, startTestIdx, iSize);
            DAAL_CHECK_BLOCK_STATUS(indexesBlock);
            int * indices = indexesBlock.get();

            DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, blockSize * sizeof(*indices), k);
            const size_t size = blockSize * k * sizeof(*indices);
            DAAL_CHECK(!daal::services::internal::daal_memcpy_s(indices, size, kIndexes, size), daal::services::ErrorMemoryCopyFailedInternal);
        }

        if (resultsToCompute & computeDistances)
        {
            daal::internal::WriteOnlyRows<FPType, cpu> distancesBlock(distancesTable, startTestIdx, iSize);
            DAAL_CHECK_BLOCK_STATUS(distancesBlock);
            FPType * distances = distancesBlock.get();

            DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, blockSize * sizeof(FPType), k);
            const size_t size = blockSize * k * sizeof(FPType);
            DAAL_CHECK(!daal::services::internal::daal_memcpy_s(distances, size, kDistances, size), daal::services::ErrorMemoryCopyFailedInternal);
        }

        if (resultsToEvaluate & daal::algorithms::classifier::computeClassLabels)
        {
            daal::internal::WriteOnlyRows<int, cpu> testLabelRows(testLabelTable, startTestIdx, iSize);
            DAAL_CHECK_BLOCK_STATUS(testLabelRows);
            int * testLabel = testLabelRows.get();

            FPType * voting = tlsVoting.local();
            DAAL_CHECK_MALLOC(voting);

            if (voteWeights == VoteWeights::voteUniform)
            {
                DAAL_CHECK_STATUS_VAR(uniformWeightedVoting(nClasses, k, iSize, nTrain, kIndexes, trainLabel, testLabel, voting));
            }
            else
            {
                DAAL_CHECK_STATUS_VAR(distanceWeightedVoting(nClasses, k, iSize, nTrain, kDistances, kIndexes, trainLabel, testLabel, voting));
            }
        }

        return services::Status();
    }

    size_t getIndexesWithLessDistances(int * idx, FPType * array, size_t size, FPType cmp)
    {
        size_t count = 0;

        for (size_t i = 0; i < size; ++i)
        {
            if (array[i] < cmp)
            {
                idx[count++] = i;
            }
        }
        return count;
    }

    void updateLocalNeighbours(size_t indexes, int * idx, size_t jSize, size_t i, size_t k, FPType * maxs, FPType * distances, size_t j1,
                               HeapType & heap)
    {
        for (size_t j = 0; j < indexes; ++j)
        {
            FPType d = distances[i * jSize + idx[j]];

            Neighbors neigh;
            neigh.distance = d;
            neigh.index    = idx[j] + j1;

            heap.replaceMaxIfNeeded(neigh, k);
        }

        maxs[i] = heap.getMax()->distance;
    }

    services::Status uniformWeightedVoting(const size_t nClasses, const size_t k, const size_t n, const size_t nTrain, int * indices,
                                           const FPType * trainLabel, int * testLabel, FPType * classWeights)
    {
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < nClasses; ++j)
            {
                classWeights[j] = 0;
            }
            for (size_t j = 0; j < k; ++j)
            {
                const int label = static_cast<int>(trainLabel[indices[i * k + j]]);
                classWeights[label] += 1;
            }
            size_t maxWeightClass = 0;
            size_t maxWeight      = 0;
            for (size_t j = 0; j < nClasses; ++j)
            {
                if (classWeights[j] > maxWeight)
                {
                    maxWeight      = classWeights[j];
                    maxWeightClass = j;
                }
            }
            testLabel[i] = maxWeightClass;
        }
        return services::Status();
    }

    services::Status distanceWeightedVoting(const size_t nClasses, const size_t k, const size_t n, const size_t nTrain, FPType * distances,
                                            int * indices, const FPType * trainLabel, int * testLabel, FPType * classWeights)
    {
        const FPType epsilon = daal::services::internal::EpsilonVal<FPType>::get();

        for (size_t i = 0; i < n; ++i)
        {
            bool isContainZero = false;
            for (size_t j = 0; j < k * n; ++j)
            {
                if (distances[j] < epsilon)
                {
                    isContainZero = true;
                    break;
                }
            }
            for (size_t j = 0; j < nClasses; ++j)
            {
                classWeights[j] = 0;
            }
            if (isContainZero)
            {
                for (size_t j = 0; j < k; ++j)
                {
                    if (distances[j] < epsilon)
                    {
                        const int label = static_cast<int>(trainLabel[indices[i * k + j]]);
                        classWeights[label] += 1;
                    }
                }
            }
            else
            {
                for (size_t j = 0; j < k; ++j)
                {
                    const int label = static_cast<int>(trainLabel[indices[i * k + j]]);
                    classWeights[label] += 1 / distances[i * k + j];
                }
            }
            size_t maxWeightClass = 0;
            FPType maxWeight      = 0;
            for (size_t j = 0; j < nClasses; ++j)
            {
                if (classWeights[j] > maxWeight)
                {
                    maxWeight      = classWeights[j];
                    maxWeightClass = j;
                }
            }
            testLabel[i] = maxWeightClass;
        }
        return services::Status();
    }
};

} // namespace internal
} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal

#endif
