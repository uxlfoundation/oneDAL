/* file: kdtree_knn_classification_train_kernel.h */
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
//  Declaration of structure containing kernels for K-Nearest Neighbors training.
//--
*/

#ifndef __KDTREE_KNN_CLASSIFICATION_TRAIN_KERNEL_H__
#define __KDTREE_KNN_CLASSIFICATION_TRAIN_KERNEL_H__

#include "data_management/data/numeric_table.h"
#include "algorithms/algorithm_base_common.h"
#include "algorithms/k_nearest_neighbors/kdtree_knn_classification_training_types.h"
#include "src/algorithms/service_error_handling.h"

#if defined(_MSC_VER)
    #define DAAL_FORCEINLINE   __forceinline
    #define DAAL_FORCENOINLINE __declspec(noinline)
#else
    #define DAAL_FORCEINLINE   inline __attribute__((always_inline))
    #define DAAL_FORCENOINLINE __attribute__((noinline))
#endif

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace training
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services;

template <typename algorithmFpType, training::Method method, CpuType cpu>
class KNNClassificationTrainBatchKernel
{};

template <typename T, CpuType cpu>
class Queue;
struct BuildNode;
template <typename T>
struct BoundingBox;
template <typename algorithmFpType, CpuType cpu>
struct IndexValuePair;

template <typename algorithmFpType, CpuType cpu>
class KNNClassificationTrainBatchKernel<algorithmFpType, training::defaultDense, cpu> : public daal::algorithms::Kernel
{
public:
    services::Status compute(NumericTable * x, NumericTable * y, kdtree_knn_classification::Model * r, engines::BatchBase & engine);

protected:
    Status buildFirstPartOfKDTree(Queue<BuildNode, cpu> & q, BoundingBox<algorithmFpType> *& bboxQ, const NumericTable & x,
                                  kdtree_knn_classification::Model & r, size_t * indexes, engines::BatchBase & engine);

    Status computeLocalBoundingBoxOfKDTree(BoundingBox<algorithmFpType> * localBBox, const NumericTable & x, const size_t * indexes);

    size_t selectDimensionSophisticated(size_t start, size_t end, size_t * sampleIndexes, algorithmFpType * sampleValues, size_t sampleCount,
                                        const NumericTable & x, const size_t * indexes, engines::BatchBase * engine);

    algorithmFpType computeApproximatedMedianInParallel(size_t start, size_t end, size_t dimension, algorithmFpType upper, const NumericTable & x,
                                                        const size_t * indexes, engines::BatchBase & engine, algorithmFpType * subSamples,
                                                        size_t subSampleCapacity, Status & status);

    DAAL_FORCEINLINE size_t computeBucketID(algorithmFpType * samples, size_t sampleCount, algorithmFpType * subSamples, size_t subSampleCount,
                                            size_t subSampleCount16, algorithmFpType value);

    size_t adjustIndexesInParallel(size_t start, size_t end, size_t dimension, algorithmFpType median, const NumericTable & x, size_t * indexes,
                                   services::Status & status);

    void copyBBox(BoundingBox<algorithmFpType> * dest, const BoundingBox<algorithmFpType> * src, size_t n);

    Status rearrangePoints(NumericTable & x, const size_t * indexes);

    Status buildSecondPartOfKDTree(Queue<BuildNode, cpu> & q, BoundingBox<algorithmFpType> *& bboxQ, const NumericTable & x,
                                   kdtree_knn_classification::Model & r, size_t * indexes, engines::BatchBase & engine);

    algorithmFpType computeApproximatedMedianInSerial(size_t start, size_t end, size_t dimension, algorithmFpType upper,
                                                      IndexValuePair<algorithmFpType, cpu> * inSortValues,
                                                      IndexValuePair<algorithmFpType, cpu> * outSortValues, size_t sortValueCount,
                                                      const NumericTable & x, size_t * indexes, engines::BatchBase * engine,
                                                      services::Status & status);

    size_t adjustIndexesInSerial(size_t start, size_t end, size_t dimension, algorithmFpType median, const NumericTable & x, size_t * indexes);

    DAAL_FORCEINLINE void radixSort(IndexValuePair<algorithmFpType, cpu> * inValues, size_t valueCount,
                                    IndexValuePair<algorithmFpType, cpu> * outValues);
};

} // namespace internal
} // namespace training
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

#endif
