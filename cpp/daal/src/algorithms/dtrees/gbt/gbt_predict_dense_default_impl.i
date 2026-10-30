/* file: gbt_predict_dense_default_impl.i */
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
//  Implementation of auxiliary functions for gradient boosted trees prediction
//  (defaultDense) method.
//--
*/

#ifndef __GBT_PREDICT_DENSE_DEFAULT_IMPL_I__
#define __GBT_PREDICT_DENSE_DEFAULT_IMPL_I__

#include "data_management/data/internal/finiteness_checker.h"
#include "src/algorithms/dtrees/dtrees_feature_type_helper.h"
#include "src/algorithms/dtrees/dtrees_model_impl.h"
#include "src/algorithms/dtrees/dtrees_predict_dense_default_impl.i"
#include "src/algorithms/dtrees/dtrees_train_data_helper.i"
#include "src/algorithms/dtrees/gbt/gbt_internal.h"
#include "src/algorithms/dtrees/gbt/gbt_model_impl.h"
#include "src/services/service_arrays.h"
#include "src/services/service_defines.h"
#include "src/services/service_environment.h"
#include "src/services/service_utils.h"
#include "src/threading/threading.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace prediction
{
namespace internal
{
using gbt::internal::ModelFPType;
using gbt::internal::FeatureIndexType;

template <bool hasUnorderedFeatures, bool hasAnyMissing>
struct PredictDispatcher
{
    typedef PredictDispatcher<hasUnorderedFeatures, hasAnyMissing> type;
};

// In updateIndexDense implementations leftChildIndexes array is not used as we suppose that left child index is idx * 2
// updateIndex must be used instead for cases where this condition is not satisfied, otherwise results will be incorrect

// These functions are implemented in a way to minimize branching and improve performance, the output is the index of the correct child
// Here index of left child is idx * 2, index of right child is idx * 2 + 1, therefore we can calculate the index of child in the following way
// idx * 2 + (condition when we go to right child is satisfied)

template <typename algorithmFPType>
inline FeatureIndexType updateIndexDense(FeatureIndexType idx, algorithmFPType valueFromDataSet, const ModelFPType * splitPoints,
                                         const int * defaultLeft, const FeatureTypes & featTypes, FeatureIndexType splitFeature,
                                         const PredictDispatcher<false, false> & dispatcher)
{
    // Only continious features, no missing values
    return idx * 2 + (valueFromDataSet > splitPoints[idx]);
}

template <typename algorithmFPType>
inline FeatureIndexType updateIndexDense(FeatureIndexType idx, algorithmFPType valueFromDataSet, const ModelFPType * splitPoints,
                                         const int * defaultLeft, const FeatureTypes & featTypes, FeatureIndexType splitFeature,
                                         const PredictDispatcher<true, false> & dispatcher)
{
    // Continious and categorical features, no missing values
    return idx * 2 + (featTypes.isUnordered(splitFeature) ? valueFromDataSet != splitPoints[idx] : valueFromDataSet > splitPoints[idx]);
}

template <typename algorithmFPType>
inline FeatureIndexType updateIndexDense(FeatureIndexType idx, algorithmFPType valueFromDataSet, const ModelFPType * splitPoints,
                                         const int * defaultLeft, const FeatureTypes & featTypes, FeatureIndexType splitFeature,
                                         const PredictDispatcher<false, true> & dispatcher)
{
    // Only continious features, missing values might be present
    if (checkFinitenessByComparison(valueFromDataSet))
    {
        // If defaultLeft == 1 and value is missing we go to left child, if it is 0 we go to right child
        return idx * 2 + (defaultLeft[idx] != 1);
    }
    else
    {
        return idx * 2 + (valueFromDataSet > splitPoints[idx]);
    }
}

template <typename algorithmFPType>
inline FeatureIndexType updateIndexDense(FeatureIndexType idx, algorithmFPType valueFromDataSet, const ModelFPType * splitPoints,
                                         const int * defaultLeft, const FeatureTypes & featTypes, FeatureIndexType splitFeature,
                                         const PredictDispatcher<true, true> & dispatcher)
{
    // Continious and categorical features, missing values might be present
    if (checkFinitenessByComparison(valueFromDataSet))
    {
        // If defaultLeft == 1 and value is missing we go to left child, if it is 0 we go to right child
        return idx * 2 + (defaultLeft[idx] != 1);
    }
    else
    {
        return idx * 2 + (featTypes.isUnordered(splitFeature) ? valueFromDataSet != splitPoints[idx] : valueFromDataSet > splitPoints[idx]);
    }
}

// In updateIndex implementations leftChildIndexes array is used to determine the position of node's children.
// They can be used for all levels, however if tree is a full binary (on first numDenseLayers)
// it's recommended to use updateIndexDense function for faster inference

// These functions are implemented in a way to minimize branching and improve performance, the output is the index of the correct child
// Here index of left child is leftChildIndexes[idx], index of right child is leftChildIndexes[idx] + 1, therefore we can calculate the index of child in the following way
// leftChildIndexes[idx] + ((leftChildIndexes[idx] != idx) && condition when we go to right child is satisfied)
// Note that if leftChildIndexes[idx] == idx it means that we are in leaf node and we should perform transition to the same node
// In order to achieve that we need the condition to always be zero, therefore we add leftChildIndexes[idx] != idx to this condition:

template <typename algorithmFPType>
inline FeatureIndexType updateIndex(FeatureIndexType idx, algorithmFPType valueFromDataSet, const ModelFPType * splitPoints,
                                    const FeatureIndexType * leftChildIndexes, const int * defaultLeft, const FeatureTypes & featTypes,
                                    FeatureIndexType splitFeature, const PredictDispatcher<false, false> & dispatcher)
{
    // Only continious features, no missing values
    return leftChildIndexes[idx] + ((idx != leftChildIndexes[idx]) && (valueFromDataSet > splitPoints[idx]));
}

template <typename algorithmFPType>
inline FeatureIndexType updateIndex(FeatureIndexType idx, algorithmFPType valueFromDataSet, const ModelFPType * splitPoints,
                                    const FeatureIndexType * leftChildIndexes, const int * defaultLeft, const FeatureTypes & featTypes,
                                    FeatureIndexType splitFeature, const PredictDispatcher<true, false> & dispatcher)
{
    // Continious and categorical features, no missing values
    return leftChildIndexes[idx]
           + ((idx != leftChildIndexes[idx])
              && (featTypes.isUnordered(splitFeature) ? valueFromDataSet != splitPoints[idx] : valueFromDataSet > splitPoints[idx]));
}

template <typename algorithmFPType>
inline FeatureIndexType updateIndex(FeatureIndexType idx, algorithmFPType valueFromDataSet, const ModelFPType * splitPoints,
                                    const FeatureIndexType * leftChildIndexes, const int * defaultLeft, const FeatureTypes & featTypes,
                                    FeatureIndexType splitFeature, const PredictDispatcher<false, true> & dispatcher)
{
    // Only continious features, missing values might be present
    if (checkFinitenessByComparison(valueFromDataSet))
    {
        // If defaultLeft == 1 and value is missing we go to left child, if it is 0 we go to right child
        return leftChildIndexes[idx] + ((idx != leftChildIndexes[idx]) && (defaultLeft[idx] != 1));
    }
    else
    {
        return leftChildIndexes[idx] + ((idx != leftChildIndexes[idx]) && (valueFromDataSet > splitPoints[idx]));
    }
}

template <typename algorithmFPType>
inline FeatureIndexType updateIndex(FeatureIndexType idx, algorithmFPType valueFromDataSet, const ModelFPType * splitPoints,
                                    const FeatureIndexType * leftChildIndexes, const int * defaultLeft, const FeatureTypes & featTypes,
                                    FeatureIndexType splitFeature, const PredictDispatcher<true, true> & dispatcher)
{
    // Continious and categorical features, missing values might be present
    if (checkFinitenessByComparison(valueFromDataSet))
    {
        // If defaultLeft == 1 and value is missing we go to left child, if it is 0 we go to right child
        return leftChildIndexes[idx] + ((idx != leftChildIndexes[idx]) && (defaultLeft[idx] != 1));
    }
    else
    {
        return leftChildIndexes[idx]
               + ((idx != leftChildIndexes[idx])
                  && (featTypes.isUnordered(splitFeature) ? valueFromDataSet != splitPoints[idx] : valueFromDataSet > splitPoints[idx]));
    }
}

template <typename algorithmFPType, typename DecisionTreeType, CpuType cpu, bool hasUnorderedFeatures, bool hasAnyMissing, size_t vectorBlockSize>
inline void predictForTreeVector(const DecisionTreeType & t, const FeatureTypes & featTypes, const algorithmFPType * x, algorithmFPType v[],
                                 const PredictDispatcher<hasUnorderedFeatures, hasAnyMissing> & dispatcher)
{
    const ModelFPType * const values        = t.getSplitPoints() - 1;
    const FeatureIndexType * const leftIds  = t.getLeftChildIndexes() - 1;
    const FeatureIndexType * const fIndexes = t.getFeatureIndexesForSplit() - 1;
    const int * const defaultLeft           = t.getDefaultLeftForSplit() - 1;
    const size_t nNodes                     = t.getNumberOfNodes();
    const FeatureIndexType nFeat            = featTypes.getNumberOfFeatures();
    FeatureIndexType i[vectorBlockSize];
    services::internal::service_memset_seq<FeatureIndexType, cpu>(i, FeatureIndexType(1), vectorBlockSize);
    const FeatureIndexType maxLvl         = t.getMaxLvl();
    const FeatureIndexType numDenseLayers = t.getNumDenseLayers();
    for (FeatureIndexType itr = 0; itr < std::min(maxLvl, numDenseLayers); itr++)
    {
        for (FeatureIndexType k = 0; k < vectorBlockSize; k++)
        {
            const FeatureIndexType idx          = i[k];
            const FeatureIndexType splitFeature = fIndexes[idx];
            i[k] = updateIndexDense<algorithmFPType>(idx, x[splitFeature + k * nFeat], values, defaultLeft, featTypes, splitFeature, dispatcher);
            // After itr, node depth is itr + 1
            // Node indexes on level k are in range [2^k, 2^(k + 1) - 1]
            DAAL_ASSERT(i[k] >= (static_cast<size_t>(1) << (itr + 1)))
            DAAL_ASSERT(i[k] < (static_cast<size_t>(1) << (itr + 2)))
        }
    }
    // TODO: Implement performance optimization for deep but sparse trees
    for (FeatureIndexType itr = numDenseLayers; itr < maxLvl; itr++)
    {
        for (FeatureIndexType k = 0; k < vectorBlockSize; k++)
        {
            const FeatureIndexType idx          = i[k];
            const FeatureIndexType splitFeature = fIndexes[idx];
            i[k] = updateIndex<algorithmFPType>(idx, x[splitFeature + k * nFeat], values, leftIds, defaultLeft, featTypes, splitFeature, dispatcher);
            DAAL_ASSERT(i[k] <= nNodes)
        }
    }

    PRAGMA_OMP_SIMD
    PRAGMA_VECTOR_ALWAYS
    for (FeatureIndexType k = 0; k < vectorBlockSize; k++)
    {
        v[k] = values[i[k]];
    }
}

// Traverses nTrees <= maxTrees trees for a single row, interleaving the trees to hide memory latency.
// Leaves below the dense layers point to themselves, so extra updateIndex steps for shallower trees are no-ops.
template <typename algorithmFPType, typename DecisionTreeType, CpuType cpu, bool hasUnorderedFeatures, bool hasAnyMissing, size_t maxTrees>
inline void predictForTreesVector(const DecisionTreeType * const * trees, size_t nTrees, const FeatureTypes & featTypes, const algorithmFPType * x,
                                  algorithmFPType v[], const PredictDispatcher<hasUnorderedFeatures, hasAnyMissing> & dispatcher)
{
    DAAL_ASSERT(nTrees <= maxTrees)
    const ModelFPType * values[maxTrees];
    const FeatureIndexType * leftIds[maxTrees];
    const FeatureIndexType * fIndexes[maxTrees];
    const int * defaultLeft[maxTrees];
    FeatureIndexType i[maxTrees];
    FeatureIndexType nDenseLvl = trees[0]->getMaxLvl();
    FeatureIndexType maxLvl    = 0;
    for (size_t k = 0; k < nTrees; ++k)
    {
        const DecisionTreeType & t = *trees[k];
        values[k]                  = t.getSplitPoints() - 1;
        leftIds[k]                 = t.getLeftChildIndexes() - 1;
        fIndexes[k]                = t.getFeatureIndexesForSplit() - 1;
        defaultLeft[k]             = t.getDefaultLeftForSplit() - 1;
        i[k]                       = 1;
        const FeatureIndexType lvl = t.getMaxLvl();
        nDenseLvl =
            services::internal::min<cpu, FeatureIndexType>(nDenseLvl, services::internal::min<cpu, FeatureIndexType>(t.getNumDenseLayers(), lvl));
        maxLvl = services::internal::max<cpu, FeatureIndexType>(maxLvl, lvl);
    }
    for (FeatureIndexType itr = 0; itr < nDenseLvl; itr++)
    {
        PRAGMA_OMP_SIMD
        for (size_t k = 0; k < nTrees; ++k)
        {
            const FeatureIndexType splitFeature = fIndexes[k][i[k]];
            i[k] = updateIndexDense<algorithmFPType>(i[k], x[splitFeature], values[k], defaultLeft[k], featTypes, splitFeature, dispatcher);
        }
    }
    for (FeatureIndexType itr = nDenseLvl; itr < maxLvl; itr++)
    {
        PRAGMA_OMP_SIMD
        for (size_t k = 0; k < nTrees; ++k)
        {
            const FeatureIndexType splitFeature = fIndexes[k][i[k]];
            i[k] = updateIndex<algorithmFPType>(i[k], x[splitFeature], values[k], leftIds[k], defaultLeft[k], featTypes, splitFeature, dispatcher);
            DAAL_ASSERT(i[k] <= trees[k]->getNumberOfNodes())
        }
    }
    for (size_t k = 0; k < nTrees; ++k) v[k] = values[k][i[k]];
}

// Tree-parallel prediction: each of nBlocks blocks collects its trees of the model into trees[iFirstTree, iFirstTree + nTrees),
// predictBlock(iFirstTree, nTrees, out) accumulates them into zeroed out, blocks are reduced into res
template <typename algorithmFPType, CpuType cpu, typename F>
services::Status predictByTreeBlocks(const gbt::internal::ModelImpl & model, const gbt::internal::GbtDecisionTree ** trees, size_t nTrees,
                                     size_t nBlocks, size_t nVals, algorithmFPType bias, algorithmFPType * res, const F & predictBlock)
{
    daal::services::internal::TArrayCalloc<algorithmFPType, cpu> partial(nBlocks * nVals);
    DAAL_CHECK_MALLOC(partial.get());
    const auto blockBody = [&](size_t iBlock, size_t) {
        const size_t iFirstTree = iBlock * nTrees / nBlocks;
        const size_t iLastTree  = (iBlock + 1) * nTrees / nBlocks;
        for (size_t iTree = iFirstTree; iTree < iLastTree; ++iTree) trees[iTree] = model.at(iTree);
        predictBlock(iFirstTree, iLastTree - iFirstTree, partial.get() + iBlock * nVals);
    };
    if (nBlocks > 1)
        daal::static_threader_for(nBlocks, blockBody);
    else
        blockBody(0, 0);
    for (size_t i = 0; i < nVals; ++i)
    {
        algorithmFPType sum = bias;
        for (size_t iBlock = 0; iBlock < nBlocks; ++iBlock) sum += partial[iBlock * nVals + i];
        res[i] = sum;
    }
    return services::Status();
}

template <typename algorithmFPType>
struct TileDimensions
{
    size_t nRowsTotal    = 0;
    size_t nTreesTotal   = 0;
    size_t nCols         = 0;
    size_t nRowsInBlock  = 0;
    size_t nTreesInBlock = 0;
    size_t nLargeBlocks  = 0;
    size_t nDataBlocks   = 0;
    size_t nTreeBlocks   = 0;

    // vectorBlockSize = vectorBlockSizeFactor * vectorBlockSizeStep
    size_t vectorBlockSizeFactor                     = 0;
    static constexpr size_t maxVectorBlockSizeFactor = 16;
    static constexpr size_t minVectorBlockSizeFactor = 2;
    static constexpr size_t vectorBlockSizeStep      = 16;
    // optimalBlockSizeFactor is selected from benchmarking
    static constexpr size_t optimalBlockSizeFactor = 5;
    // number of trees traversed simultaneously for a single row
    static constexpr size_t treeVectorBlockSize = 32;
    // minimal number of trees per thread in tree-parallel prediction, smaller blocks don't amortize threading overhead
    static constexpr size_t minTreesInBlock = 128;

    // Number of blocks in tree-parallel prediction of nRows, 0 if row-parallel prediction is preferable.
    // Each thread visits only its slice of trees, so tree blocks are faster even with 2x fewer blocks than minimal row blocks
    static size_t getNumberOfTreeParallelBlocks(size_t nTrees, size_t nRows)
    {
        const size_t nThreads = daal::threader_get_threads_number();
        size_t nBlocks        = nTrees / minTreesInBlock;
        if (nBlocks > nThreads) nBlocks = nThreads;
        if (nBlocks < 1) nBlocks = 1;
        return nRows < 2 * minVectorBlockSizeFactor * vectorBlockSizeStep * nBlocks ? nBlocks : 0;
    }

    TileDimensions(const NumericTable & data, size_t nTrees, const gbt::internal::GbtDecisionTree * const * trees)
        : nTreesTotal(nTrees), nRowsTotal(data.getNumberOfRows()), nCols(data.getNumberOfColumns())
    {
        // Decrease vectorBlockSize if number of rows is too small
        const size_t twoBlocksPerThreadFactor = nRowsTotal / (2 * daal::threader_get_threads_number() * vectorBlockSizeStep);
        size_t flexibleBlockSizeFactor        = maxVectorBlockSizeFactor;
        // Use smaller vectorBlockSize if trees fit to L2, the trees are visited only when it affects the result
        if (twoBlocksPerThreadFactor > optimalBlockSizeFactor)
        {
            size_t nNodes = 0;
            for (size_t iTree = 0; iTree < nTrees; ++iTree) nNodes += trees[iTree]->getNumberOfNodes();
            // Each node contain 3 values
            const size_t nodesSize = (sizeof(ModelFPType) + sizeof(FeatureIndexType) + sizeof(int)) * nNodes;
            if (nodesSize < daal::services::internal::getL2CacheSize()) flexibleBlockSizeFactor = optimalBlockSizeFactor;
        }
        if (flexibleBlockSizeFactor > twoBlocksPerThreadFactor) flexibleBlockSizeFactor = twoBlocksPerThreadFactor;
        if (flexibleBlockSizeFactor < minVectorBlockSizeFactor) flexibleBlockSizeFactor = minVectorBlockSizeFactor;

        DAAL_SAFE_CPU_CALL(vectorBlockSizeFactor = flexibleBlockSizeFactor, vectorBlockSizeFactor = optimalBlockSizeFactor)
        size_t vectorBlockSize = vectorBlockSizeStep * vectorBlockSizeFactor;
        nRowsInBlock           = nRowsTotal > vectorBlockSize ? vectorBlockSize : nRowsTotal;
        nDataBlocks            = nRowsTotal / nRowsInBlock + bool(nRowsTotal % nRowsInBlock);

        nTreesInBlock = nTreesTotal;
        nTreeBlocks   = 1;
    }
};

} /* namespace internal */
} /* namespace prediction */
} /* namespace gbt */
} /* namespace algorithms */
} /* namespace daal */

#endif
