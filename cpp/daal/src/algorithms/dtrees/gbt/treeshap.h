/* file: treeshap.h */
/*******************************************************************************
* Copyright 2023 Intel Corporation
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

/**
 * Original TreeSHAP algorithm by Scott Lundberg, 2018
 * https://arxiv.org/abs/1802.03888
 * Originally contributed to XGBoost in
 *   - https://github.com/dmlc/xgboost/pull/2438
 *   - https://github.com/dmlc/xgboost/pull/3043
 * XGBoost is licensed under Apache-2 (https://github.com/dmlc/xgboost/blob/master/LICENSE)
 *
 * Fast TreeSHAP algorithm v1 and v2 by Jilei Yang, 2021
 * https://arxiv.org/abs/2109.09847
 * C code available at https://github.com/linkedin/FastTreeSHAP/blob/master/fasttreeshap/cext/_cext.cc
 * Fast TreeSHAP is licensed under BSD-2 (https://github.com/linkedin/FastTreeSHAP/blob/master/LICENSE)
 */

/*
//++
//  Implementation of the treeShap algorithm
//--
*/

#ifndef __TREESHAP_H__
#define __TREESHAP_H__

#include "services/daal_defines.h"
#include "services/error_handling.h"
#include "src/algorithms/dtrees/dtrees_feature_type_helper.h"
#include "src/algorithms/dtrees/gbt/gbt_model_impl.h"
#include "src/services/service_arrays.h"
#include "src/algorithms/dtrees/gbt/gbt_predict_dense_default_impl.i"
#include <cfloat> // FLT_EPSILON

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace treeshap
{
using gbt::internal::FeatureIndexType;
using gbt::internal::ModelFPType;
using FeatureTypes = algorithms::dtrees::internal::FeatureTypes;

/**
 * Decision Path context
*/
struct PathElement
{
    int featureIndex                 = 0;
    float zeroFraction               = 0;
    float oneFraction                = 0;
    float partialWeight              = 0;
    PathElement()                    = default;
    PathElement(const PathElement &) = default;
};

namespace internal
{

namespace v0
{

void extendPath(PathElement * uniquePath, size_t uniqueDepth, float zeroFraction, float oneFraction, int featureIndex);
void unwindPath(PathElement * uniquePath, size_t uniqueDepth, size_t pathIndex);
float unwoundPathSum(const PathElement * uniquePath, size_t uniqueDepth, size_t pathIndex);

/** Recursive treeShap function
 * \param nodeIndex the index of the current node in the tree, counted from 1
 * \param depth how deep are we in the tree
 * \param uniqueDepth how many unique features are above the current node in the tree
 * \param parentUniquePath a vector of statistics about our current path through the tree
 * \param parentZeroFraction what fraction of the parent path weight is coming as 0 (integrated)
 * \param parentOneFraction what fraction of the parent path weight is coming as 1 (fixed)
 * \param parentFeatureIndex what feature the parent node used to split
 * \param conditionFraction what fraction of the current weight matches our conditioning feature
 */
template <typename algorithmFPType, bool hasUnorderedFeatures, bool hasAnyMissing>
inline void treeShap(const gbt::internal::GbtDecisionTree * tree, const algorithmFPType * x, algorithmFPType * phi,
                     const FeatureTypes * featureHelper, size_t nodeIndex, size_t depth, size_t uniqueDepth, PathElement * parentUniquePath,
                     float parentZeroFraction, float parentOneFraction, int parentFeatureIndex, int condition, FeatureIndexType conditionFeature,
                     float conditionFraction)
{
    DAAL_ASSERT(parentUniquePath);

    // stop if we have no weight coming down to us
    if (conditionFraction < FLT_EPSILON) return;

    const ModelFPType * const splitValues     = tree->getSplitPoints() - 1;
    const size_t * const leftIds              = tree->getLeftChildIndexes() - 1;
    const FeatureIndexType * const fIndexes   = tree->getFeatureIndexesForSplit() - 1;
    const ModelFPType * const nodeCoverValues = tree->getNodeCoverValues() - 1;
    const int * const defaultLeft             = tree->getDefaultLeftForSplit() - 1;

    PathElement * uniquePath = parentUniquePath + uniqueDepth + 1;
    const size_t nBytes      = (uniqueDepth + 1) * sizeof(PathElement);
    const int copyStatus     = daal::services::internal::daal_memcpy_s(uniquePath, nBytes, parentUniquePath, nBytes);
    DAAL_ASSERT(copyStatus == 0);

    if (condition == 0 || conditionFeature != static_cast<FeatureIndexType>(parentFeatureIndex))
    {
        extendPath(uniquePath, uniqueDepth, parentZeroFraction, parentOneFraction, parentFeatureIndex);
    }

    const bool isLeaf = gbt::internal::ModelImpl::nodeIsLeaf(nodeIndex, *tree, depth);

    // leaf node
    if (isLeaf)
    {
        for (size_t i = 1; i <= uniqueDepth; ++i)
        {
            const float w          = unwoundPathSum(uniquePath, uniqueDepth, i);
            const PathElement & el = uniquePath[i];
            phi[el.featureIndex] += w * (el.oneFraction - el.zeroFraction) * splitValues[nodeIndex] * conditionFraction;
        }

        return;
    }

    const FeatureIndexType splitIndex = fIndexes[nodeIndex];
    const algorithmFPType dataValue   = x[splitIndex];

    gbt::prediction::internal::PredictDispatcher<hasUnorderedFeatures, hasAnyMissing> dispatcher;

    // TODO: fix indexing
    size_t hotIndex        = updateIndex(nodeIndex, dataValue, splitValues, leftIds, defaultLeft, *featureHelper, splitIndex, dispatcher);
    const size_t coldIndex = 2 * nodeIndex + (hotIndex == (2 * nodeIndex));

    const float w = nodeCoverValues[nodeIndex];
    DAAL_ASSERT(w > 0);
    const float hotZeroFraction  = nodeCoverValues[hotIndex] / w;
    const float coldZeroFraction = nodeCoverValues[coldIndex] / w;
    float incomingZeroFraction   = 1.0f;
    float incomingOneFraction    = 1.0f;

    DAAL_ASSERT(hotZeroFraction < 1.0f);
    DAAL_ASSERT(coldZeroFraction < 1.0f);

    // see if we have already split on this feature,
    // if so we undo that split so we can redo it for this node
    size_t previousSplitPathIndex = 0ul;
    for (; previousSplitPathIndex <= uniqueDepth; ++previousSplitPathIndex)
    {
        const FeatureIndexType castIndex = static_cast<FeatureIndexType>(uniquePath[previousSplitPathIndex].featureIndex);

        // It cannot be that a feature that is ignored is in the uniquePath
        DAAL_ASSERT((condition == 0) || (castIndex != conditionFeature));

        if (castIndex == splitIndex)
        {
            break;
        }
    }
    if (previousSplitPathIndex != uniqueDepth + 1)
    {
        incomingZeroFraction = uniquePath[previousSplitPathIndex].zeroFraction;
        incomingOneFraction  = uniquePath[previousSplitPathIndex].oneFraction;
        unwindPath(uniquePath, uniqueDepth, previousSplitPathIndex);
        uniqueDepth -= 1;
    }

    // divide up the conditionFraction among the recursive calls
    float hotConditionFraction  = conditionFraction;
    float coldConditionFraction = conditionFraction;
    if (condition > 0 && splitIndex == conditionFeature)
    {
        coldConditionFraction = 0;
        uniqueDepth -= 1;
    }
    else if (condition < 0 && splitIndex == conditionFeature)
    {
        hotConditionFraction *= hotZeroFraction;
        coldConditionFraction *= coldZeroFraction;
        uniqueDepth -= 1;
    }

    treeShap<algorithmFPType, hasUnorderedFeatures, hasAnyMissing>(tree, x, phi, featureHelper, hotIndex, depth + 1, uniqueDepth + 1, uniquePath,
                                                                   hotZeroFraction * incomingZeroFraction, incomingOneFraction, splitIndex, condition,
                                                                   conditionFeature, hotConditionFraction);
    treeShap<algorithmFPType, hasUnorderedFeatures, hasAnyMissing>(tree, x, phi, featureHelper, coldIndex, depth + 1, uniqueDepth + 1, uniquePath,
                                                                   coldZeroFraction * incomingZeroFraction, 0, splitIndex, condition,
                                                                   conditionFeature, coldConditionFraction);
}

/**
 * \brief Version 0, i.e. the original TreeSHAP algorithm to compute feature attributions for a single tree
 * \param tree current tree
 * \param x dense data matrix
 * \param phi dense output matrix of feature attributions
 * \param featureHelper pointer to a FeatureTypes object (required to traverse tree)
 * \param condition fix one feature to either off (-1) on (1) or not fixed (0 default)
 * \param conditionFeature the index of the feature to fix
 */
template <typename algorithmFPType, CpuType cpu, bool hasUnorderedFeatures, bool hasAnyMissing>
inline services::Status treeShap(const gbt::internal::GbtDecisionTree * tree, const algorithmFPType * x, algorithmFPType * phi,
                                 const FeatureTypes * featureHelper, int condition, FeatureIndexType conditionFeature)
{
    services::Status st;
    const int depth          = tree->getMaxLvl() + 2;
    const size_t nUniquePath = ((depth * (depth + 1)) / 2);

    TArray<PathElement, cpu> uniquePathData(nUniquePath);
    DAAL_CHECK_MALLOC(uniquePathData.get());

    treeShap<algorithmFPType, hasUnorderedFeatures, hasAnyMissing>(tree, x, phi, featureHelper, 1, 0, 0, uniquePathData.get(), 1, 1, -1, condition,
                                                                   conditionFeature, 1);

    return st;
}

} // namespace v0

namespace v1
{

void extendPath(PathElement * uniquePath, float * pWeights, uint32_t uniqueDepth, uint32_t uniqueDepthPWeights, float zeroFraction, float oneFraction,
                int featureIndex);
void unwindPath(PathElement * uniquePath, float * pWeights, uint32_t uniqueDepth, uint32_t uniqueDepthPWeights, uint32_t pathIndex);
float unwoundPathSum(const PathElement * uniquePath, const float * pWeights, uint32_t uniqueDepth, uint32_t uniqueDepthPWeights, uint32_t pathIndex);
float unwoundPathSumZero(const float * pWeights, uint32_t uniqueDepth, uint32_t uniqueDepthPWeights);

/**
 * Recursive Fast TreeSHAP version 1
 * Important: nodeIndex is counted from 0 here!
*/
template <typename algorithmFPType, bool hasUnorderedFeatures, bool hasAnyMissing>
inline void treeShap(const gbt::internal::GbtDecisionTree * tree, const algorithmFPType * x, algorithmFPType * phi, size_t numOutputs,
                     const FeatureTypes * featureHelper, size_t nodeIndex, size_t depth, size_t uniqueDepth, size_t uniqueDepthPWeights,
                     PathElement * parentUniquePath, float * parentPWeights, algorithmFPType pWeightsResidual, float parentZeroFraction,
                     float parentOneFraction, int parentFeatureIndex, int condition, FeatureIndexType conditionFeature, float conditionFraction)
{
    // stop if we have no weight coming down to us
    if (conditionFraction < FLT_EPSILON) return;

    const ModelFPType * const splitValues     = tree->getSplitPoints() - 1;
    const size_t * const leftIds              = tree->getLeftChildIndexes() - 1;
    const int * const defaultLeft             = tree->getDefaultLeftForSplit() - 1;
    const FeatureIndexType * const fIndexes   = tree->getFeatureIndexesForSplit() - 1;
    const ModelFPType * const nodeCoverValues = tree->getNodeCoverValues() - 1;

    // extend the unique path
    PathElement * uniquePath = parentUniquePath + uniqueDepth + 1;
    size_t nBytes            = (uniqueDepth + 1) * sizeof(PathElement);
    int copyStatus           = daal::services::internal::daal_memcpy_s(uniquePath, nBytes, parentUniquePath, nBytes);
    DAAL_ASSERT(copyStatus == 0);
    // extend pWeights
    float * pWeights = parentPWeights + uniqueDepthPWeights + 1;
    nBytes           = (uniqueDepthPWeights + 1) * sizeof(float);
    copyStatus       = daal::services::internal::daal_memcpy_s(pWeights, nBytes, parentPWeights, nBytes);
    DAAL_ASSERT(copyStatus == 0);

    if (condition == 0 || conditionFeature != static_cast<uint32_t>(parentFeatureIndex))
    {
        extendPath(uniquePath, pWeights, uniqueDepth, uniqueDepthPWeights, parentZeroFraction, parentOneFraction, parentFeatureIndex);
        // update pWeightsResidual if the feature of the last split does not satisfy the threshold
        if (parentOneFraction != 1)
        {
            pWeightsResidual *= parentZeroFraction;
            uniqueDepthPWeights -= 1;
        }
    }

    const bool isLeaf = gbt::internal::ModelImpl::nodeIsLeaf(nodeIndex, *tree, depth);

    if (isLeaf)
    {
        const size_t valuesOffset   = nodeIndex * numOutputs;
        uint32_t valuesNonZeroInd   = 0;
        uint32_t valuesNonZeroCount = 0;
        for (uint32_t j = 0; j < numOutputs; ++j)
        {
            if (splitValues[valuesOffset + j] != 0)
            {
                valuesNonZeroInd = j;
                valuesNonZeroCount++;
            }
        }
        // pre-calculate wZero for all features not satisfying the thresholds
        const algorithmFPType wZero     = unwoundPathSumZero(pWeights, uniqueDepth, uniqueDepthPWeights);
        const algorithmFPType scaleZero = -wZero * pWeightsResidual * conditionFraction;
        algorithmFPType scale;
        for (uint32_t i = 1; i <= uniqueDepth; ++i)
        {
            const PathElement & el   = uniquePath[i];
            const uint32_t phiOffset = el.featureIndex * numOutputs;
            // update contributions to SHAP values for features satisfying the thresholds and not satisfying the thresholds separately
            if (el.oneFraction != 0)
            {
                const algorithmFPType w = unwoundPathSum(uniquePath, pWeights, uniqueDepth, uniqueDepthPWeights, i);
                scale                   = w * pWeightsResidual * (1 - el.zeroFraction) * conditionFraction;
            }
            else
            {
                scale = scaleZero;
            }
            if (valuesNonZeroCount == 1)
            {
                phi[phiOffset + valuesNonZeroInd] += scale * splitValues[valuesOffset + valuesNonZeroInd];
            }
            else
            {
                for (uint32_t j = 0; j < numOutputs; ++j)
                {
                    phi[phiOffset + j] += scale * splitValues[valuesOffset + j];
                }
            }
        }

        return;
    }

    const FeatureIndexType splitIndex = fIndexes[nodeIndex];
    const algorithmFPType dataValue   = x[splitIndex];

    gbt::prediction::internal::PredictDispatcher<hasUnorderedFeatures, hasAnyMissing> dispatcher;
    // TODO: fix indexing
    size_t hotIndex        = updateIndex(nodeIndex, dataValue, splitValues, leftIds, defaultLeft, *featureHelper, splitIndex, dispatcher);
    // const size_t coldIndex = 2 * nodeIndex + (hotIndex == (2 * nodeIndex));
    const size_t coldIndex = leftIds[nodeIndex] + (hotIndex == leftIds[nodeIndex]);

    const algorithmFPType w                = nodeCoverValues[nodeIndex];
    const algorithmFPType hotZeroFraction  = nodeCoverValues[hotIndex] / w;
    const algorithmFPType coldZeroFraction = nodeCoverValues[coldIndex] / w;
    algorithmFPType incomingZeroFraction   = 1;
    algorithmFPType incomingOneFraction    = 1;

    // see if we have already split on this feature,
    // if so we undo that split so we can redo it for this node
    uint32_t pathIndex = 0;
    for (; pathIndex <= uniqueDepth; ++pathIndex)
    {
        if (uniquePath[pathIndex].featureIndex == splitIndex) break;
    }
    if (pathIndex != uniqueDepth + 1)
    {
        incomingZeroFraction = uniquePath[pathIndex].zeroFraction;
        incomingOneFraction  = uniquePath[pathIndex].oneFraction;
        unwindPath(uniquePath, pWeights, uniqueDepth, uniqueDepthPWeights, pathIndex);
        --uniqueDepth;
        // update pWeightsResidual iff the duplicated feature does not satisfy the threshold
        if (incomingOneFraction != 0.)
        {
            uniqueDepthPWeights -= 1;
        }
        else
        {
            pWeightsResidual /= incomingZeroFraction;
        }
    }

    // divide up the conditionFraction among the recursive calls
    algorithmFPType hotConditionFraction  = conditionFraction;
    algorithmFPType coldConditionFraction = conditionFraction;
    if (condition > 0 && splitIndex == conditionFeature)
    {
        coldConditionFraction = 0;
        --uniqueDepth;
        --uniqueDepthPWeights;
    }
    else if (condition < 0 && splitIndex == conditionFeature)
    {
        hotConditionFraction *= hotZeroFraction;
        coldConditionFraction *= coldZeroFraction;
        --uniqueDepth;
        --uniqueDepthPWeights;
    }

    treeShap<algorithmFPType, hasUnorderedFeatures, hasAnyMissing>(
        tree, x, phi, numOutputs, featureHelper, hotIndex, depth + 1, uniqueDepth + 1, uniqueDepthPWeights + 1, uniquePath, pWeights,
        pWeightsResidual, hotZeroFraction * incomingZeroFraction, incomingOneFraction, splitIndex, condition, conditionFeature, hotConditionFraction);

    treeShap<algorithmFPType, hasUnorderedFeatures, hasAnyMissing>(
        tree, x, phi, numOutputs, featureHelper, coldIndex, depth + 1, uniqueDepth + 1, uniqueDepthPWeights + 1, uniquePath, pWeights,
        pWeightsResidual, coldZeroFraction * incomingZeroFraction, 0, splitIndex, condition, conditionFeature, coldConditionFraction);
}

/**
 * \brief Version 1, i.e. first Fast TreeSHAP algorithm
 * \param tree current tree
 * \param x dense data matrix
 * \param phi dense output matrix of feature attributions
 * \param featureHelper pointer to a FeatureTypes object (required to traverse tree)
 * \param condition fix one feature to either off (-1) on (1) or not fixed (0 default)
 * \param conditionFeature the index of the feature to fix
 */
template <typename algorithmFPType, CpuType cpu, bool hasUnorderedFeatures, bool hasAnyMissing>
inline services::Status treeShap(const gbt::internal::GbtDecisionTree * tree, const algorithmFPType * x, algorithmFPType * phi, size_t numClasses,
                                 const FeatureTypes * featureHelper, int condition, FeatureIndexType conditionFeature)
{
    services::Status st;

    // pre-allocate space for the unique path data and pWeights
    const int depth        = tree->getMaxLvl() + 2;
    const size_t nElements = (depth * (depth + 1)) / 2;

    TArray<PathElement, cpu> uniquePathData(nElements);
    DAAL_CHECK_MALLOC(uniquePathData.get());

    TArray<float, cpu> pWeights(nElements);
    DAAL_CHECK_MALLOC(pWeights.get());

    treeShap<algorithmFPType, hasUnorderedFeatures, hasAnyMissing>(tree, x, phi, numClasses, featureHelper, 1, 0, 0, 0, uniquePathData.get(),
                                                                   pWeights.get(), 1, 1, 1, -1, condition, conditionFeature, 1);

    return st;
}
} // namespace v1

} // namespace internal

enum TreeShapVersion
{
    lundberg = 0, /** https://arxiv.org/abs/1802.03888 */
    fast_v1,      /** https://arxiv.org/abs/2109.09847 */
};

/**
 * \brief Recursive function that computes the feature attributions for a single tree.
 * \param tree current tree
 * \param x dense data matrix
 * \param phi dense output matrix of feature attributions
 * \param numClasses number of classes in input data
 * \param featureHelper pointer to a FeatureTypes object (required to traverse tree)
 * \param condition fix one feature to either off (-1) on (1) or not fixed (0 default)
 * \param conditionFeature the index of the feature to fix
 */
template <typename algorithmFPType, CpuType cpu, bool hasUnorderedFeatures, bool hasAnyMissing>
inline services::Status treeShap(const gbt::internal::GbtDecisionTree * tree, const algorithmFPType * x, algorithmFPType * phi, size_t numClasses,
                                 const FeatureTypes * featureHelper, int condition, FeatureIndexType conditionFeature,
                                 TreeShapVersion shapVersion = fast_v1)
{
    DAAL_ASSERT(x);
    DAAL_ASSERT(phi);
    DAAL_ASSERT(featureHelper);

    switch (shapVersion)
    {
    case lundberg:
        DAAL_CHECK(numClasses == 1, services::ErrorIncorrectParameter); // our lundberg version only supports single-class/regression
        return treeshap::internal::v0::treeShap<algorithmFPType, cpu, hasUnorderedFeatures, hasAnyMissing>(tree, x, phi, featureHelper, condition,
                                                                                                           conditionFeature);
    case fast_v1:
        return treeshap::internal::v1::treeShap<algorithmFPType, cpu, hasUnorderedFeatures, hasAnyMissing>(tree, x, phi, numClasses, featureHelper,
                                                                                                           condition, conditionFeature);
    default: return services::Status(ErrorMethodNotImplemented);
    }
}

} // namespace treeshap
} // namespace gbt
} // namespace algorithms
} // namespace daal

#endif // __TREESHAP_H__
