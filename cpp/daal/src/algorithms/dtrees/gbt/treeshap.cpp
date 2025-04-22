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

#include "src/algorithms/dtrees/gbt/treeshap.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace treeshap
{

namespace internal
{

namespace v0
{

// extend our decision path with a fraction of one and zero extensions
void extendPath(PathElement * uniquePath, size_t uniqueDepth, float zeroFraction, float oneFraction, int featureIndex)
{
    if (uniqueDepth >= INT_MAX)
    {
        // this is virtually impossible because memory consumption increases exponentially and we'd
        // run of memory way before we hit this limit
        throw services::ErrorIncorrectIndex;
    }
    int depth = static_cast<int>(uniqueDepth);

    uniquePath[depth].featureIndex  = featureIndex;
    uniquePath[depth].zeroFraction  = zeroFraction;
    uniquePath[depth].oneFraction   = oneFraction;
    uniquePath[depth].partialWeight = (depth == 0 ? 1.0f : 0.0f);

    const float constant = 1.0f / static_cast<float>(depth + 1);
    for (int i = depth - 1; i >= 0; --i)
    {
        uniquePath[i + 1].partialWeight += oneFraction * uniquePath[i].partialWeight * (i + 1) * constant;
        uniquePath[i].partialWeight = zeroFraction * uniquePath[i].partialWeight * (depth - i) * constant;
    }
}

// undo a previous extension of the decision path
void unwindPath(PathElement * uniquePath, size_t uniqueDepth, size_t pathIndex)
{
    if (uniqueDepth >= INT_MAX)
    {
        // this is virtually impossible because memory consumption increases exponentially and we'd
        // run of memory way before we hit this limit
        throw services::ErrorIncorrectIndex;
    }
    int depth = static_cast<int>(uniqueDepth);

    const float oneFraction  = uniquePath[pathIndex].oneFraction;
    const float zeroFraction = uniquePath[pathIndex].zeroFraction;
    float nextOnePortion     = uniquePath[depth].partialWeight;

    if (oneFraction != 0)
    {
        for (int i = depth - 1; i >= 0; --i)
        {
            const float tmp             = uniquePath[i].partialWeight;
            uniquePath[i].partialWeight = nextOnePortion * (depth + 1) / static_cast<float>((i + 1) * oneFraction);
            nextOnePortion              = tmp - uniquePath[i].partialWeight * zeroFraction * (depth - i) / static_cast<float>(depth + 1);
        }
    }
    else
    {
        for (int i = 0; i < depth; ++i)
        {
            uniquePath[i].partialWeight = (uniquePath[i].partialWeight * (depth + 1)) / static_cast<float>(zeroFraction * (depth - i));
        }
    }

    for (size_t i = pathIndex; i < depth; ++i)
    {
        uniquePath[i].featureIndex = uniquePath[i + 1].featureIndex;
        uniquePath[i].zeroFraction = uniquePath[i + 1].zeroFraction;
        uniquePath[i].oneFraction  = uniquePath[i + 1].oneFraction;
    }
}

// determine what the total permutation weight would be if we unwound a previous extension in the decision path
float unwoundPathSum(const PathElement * uniquePath, size_t uniqueDepth, size_t pathIndex)
{
    if (uniqueDepth >= INT_MAX)
    {
        // this is virtually impossible because memory consumption increases exponentially and we'd
        // run of memory way before we hit this limit
        throw services::ErrorIncorrectIndex;
    }
    int depth = static_cast<int>(uniqueDepth);

    const float oneFraction  = uniquePath[pathIndex].oneFraction;
    const float zeroFraction = uniquePath[pathIndex].zeroFraction;

    float nextOnePortion = uniquePath[depth].partialWeight;
    float total          = 0;

    if (oneFraction != 0)
    {
        const float frac = zeroFraction / oneFraction;
        for (int i = depth - 1; i >= 0; --i)
        {
            const float tmp = nextOnePortion / (i + 1);
            total += tmp;
            nextOnePortion = uniquePath[i].partialWeight - tmp * frac * (depth - i);
        }
        total *= (depth + 1) / oneFraction;
    }
    else if (zeroFraction != 0)
    {
        for (int i = 0; i < depth; ++i)
        {
            total += uniquePath[i].partialWeight / (depth - i);
        }
        total *= (depth + 1) / zeroFraction;
    }
    else
    {
        for (int i = 0; i < depth; ++i)
        {
            DAAL_ASSERT(uniquePath[i].partialWeight == 0);
        }
    }

    return total;
}

} // namespace v0

namespace v1
{
void extendPath(PathElement * uniquePath, float * partialWeights, uint32_t uniqueDepth, uint32_t uniqueDepthPartialWeights, float zeroFraction,
                float oneFraction, int featureIndex)
{
    if (uniqueDepth >= INT_MAX || uniqueDepthPartialWeights >= INT_MAX)
    {
        // this is virtually impossible because memory consumption increases exponentially and we'd
        // run of memory way before we hit this limit
        throw services::ErrorIncorrectIndex;
    }
    int depth               = static_cast<int>(uniqueDepth);
    int depthPartialWeights = static_cast<int>(uniqueDepthPartialWeights);

    uniquePath[depth].featureIndex = featureIndex;
    uniquePath[depth].zeroFraction = zeroFraction;
    uniquePath[depth].oneFraction  = oneFraction;
    if (oneFraction != 0)
    {
        // extend partialWeights iff the feature of the last split satisfies the threshold
        partialWeights[uniqueDepthPartialWeights] = (uniqueDepthPartialWeights == 0 ? 1.0f : 0.0f);
        for (int i = depthPartialWeights - 1; i >= 0; i--)
        {
            partialWeights[i + 1] += partialWeights[i] * (i + 1) / static_cast<float>(depth + 1);
            partialWeights[i] *= zeroFraction * (depth - i) / static_cast<float>(depth + 1);
        }
    }
    else
    {
        for (int i = depthPartialWeights - 1; i >= 0; i--)
        {
            partialWeights[i] *= (depth - i) / static_cast<float>(depth + 1);
        }
    }
}

void unwindPath(PathElement * uniquePath, float * partialWeights, uint32_t uniqueDepth, uint32_t uniqueDepthPartialWeights, uint32_t pathIndex)
{
    if (uniqueDepth >= INT_MAX || uniqueDepthPartialWeights >= INT_MAX)
    {
        // this is virtually impossible because memory consumption increases exponentially and we'd
        // run of memory way before we hit this limit
        throw services::ErrorIncorrectIndex;
    }
    int depth               = static_cast<int>(uniqueDepth);
    int depthPartialWeights = static_cast<int>(uniqueDepthPartialWeights);

    const float oneFraction  = uniquePath[pathIndex].oneFraction;
    const float zeroFraction = uniquePath[pathIndex].zeroFraction;
    float nextOnePortion     = partialWeights[uniqueDepthPartialWeights];

    if (oneFraction != 0)
    {
        // shrink partialWeights iff the feature satisfies the threshold
        for (uint32_t i = depthPartialWeights - 1;; --i)
        {
            const float tmp   = partialWeights[i];
            partialWeights[i] = nextOnePortion * (depth + 1) / static_cast<float>(i + 1);
            nextOnePortion    = tmp - partialWeights[i] * zeroFraction * (depth - i) / static_cast<float>(depth + 1);
            if (i == 0) break;
        }
    }
    else
    {
        for (uint32_t i = 0; i <= uniqueDepthPartialWeights; ++i)
        {
            partialWeights[i] *= (depth + 1) / static_cast<float>(depth - i);
        }
    }

    for (uint32_t i = pathIndex; i < uniqueDepth; ++i)
    {
        uniquePath[i].featureIndex = uniquePath[i + 1].featureIndex;
        uniquePath[i].zeroFraction = uniquePath[i + 1].zeroFraction;
        uniquePath[i].oneFraction  = uniquePath[i + 1].oneFraction;
    }
}

// determine what the total permuation weight would be if
// we unwound a previous extension in the decision path (for feature satisfying the threshold)
float unwoundPathSum(const PathElement * uniquePath, const float * partialWeights, uint32_t uniqueDepth, uint32_t uniqueDepthPartialWeights,
                     uint32_t pathIndex)
{
    if (uniqueDepth >= INT_MAX || uniqueDepthPartialWeights >= INT_MAX)
    {
        // this is virtually impossible because memory consumption increases exponentially and we'd
        // run of memory way before we hit this limit
        throw services::ErrorIncorrectIndex;
    }
    int depth               = static_cast<int>(uniqueDepth);
    int depthPartialWeights = static_cast<int>(uniqueDepthPartialWeights);

    float total              = 0;
    const float zeroFraction = uniquePath[pathIndex].zeroFraction;
    float nextOnePortion     = partialWeights[uniqueDepthPartialWeights];
    for (int i = depthPartialWeights - 1; i >= 0; --i)
    {
        const float tmp = nextOnePortion / static_cast<float>(i + 1);
        total += tmp;
        nextOnePortion = partialWeights[i] - tmp * zeroFraction * (depth - i);
    }
    return total * (depth + 1);
}

float unwoundPathSumZero(const float * partialWeights, uint32_t uniqueDepth, uint32_t uniqueDepthPartialWeights)
{
    float total = 0;
    if (uniqueDepth > uniqueDepthPartialWeights)
    {
        for (uint32_t i = 0; i <= uniqueDepthPartialWeights; ++i)
        {
            total += partialWeights[i] / static_cast<float>(uniqueDepth - i);
        }
    }
    return total * (uniqueDepth + 1);
}
} // namespace v1

} // namespace internal
} // namespace treeshap
} // namespace gbt
} // namespace algorithms
} // namespace daal
