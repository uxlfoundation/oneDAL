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
    uniquePath[uniqueDepth].featureIndex  = featureIndex;
    uniquePath[uniqueDepth].zeroFraction  = zeroFraction;
    uniquePath[uniqueDepth].oneFraction   = oneFraction;
    uniquePath[uniqueDepth].partialWeight = (uniqueDepth == 0 ? 1.0f : 0.0f);

    const float constant = 1.0f / static_cast<float>(uniqueDepth + 1);
    for (int i = uniqueDepth - 1; i >= 0; --i)
    {
        uniquePath[i + 1].partialWeight += oneFraction * uniquePath[i].partialWeight * (i + 1) * constant;
        uniquePath[i].partialWeight = zeroFraction * uniquePath[i].partialWeight * (uniqueDepth - i) * constant;
    }
}

// undo a previous extension of the decision path
void unwindPath(PathElement * uniquePath, size_t uniqueDepth, size_t pathIndex)
{
    const float oneFraction  = uniquePath[pathIndex].oneFraction;
    const float zeroFraction = uniquePath[pathIndex].zeroFraction;
    float nextOnePortion     = uniquePath[uniqueDepth].partialWeight;

    if (oneFraction != 0)
    {
        for (int i = uniqueDepth - 1; i >= 0; --i)
        {
            const float tmp             = uniquePath[i].partialWeight;
            uniquePath[i].partialWeight = nextOnePortion * (uniqueDepth + 1) / static_cast<float>((i + 1) * oneFraction);
            nextOnePortion              = tmp - uniquePath[i].partialWeight * zeroFraction * (uniqueDepth - i) / static_cast<float>(uniqueDepth + 1);
        }
    }
    else
    {
        for (int i = 0; i < uniqueDepth; ++i)
        {
            uniquePath[i].partialWeight = (uniquePath[i].partialWeight * (uniqueDepth + 1)) / static_cast<float>(zeroFraction * (uniqueDepth - i));
        }
    }

    for (size_t i = pathIndex; i < uniqueDepth; ++i)
    {
        uniquePath[i].featureIndex = uniquePath[i + 1].featureIndex;
        uniquePath[i].zeroFraction = uniquePath[i + 1].zeroFraction;
        uniquePath[i].oneFraction  = uniquePath[i + 1].oneFraction;
    }
}

// determine what the total permutation weight would be if we unwound a previous extension in the decision path
float unwoundPathSum(const PathElement * uniquePath, size_t uniqueDepth, size_t pathIndex)
{
    const float oneFraction  = uniquePath[pathIndex].oneFraction;
    const float zeroFraction = uniquePath[pathIndex].zeroFraction;

    float nextOnePortion = uniquePath[uniqueDepth].partialWeight;
    float total          = 0;

    if (oneFraction != 0)
    {
        const float frac = zeroFraction / oneFraction;
        for (int i = uniqueDepth - 1; i >= 0; --i)
        {
            const float tmp = nextOnePortion / (i + 1);
            total += tmp;
            nextOnePortion = uniquePath[i].partialWeight - tmp * frac * (uniqueDepth - i);
        }
        total *= (uniqueDepth + 1) / oneFraction;
    }
    else if (zeroFraction != 0)
    {
        for (int i = 0; i < uniqueDepth; ++i)
        {
            total += uniquePath[i].partialWeight / (uniqueDepth - i);
        }
        total *= (uniqueDepth + 1) / zeroFraction;
    }
    else
    {
        for (int i = 0; i < uniqueDepth; ++i)
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
    uniquePath[uniqueDepth].featureIndex = featureIndex;
    uniquePath[uniqueDepth].zeroFraction = zeroFraction;
    uniquePath[uniqueDepth].oneFraction  = oneFraction;
    if (oneFraction != 0)
    {
        // extend partialWeights iff the feature of the last split satisfies the threshold
        partialWeights[uniqueDepthPartialWeights] = (uniqueDepthPartialWeights == 0 ? 1.0f : 0.0f);
        for (int i = uniqueDepthPartialWeights - 1; i >= 0; i--)
        {
            partialWeights[i + 1] += partialWeights[i] * (i + 1) / static_cast<float>(uniqueDepth + 1);
            partialWeights[i] *= zeroFraction * (uniqueDepth - i) / static_cast<float>(uniqueDepth + 1);
        }
    }
    else
    {
        for (int i = uniqueDepthPartialWeights - 1; i >= 0; i--)
        {
            partialWeights[i] *= (uniqueDepth - i) / static_cast<float>(uniqueDepth + 1);
        }
    }
}

void unwindPath(PathElement * uniquePath, float * partialWeights, uint32_t uniqueDepth, uint32_t uniqueDepthPartialWeights, uint32_t pathIndex)
{
    const float oneFraction  = uniquePath[pathIndex].oneFraction;
    const float zeroFraction = uniquePath[pathIndex].zeroFraction;
    float nextOnePortion     = partialWeights[uniqueDepthPartialWeights];

    if (oneFraction != 0)
    {
        // shrink partialWeights iff the feature satisfies the threshold
        for (uint32_t i = uniqueDepthPartialWeights - 1;; --i)
        {
            const float tmp   = partialWeights[i];
            partialWeights[i] = nextOnePortion * (uniqueDepth + 1) / static_cast<float>(i + 1);
            nextOnePortion    = tmp - partialWeights[i] * zeroFraction * (uniqueDepth - i) / static_cast<float>(uniqueDepth + 1);
            if (i == 0) break;
        }
    }
    else
    {
        for (uint32_t i = 0; i <= uniqueDepthPartialWeights; ++i)
        {
            partialWeights[i] *= (uniqueDepth + 1) / static_cast<float>(uniqueDepth - i);
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
    float total              = 0;
    const float zeroFraction = uniquePath[pathIndex].zeroFraction;
    float nextOnePortion     = partialWeights[uniqueDepthPartialWeights];
    for (int i = uniqueDepthPartialWeights - 1; i >= 0; --i)
    {
        const float tmp = nextOnePortion / static_cast<float>(i + 1);
        total += tmp;
        nextOnePortion = partialWeights[i] - tmp * zeroFraction * (uniqueDepth - i);
    }
    return total * (uniqueDepth + 1);
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
