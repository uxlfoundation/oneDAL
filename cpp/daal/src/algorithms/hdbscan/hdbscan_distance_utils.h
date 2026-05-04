/* file: hdbscan_distance_utils.h */
/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#pragma once

#include <cmath>
#include <algorithm>
#include <limits>

namespace daal
{
namespace algorithms
{
namespace hdbscan
{
namespace internal
{

// =========================================================================
// Distance functors for parameterizing tree queries by metric
// Each provides:
//   pointDist(a, b, nCols)       — full point-to-point distance
//   bboxLowerBound(q, lo, hi, nCols) — minimum distance from query to bbox
//   planeDist(diff)              — distance to splitting hyperplane
// =========================================================================
template <typename FPType>
struct EuclideanDist
{
    static FPType pointDist(const FPType * a, const FPType * b, int nCols)
    {
        FPType sum = FPType(0);
        for (int d = 0; d < nCols; d++)
        {
            const FPType diff = a[d] - b[d];
            sum += diff * diff;
        }
        return static_cast<FPType>(sqrt(static_cast<double>(sum)));
    }
    static FPType bboxLowerBound(const FPType * query, const FPType * lo, const FPType * hi, int nCols)
    {
        FPType sum = FPType(0);
        for (int d = 0; d < nCols; d++)
        {
            FPType excess = FPType(0);
            if (query[d] < lo[d])
                excess = lo[d] - query[d];
            else if (query[d] > hi[d])
                excess = query[d] - hi[d];
            sum += excess * excess;
        }
        return static_cast<FPType>(sqrt(static_cast<double>(sum)));
    }
    static FPType planeDist(FPType diff) { return (diff < FPType(0)) ? -diff : diff; }
};

template <typename FPType>
struct ManhattanDist
{
    static FPType pointDist(const FPType * a, const FPType * b, int nCols)
    {
        FPType sum = FPType(0);
        for (int d = 0; d < nCols; d++)
        {
            FPType diff = a[d] - b[d];
            sum += (diff >= FPType(0)) ? diff : -diff;
        }
        return sum;
    }
    static FPType bboxLowerBound(const FPType * query, const FPType * lo, const FPType * hi, int nCols)
    {
        FPType sum = FPType(0);
        for (int d = 0; d < nCols; d++)
        {
            FPType excess = FPType(0);
            if (query[d] < lo[d])
                excess = lo[d] - query[d];
            else if (query[d] > hi[d])
                excess = query[d] - hi[d];
            sum += excess;
        }
        return sum;
    }
    static FPType planeDist(FPType diff) { return (diff < FPType(0)) ? -diff : diff; }
};

template <typename FPType>
struct MinkowskiDist
{
    double p;
    double invp;
    MinkowskiDist(double degree) : p(degree), invp(1.0 / degree) {}

    FPType pointDist(const FPType * a, const FPType * b, int nCols) const
    {
        double sum = 0.0;
        for (int d = 0; d < nCols; d++)
        {
            double diff = static_cast<double>(a[d] - b[d]);
            if (diff < 0.0) diff = -diff;
            sum += pow(diff, p);
        }
        return static_cast<FPType>(pow(sum, invp));
    }
    FPType bboxLowerBound(const FPType * query, const FPType * lo, const FPType * hi, int nCols) const
    {
        double sum = 0.0;
        for (int d = 0; d < nCols; d++)
        {
            double excess = 0.0;
            if (query[d] < lo[d])
                excess = static_cast<double>(lo[d] - query[d]);
            else if (query[d] > hi[d])
                excess = static_cast<double>(query[d] - hi[d]);
            sum += pow(excess, p);
        }
        return static_cast<FPType>(pow(sum, invp));
    }
    FPType planeDist(FPType diff) const { return (diff < FPType(0)) ? -diff : diff; }
};

template <typename FPType>
struct ChebyshevDist
{
    static FPType pointDist(const FPType * a, const FPType * b, int nCols)
    {
        FPType mx = FPType(0);
        for (int d = 0; d < nCols; d++)
        {
            FPType diff = a[d] - b[d];
            if (diff < FPType(0)) diff = -diff;
            if (diff > mx) mx = diff;
        }
        return mx;
    }
    static FPType bboxLowerBound(const FPType * query, const FPType * lo, const FPType * hi, int nCols)
    {
        FPType mx = FPType(0);
        for (int d = 0; d < nCols; d++)
        {
            FPType excess = FPType(0);
            if (query[d] < lo[d])
                excess = lo[d] - query[d];
            else if (query[d] > hi[d])
                excess = query[d] - hi[d];
            if (excess > mx) mx = excess;
        }
        return mx;
    }
    static FPType planeDist(FPType diff) { return (diff < FPType(0)) ? -diff : diff; }
};

// =========================================================================
// Alpha-scaling wrapper: multiplies all distances by 1/alpha
// =========================================================================
template <typename FPType, typename BaseDist>
struct AlphaScaledDist
{
    BaseDist base;
    FPType invAlpha;

    AlphaScaledDist(const BaseDist & b, double alpha) : base(b), invAlpha(static_cast<FPType>(1.0 / alpha)) {}

    FPType pointDist(const FPType * a, const FPType * b, int nCols) const { return base.pointDist(a, b, nCols) * invAlpha; }
    FPType bboxLowerBound(const FPType * query, const FPType * lo, const FPType * hi, int nCols) const
    {
        return base.bboxLowerBound(query, lo, hi, nCols) * invAlpha;
    }
    FPType planeDist(FPType diff) const { return base.planeDist(diff) * invAlpha; }
};

// =========================================================================
// Max-heap for k nearest neighbors
// =========================================================================
template <typename FPType>
struct KnnHeap
{
    FPType * dists;
    int * indices;
    int capacity;
    int size;

    void init(FPType * d, int * idx, int cap)
    {
        dists    = d;
        indices  = idx;
        capacity = cap;
        size     = 0;
    }

    FPType maxDist() const { return (size > 0) ? dists[0] : std::numeric_limits<FPType>::max(); }

    void push(FPType dist, int idx)
    {
        if (size < capacity)
        {
            dists[size]   = dist;
            indices[size] = idx;
            size++;
            int i = size - 1;
            while (i > 0)
            {
                int parent = (i - 1) / 2;
                if (dists[i] > dists[parent])
                {
                    std::swap(dists[i], dists[parent]);
                    std::swap(indices[i], indices[parent]);
                    i = parent;
                }
                else
                    break;
            }
        }
        else if (dist < dists[0])
        {
            dists[0]   = dist;
            indices[0] = idx;
            int i      = 0;
            while (true)
            {
                int l       = 2 * i + 1;
                int r       = 2 * i + 2;
                int largest = i;
                if (l < size && dists[l] > dists[largest]) largest = l;
                if (r < size && dists[r] > dists[largest]) largest = r;
                if (largest != i)
                {
                    std::swap(dists[i], dists[largest]);
                    std::swap(indices[i], indices[largest]);
                    i = largest;
                }
                else
                    break;
            }
        }
    }
};

} // namespace internal
} // namespace hdbscan
} // namespace algorithms
} // namespace daal
