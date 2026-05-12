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
#include <utility>

#include "src/services/service_arrays.h"
#include "src/services/service_data_utils.h"
#include "src/services/service_defines.h"

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
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
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
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
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
// Bounded max-heap of the k nearest neighbors seen so far.
//
// Ordering invariant: dists_[0] is the largest distance currently in the heap
// (binary max-heap over dists_; indices_ moves in lockstep). While fewer than
// capacity neighbors have been pushed the heap keeps filling; once full,
// dists_[0] is the current k-th nearest distance and push() only replaces the
// top when a strictly closer neighbor arrives. maxDist() returns that top or
// +infinity while the heap is not yet full, which lets tree traversals use it
// as a pruning radius. The heap owns its storage via TArrayScalable so each
// thread can allocate its own without external TlsMem scaffolding.
// =========================================================================
template <typename FPType, daal::internal::CpuType cpu>
struct KnnHeap
{
    KnnHeap(int cap) : capacity_(cap), size_(0), distsArr_(cap), indicesArr_(cap)
    {
        dists_   = distsArr_.get();
        indices_ = indicesArr_.get();
    }

    KnnHeap(const KnnHeap &)             = delete;
    KnnHeap & operator=(const KnnHeap &) = delete;

    bool ok() const { return dists_ != nullptr && indices_ != nullptr; }

    FPType maxDist() const { return (size_ > 0) ? dists_[0] : daal::services::internal::MaxVal<FPType>::get(); }

    void push(FPType dist, int idx)
    {
        if (size_ < capacity_)
        {
            dists_[size_]   = dist;
            indices_[size_] = idx;
            size_++;
            int i = size_ - 1;
            while (i > 0)
            {
                int parent = (i - 1) / 2;
                if (dists_[i] > dists_[parent])
                {
                    std::swap(dists_[i], dists_[parent]);
                    std::swap(indices_[i], indices_[parent]);
                    i = parent;
                }
                else
                    break;
            }
        }
        else if (dist < dists_[0])
        {
            dists_[0]   = dist;
            indices_[0] = idx;
            int i       = 0;
            while (true)
            {
                int l       = 2 * i + 1;
                int r       = 2 * i + 2;
                int largest = i;
                if (l < size_ && dists_[l] > dists_[largest]) largest = l;
                if (r < size_ && dists_[r] > dists_[largest]) largest = r;
                if (largest != i)
                {
                    std::swap(dists_[i], dists_[largest]);
                    std::swap(indices_[i], indices_[largest]);
                    i = largest;
                }
                else
                    break;
            }
        }
    }

private:
    int capacity_;
    int size_;
    daal::services::internal::TArrayScalable<FPType, cpu> distsArr_;
    daal::services::internal::TArrayScalable<int, cpu> indicesArr_;
    FPType * dists_; // distances from points in the heap to the current query (heap root is the largest)
    int * indices_;  // indices of points in the heap, kept in lockstep with dists_
};

} // namespace internal
} // namespace hdbscan
} // namespace algorithms
} // namespace daal
