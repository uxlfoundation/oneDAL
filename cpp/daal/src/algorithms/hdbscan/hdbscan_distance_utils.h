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

#include "src/externals/service_blas.h"
#include "src/externals/service_math.h"
#include "src/services/service_arrays.h"
#include "src/services/service_data_utils.h"
#include "src/services/service_defines.h"
#include "src/threading/threading.h"

namespace daal
{
namespace algorithms
{
namespace hdbscan
{
namespace internal
{

/// Compute the squared L2 norm of a row, vectorized.
/// Used by EuclideanDist::blockDist to cache ‖x_i‖² between sweeps.
template <typename FPType>
static FPType rowNormSquared(const FPType * row, int nCols)
{
    FPType sum = FPType(0);
    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (int d = 0; d < nCols; d++) sum += row[d] * row[d];
    return sum;
}

/// Compute squared L2 norms for `count` contiguous row-major rows.
/// Caller pre-allocates outNorms of length `count`.
/// Used to amortize ‖x‖² across the three pivot sweeps inside one ball-tree node.
template <typename FPType>
static void rowNormsSquared(const FPType * rows, int count, int nCols, FPType * outNorms)
{
    for (int i = 0; i < count; i++) outNorms[i] = rowNormSquared(rows + i * nCols, nCols);
}

/// Fill a symmetric `nRows × nRows` distance matrix in row-major layout using a
/// scalar metric functor. Exploits symmetry: only the upper triangle (j >= i) is
/// computed, the lower triangle is mirrored. Outer parallelization over row
/// blocks via `daal::threader_for`. Diagonal entries are zeroed.
///
/// Used by the dense brute-force HDBSCAN for non-Euclidean metrics where there
/// is no GEMM identity to exploit. Centralises the row-pair loop so Manhattan,
/// Chebyshev, and Minkowski share one implementation instead of three near
/// duplicates.
template <typename FPType, daal::internal::CpuType cpu, typename DistFunc>
static void fillFullDistMatrix(const FPType * data, size_t nRows, size_t nCols, const DistFunc & distFunc, FPType * outDist)
{
    constexpr size_t blockSize = 64;
    const size_t nBlocks       = (nRows + blockSize - 1) / blockSize;

    daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock) {
        const size_t i_begin = iBlock * blockSize;
        const size_t i_end   = (i_begin + blockSize > nRows) ? nRows : i_begin + blockSize;
        for (size_t i = i_begin; i < i_end; i++)
        {
            const FPType * row_i = data + i * nCols;
            FPType * dist_row    = outDist + i * nRows;
            for (size_t j = i; j < nRows; j++)
            {
                const FPType d         = distFunc.pointDist(row_i, data + j * nCols, static_cast<int>(nCols));
                dist_row[j]            = d;
                outDist[j * nRows + i] = d;
            }
            dist_row[i] = FPType(0);
        }
    });
}

// =========================================================================
// Distance functors for parameterizing tree queries by metric.
// Each provides:
//   pointDist(a, b, nCols)              — full point-to-point distance
//   bboxLowerBound(q, lo, hi, nCols)    — minimum distance from query to bbox
//   planeDist(diff)                     — distance to splitting hyperplane
//   blockDist(pivotPt, scratchRows, rowNorms2, count, nCols, outDists, scratch)
//                                       — distance from one pivot to `count`
//                                         contiguous rows; Euclidean uses BLAS
//                                         xxgemv + cached row norms², others use
//                                         a vectorized inner loop. `rowNorms2`
//                                         and `scratch` may be nullptr for
//                                         non-Euclidean metrics.
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

    /// Vectorized pivot-to-block Euclidean distance via BLAS xxgemv.
    /// Identity used: ‖x_i − p‖² = ‖x_i‖² + ‖p‖² − 2·⟨x_i, p⟩.
    /// `scratchRows` is row-major `count × nCols`. `rowNorms2[i] = ‖scratchRows[i]‖²`
    /// must be precomputed by the caller (see rowNormsSquared). `outDists` (length
    /// `count`) is filled with non-negative Euclidean distances. Negative squared
    /// distances from rounding noise are clamped to zero before sqrt.
    template <daal::internal::CpuType cpu>
    static void blockDist(const FPType * pivotPt, const FPType * scratchRows, const FPType * rowNorms2, int count, int nCols, FPType * outDists)
    {
        const FPType pivotNorm2 = rowNormSquared(pivotPt, nCols);

        // outDists ← scratchRows · pivotPt  (count vector)
        // GEMV: y = α·op(A)·x + β·y. With trans='N', op(A)=A. We want
        // y[i] = ⟨scratchRows[i], pivotPt⟩ over nCols, so A is nCols×count
        // (column-major view of the row-major scratchRows[count×nCols]).
        const char trans    = 'N';
        const DAAL_INT m    = static_cast<DAAL_INT>(nCols);
        const DAAL_INT n    = static_cast<DAAL_INT>(count);
        const FPType alpha  = FPType(1);
        const FPType beta   = FPType(0);
        const DAAL_INT lda  = static_cast<DAAL_INT>(nCols);
        const DAAL_INT incx = 1;
        const DAAL_INT incy = 1;
        daal::internal::BlasInst<FPType, cpu>::xxgemv(&trans, &m, &n, &alpha, scratchRows, &lda, pivotPt, &incx, &beta, outDists, &incy);

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (int i = 0; i < count; i++)
        {
            FPType d2 = rowNorms2[i] + pivotNorm2 - FPType(2) * outDists[i];
            if (d2 < FPType(0)) d2 = FPType(0);
            outDists[i] = d2;
        }
        daal::internal::MathInst<FPType, cpu>::vSqrt(count, outDists, outDists);
    }
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

    /// Manhattan has no factorization that lets us batch via BLAS, but routing
    /// through a single blockDist entry point keeps callers (ball tree) symmetric
    /// across metrics. The pointDist body is already vectorized via PRAGMA_IVDEP.
    template <daal::internal::CpuType cpu>
    static void blockDist(const FPType * pivotPt, const FPType * scratchRows, const FPType * /*rowNorms2*/, int count, int nCols, FPType * outDists)
    {
        for (int i = 0; i < count; i++) outDists[i] = pointDist(pivotPt, scratchRows + i * nCols, nCols);
    }
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

    /// See ManhattanDist::blockDist — same rationale (no BLAS factorization).
    template <daal::internal::CpuType cpu>
    void blockDist(const FPType * pivotPt, const FPType * scratchRows, const FPType * /*rowNorms2*/, int count, int nCols, FPType * outDists) const
    {
        for (int i = 0; i < count; i++) outDists[i] = pointDist(pivotPt, scratchRows + i * nCols, nCols);
    }
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

    /// See ManhattanDist::blockDist — same rationale (no BLAS factorization).
    template <daal::internal::CpuType cpu>
    static void blockDist(const FPType * pivotPt, const FPType * scratchRows, const FPType * /*rowNorms2*/, int count, int nCols, FPType * outDists)
    {
        for (int i = 0; i < count; i++) outDists[i] = pointDist(pivotPt, scratchRows + i * nCols, nCols);
    }
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

    /// Delegate to the base metric's batched path then scale all distances by 1/α
    /// in one vectorized pass. Lets Euclidean keep its BLAS xxgemv fast path under
    /// alpha scaling (used for robust single linkage, alpha != 1).
    template <daal::internal::CpuType cpu>
    void blockDist(const FPType * pivotPt, const FPType * scratchRows, const FPType * rowNorms2, int count, int nCols, FPType * outDists) const
    {
        base.template blockDist<cpu>(pivotPt, scratchRows, rowNorms2, count, nCols, outDists);
        const FPType s = invAlpha;
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (int i = 0; i < count; i++) outDists[i] *= s;
    }
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
