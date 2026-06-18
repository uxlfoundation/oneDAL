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

/// Compute the squared L2 norm of a single row, vectorized.
///
/// Used by EuclideanDist::blockDist to cache `‖x_i‖²` between the three pivot
/// sweeps performed at each ball-tree node so the norm is read once and reused.
///
/// @tparam FPType Floating-point type
/// @tparam cpu    CPU dispatch tag
///
/// @param[in]  row   Pointer to the start of a row, length `nCols`
/// @param[in]  nCols Number of features (row length)
///
/// @return Sum of squared row entries
template <typename FPType, daal::internal::CpuType cpu>
static FPType rowNormSquared(const FPType * row, int nCols)
{
    FPType sum = FPType(0);
    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (int d = 0; d < nCols; d++) sum += row[d] * row[d];
    return sum;
}

/// Compute squared L2 norms for a contiguous block of row-major rows.
///
/// Used by ball-tree node construction to amortize `‖x‖²` across the three
/// pivot sweeps performed at the same node.
///
/// @tparam FPType Floating-point type
/// @tparam cpu    CPU dispatch tag
///
/// @param[in]  rows     Row-major buffer of size `count × nCols`
/// @param[in]  count    Number of rows
/// @param[in]  nCols    Number of features per row
/// @param[out] outNorms Output norms, length `count`
template <typename FPType, daal::internal::CpuType cpu>
static void rowNormsSquared(const FPType * rows, int count, int nCols, FPType * outNorms)
{
    for (int i = 0; i < count; i++) outNorms[i] = rowNormSquared<FPType, cpu>(rows + i * nCols, nCols);
}

/// Fill a symmetric `nRows × nRows` distance matrix in row-major layout using a
/// scalar metric functor.
///
/// Exploits symmetry: only the upper triangle (j >= i) is computed, the lower
/// triangle is mirrored. Outer parallelization over row blocks via
/// `daal::threader_for`. Diagonal entries are zeroed. Used by the dense
/// brute-force HDBSCAN for non-Euclidean metrics where there is no GEMM
/// identity to exploit; centralises the row-pair loop so Manhattan, Chebyshev,
/// and Minkowski share one implementation instead of three near duplicates.
///
/// @tparam FPType   Floating-point type
/// @tparam cpu      CPU dispatch tag
/// @tparam DistFunc Metric functor exposing `pointDist(a, b, nCols)`
///
/// @param[in]  data     Row-major input buffer of size `nRows × nCols`
/// @param[in]  nRows    Number of rows
/// @param[in]  nCols    Number of features per row
/// @param[in]  distFunc Distance functor instance
/// @param[out] outDist  Row-major output, length `nRows × nRows`
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
                const FPType d         = distFunc.template pointDist<cpu>(row_i, data + j * nCols, static_cast<int>(nCols));
                dist_row[j]            = d;
                outDist[j * nRows + i] = d;
            }
            dist_row[i] = FPType(0);
        }
    });
}

// =========================================================================
// Distance functors for parameterizing tree queries by metric.
//
// Each functor provides four methods:
//   - pointDist<cpu>(a, b, nCols)                 — full point-to-point distance
//   - bboxLowerBound<cpu>(q, lo, hi, nCols)       — minimum distance from query to a bbox
//   - planeDist<cpu>(diff)                        — distance to a splitting hyperplane (kd-tree)
//   - blockDist<cpu>(pivot, rows, rowNorms2, count, nCols, out)
//                                                 — pivot-to-block distances; Euclidean uses
//                                                   BLAS xxgemv + cached row norms²,
//                                                   non-Euclidean fall back to a vectorized loop.
//                                                   `rowNorms2` may be nullptr for non-Euclidean.
//
// All methods are templated on `cpu` so each per-CPU instantiation gets its
// own ISA-specific math/BLAS bindings. Used to parameterize HDBSCAN tree
// builds and Boruvka MRD queries by metric without code duplication.
// =========================================================================

/// Euclidean (L2) distance functor.
///
/// All methods are static (stateless metric). blockDist uses the identity
/// `‖x − p‖² = ‖x‖² + ‖p‖² − 2·⟨x, p⟩` to delegate the inner products to a
/// single BLAS xxgemv call, then sqrts in a vectorized pass.
///
/// @tparam FPType Floating-point type
template <typename FPType>
struct EuclideanDist
{
    /// Compute the L2 distance between two rows.
    ///
    /// @tparam cpu CPU dispatch tag
    ///
    /// @param[in]  a     First row, length `nCols`
    /// @param[in]  b     Second row, length `nCols`
    /// @param[in]  nCols Number of features
    ///
    /// @return `‖a − b‖₂`
    template <daal::internal::CpuType cpu>
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

    /// Compute the L2 distance from a query point to its nearest point in a
    /// bounding box (== 0 if the query lies inside the box).
    ///
    /// Used as a kd-tree pruning lower bound.
    ///
    /// @tparam cpu CPU dispatch tag
    ///
    /// @param[in]  query Query point, length `nCols`
    /// @param[in]  lo    Per-dimension lower bound, length `nCols`
    /// @param[in]  hi    Per-dimension upper bound, length `nCols`
    /// @param[in]  nCols Number of features
    ///
    /// @return Minimum L2 distance from `query` to the box `[lo, hi]`
    template <daal::internal::CpuType cpu>
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

    /// Distance from a query to a kd-tree axis-aligned splitting hyperplane,
    /// given the signed difference in the splitting coordinate.
    ///
    /// @tparam cpu CPU dispatch tag
    ///
    /// @param[in]  diff `query[splitDim] - splitVal`
    ///
    /// @return `|diff|` (axis-aligned plane distance for L2)
    template <daal::internal::CpuType cpu>
    static FPType planeDist(FPType diff)
    {
        return (diff < FPType(0)) ? -diff : diff;
    }

    /// Vectorized pivot-to-block Euclidean distance via BLAS xxgemv.
    ///
    /// Identity used: `‖x_i − p‖² = ‖x_i‖² + ‖p‖² − 2·⟨x_i, p⟩`. The inner
    /// products are computed as a single xxgemv on the row-major scratch
    /// buffer; `rowNorms2[i]` must equal `‖scratchRows[i]‖²` (see
    /// rowNormsSquared). Negative squared distances from rounding noise are
    /// clamped to zero before sqrt.
    ///
    /// @tparam cpu CPU dispatch tag for BLAS / vSqrt selection
    ///
    /// @param[in]  pivotPt     Pivot row, length `nCols`
    /// @param[in]  scratchRows Row-major batch, size `count × nCols`
    /// @param[in]  rowNorms2   Precomputed `‖scratchRows[i]‖²`, length `count`
    /// @param[in]  count       Number of rows in the batch
    /// @param[in]  nCols       Number of features
    /// @param[out] outDists    Output distances, length `count`
    template <daal::internal::CpuType cpu>
    static void blockDist(const FPType * pivotPt, const FPType * scratchRows, const FPType * rowNorms2, int count, int nCols, FPType * outDists)
    {
        const FPType pivotNorm2 = rowNormSquared<FPType, cpu>(pivotPt, nCols);

        // outDists ← scratchRows · pivotPt  (count vector)
        // GEMV: y = α·op(A)·x + β·y. The row-major scratchRows[count×nCols]
        // is a column-major nCols×count matrix A; we want
        // y[i] = ⟨scratchRows[i], pivotPt⟩ = Σ_d A[d,i]·pivotPt[d] = (Aᵀ·pivotPt)[i].
        // So trans='T' with m=nCols (rows of A), n=count (cols of A): x has length m,
        // y has length n.
        const char trans    = 'T';
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

/// Manhattan (L1) distance functor.
///
/// Stateless metric; same interface as EuclideanDist. blockDist falls through
/// to a vectorized scalar loop because L1 admits no GEMM identity.
///
/// @tparam FPType Floating-point type
template <typename FPType>
struct ManhattanDist
{
    /// Compute `‖a − b‖₁`.
    ///
    /// @tparam cpu CPU dispatch tag
    ///
    /// @param[in]  a     First row, length `nCols`
    /// @param[in]  b     Second row, length `nCols`
    /// @param[in]  nCols Number of features
    template <daal::internal::CpuType cpu>
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

    /// Minimum L1 distance from query point to a bbox `[lo, hi]`. 0 if inside.
    ///
    /// @tparam cpu CPU dispatch tag
    ///
    /// @param[in]  query Query point, length `nCols`
    /// @param[in]  lo    Lower bound per dimension, length `nCols`
    /// @param[in]  hi    Upper bound per dimension, length `nCols`
    /// @param[in]  nCols Number of features
    template <daal::internal::CpuType cpu>
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

    /// Distance from a query to an axis-aligned splitting plane.
    ///
    /// @tparam cpu CPU dispatch tag
    ///
    /// @param[in] diff `query[splitDim] - splitVal`
    template <daal::internal::CpuType cpu>
    static FPType planeDist(FPType diff)
    {
        return (diff < FPType(0)) ? -diff : diff;
    }

    /// Pivot-to-block Manhattan distance.
    ///
    /// L1 has no factorization that lets us batch via BLAS, but routing through
    /// a single blockDist entry point keeps ball-tree callers symmetric across
    /// metrics; pointDist is vectorized internally.
    ///
    /// @tparam cpu CPU dispatch tag (unused for this metric)
    ///
    /// @param[in]  pivotPt     Pivot row, length `nCols`
    /// @param[in]  scratchRows Row-major batch, size `count × nCols`
    /// @param[in]  rowNorms2   Unused (kept for interface symmetry with Euclidean)
    /// @param[in]  count       Number of rows
    /// @param[in]  nCols       Number of features
    /// @param[out] outDists    Output distances, length `count`
    template <daal::internal::CpuType cpu>
    static void blockDist(const FPType * pivotPt, const FPType * scratchRows, const FPType * /*rowNorms2*/, int count, int nCols, FPType * outDists)
    {
        for (int i = 0; i < count; i++) outDists[i] = pointDist<cpu>(pivotPt, scratchRows + i * nCols, nCols);
    }
};

/// Minkowski distance functor of arbitrary degree `p > 0`.
///
/// Stateful: holds `p` and `1/p`. blockDist falls through to a scalar loop.
///
/// @tparam FPType Floating-point type
template <typename FPType>
struct MinkowskiDist
{
    double p;    ///< Minkowski degree
    double invp; ///< Cached `1.0 / p`

    /// Construct a Minkowski functor of given degree.
    ///
    /// @param[in] degree Minkowski exponent `p > 0`
    MinkowskiDist(double degree) : p(degree), invp(1.0 / degree) {}

    /// Compute `(Σ |a_d − b_d|^p)^(1/p)`.
    ///
    /// @tparam cpu CPU dispatch tag
    ///
    /// @param[in]  a     First row, length `nCols`
    /// @param[in]  b     Second row, length `nCols`
    /// @param[in]  nCols Number of features
    template <daal::internal::CpuType cpu>
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

    /// Minimum Minkowski distance from a query to a bbox `[lo, hi]`.
    ///
    /// @tparam cpu CPU dispatch tag
    ///
    /// @param[in]  query Query point, length `nCols`
    /// @param[in]  lo    Lower bound per dimension, length `nCols`
    /// @param[in]  hi    Upper bound per dimension, length `nCols`
    /// @param[in]  nCols Number of features
    template <daal::internal::CpuType cpu>
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

    /// Distance from a query to an axis-aligned splitting plane.
    ///
    /// @tparam cpu CPU dispatch tag
    ///
    /// @param[in] diff `query[splitDim] - splitVal`
    template <daal::internal::CpuType cpu>
    FPType planeDist(FPType diff) const
    {
        return (diff < FPType(0)) ? -diff : diff;
    }

    /// Pivot-to-block Minkowski distance.
    ///
    /// Same rationale as ManhattanDist::blockDist — no BLAS factorization.
    ///
    /// @tparam cpu CPU dispatch tag (unused for this metric)
    ///
    /// @param[in]  pivotPt     Pivot row, length `nCols`
    /// @param[in]  scratchRows Row-major batch, size `count × nCols`
    /// @param[in]  rowNorms2   Unused
    /// @param[in]  count       Number of rows
    /// @param[in]  nCols       Number of features
    /// @param[out] outDists    Output distances, length `count`
    template <daal::internal::CpuType cpu>
    void blockDist(const FPType * pivotPt, const FPType * scratchRows, const FPType * /*rowNorms2*/, int count, int nCols, FPType * outDists) const
    {
        for (int i = 0; i < count; i++) outDists[i] = pointDist<cpu>(pivotPt, scratchRows + i * nCols, nCols);
    }
};

/// Chebyshev (L∞) distance functor.
///
/// Stateless. blockDist falls through to a scalar loop.
///
/// @tparam FPType Floating-point type
template <typename FPType>
struct ChebyshevDist
{
    /// Compute `max_d |a_d − b_d|`.
    ///
    /// @tparam cpu CPU dispatch tag
    ///
    /// @param[in]  a     First row, length `nCols`
    /// @param[in]  b     Second row, length `nCols`
    /// @param[in]  nCols Number of features
    template <daal::internal::CpuType cpu>
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

    /// Minimum Chebyshev distance from a query to a bbox `[lo, hi]`.
    ///
    /// @tparam cpu CPU dispatch tag
    ///
    /// @param[in]  query Query point, length `nCols`
    /// @param[in]  lo    Lower bound per dimension, length `nCols`
    /// @param[in]  hi    Upper bound per dimension, length `nCols`
    /// @param[in]  nCols Number of features
    template <daal::internal::CpuType cpu>
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

    /// Distance from a query to an axis-aligned splitting plane.
    ///
    /// @tparam cpu CPU dispatch tag
    ///
    /// @param[in] diff `query[splitDim] - splitVal`
    template <daal::internal::CpuType cpu>
    static FPType planeDist(FPType diff)
    {
        return (diff < FPType(0)) ? -diff : diff;
    }

    /// Pivot-to-block Chebyshev distance.
    ///
    /// Same rationale as ManhattanDist::blockDist — no BLAS factorization.
    ///
    /// @tparam cpu CPU dispatch tag (unused for this metric)
    ///
    /// @param[in]  pivotPt     Pivot row, length `nCols`
    /// @param[in]  scratchRows Row-major batch, size `count × nCols`
    /// @param[in]  rowNorms2   Unused
    /// @param[in]  count       Number of rows
    /// @param[in]  nCols       Number of features
    /// @param[out] outDists    Output distances, length `count`
    template <daal::internal::CpuType cpu>
    static void blockDist(const FPType * pivotPt, const FPType * scratchRows, const FPType * /*rowNorms2*/, int count, int nCols, FPType * outDists)
    {
        for (int i = 0; i < count; i++) outDists[i] = pointDist<cpu>(pivotPt, scratchRows + i * nCols, nCols);
    }
};

/// Bounded max-heap of the k nearest neighbors seen so far.
///
/// Ordering invariant: `dists_[0]` is the largest distance currently in the
/// heap (binary max-heap over `dists_`; `indices_` moves in lockstep). While
/// fewer than `capacity` neighbors have been pushed the heap keeps filling;
/// once full, `dists_[0]` is the current k-th nearest distance and `push()`
/// only replaces the top when a strictly closer neighbor arrives. `maxDist()`
/// returns that top or `+infinity` while the heap is not yet full, which lets
/// tree traversals use it as a pruning radius. The heap owns its storage via
/// `TArrayScalable` so each thread can allocate its own without external
/// `TlsMem` scaffolding.
///
/// @tparam FPType Floating-point type used for distances
/// @tparam cpu    CPU dispatch tag (selects the scalable allocator)
template <typename FPType, daal::internal::CpuType cpu>
struct KnnHeap
{
    /// Construct an empty heap with capacity `cap`.
    ///
    /// Allocates internal `dists_` / `indices_` arrays. After construction,
    /// `ok()` must be checked before use; allocation failure leaves the heap
    /// inert.
    ///
    /// @param[in] cap Maximum number of neighbors to keep
    KnnHeap(int cap) : capacity_(cap), size_(0), distsArr_(cap), indicesArr_(cap)
    {
        dists_   = distsArr_.get();
        indices_ = indicesArr_.get();
    }

    KnnHeap(const KnnHeap &)             = delete;
    KnnHeap & operator=(const KnnHeap &) = delete;

    /// True iff internal allocations succeeded.
    bool ok() const { return dists_ != nullptr && indices_ != nullptr; }

    /// Return the current k-th nearest distance, or `+inf` if the heap isn't full.
    ///
    /// Useful as a pruning radius for tree traversals.
    FPType maxDist() const { return (size_ > 0) ? dists_[0] : daal::services::internal::MaxVal<FPType>::get(); }

    /// Insert a candidate `(dist, idx)`; ignored if the heap is full and the
    /// distance is not strictly smaller than the current top.
    ///
    /// @param[in] dist Candidate distance
    /// @param[in] idx  Candidate point index
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
