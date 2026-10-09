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

#include "services/daal_defines.h" // DAAL_MALLOC_DEFAULT_ALIGNMENT
#include "src/algorithms/service_heap.h"
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

/// Squared L2 norm of a row with no alignment assumption; see `rowNormSquaredAligned`.
/// unpadded row of the caller's data buffer.
///
/// @tparam FPType Floating-point type
/// @tparam cpu    CPU dispatch tag
///
/// @param[in]  row   Pointer to the start of a row, length `nCols`
/// @param[in]  nCols Number of features (row length)
///
/// @return Sum of squared row entries
template <typename FPType, daal::internal::CpuType cpu>
static FPType rowNormSquared(const FPType * row, size_t nCols)
{
    FPType sum = FPType(0);
    PRAGMA_OMP_SIMD_ARGS(reduction(+ : sum))
    for (size_t d = 0; d < nCols; d++) sum += row[d] * row[d];
    return sum;
}

/// Squared L2 norm of a row that starts on a `DAAL_MALLOC_DEFAULT_ALIGNMENT` boundary.
///
/// @tparam FPType Floating-point type
/// @tparam cpu    CPU dispatch tag
///
/// @param[in]  row   Pointer to the start of an aligned row, length `nCols`
/// @param[in]  nCols Number of features (row length)
///
/// @return Sum of squared row entries
template <typename FPType, daal::internal::CpuType cpu>
static FPType rowNormSquaredAligned(const FPType * row, size_t nCols)
{
    FPType sum = FPType(0);
    PRAGMA_OMP_SIMD_ARGS(reduction(+ : sum) aligned(row : DAAL_MALLOC_DEFAULT_ALIGNMENT))
    for (size_t d = 0; d < nCols; d++) sum += row[d] * row[d];
    return sum;
}

/// Squared L2 norms of the rows of a padded, aligned row-major block; padding cells must be zero.
///
/// @tparam FPType Floating-point type
/// @tparam cpu    CPU dispatch tag
///
/// @param[in]  rows      Row-major buffer of size `count x rowStride`,
///                       base pointer aligned to `DAAL_MALLOC_DEFAULT_ALIGNMENT`
/// @param[in]  count     Number of rows
/// @param[in]  nCols     Number of features per row
/// @param[in]  rowStride Row stride of `rows` in elements (`>= nCols`),
///                       rounded up so `rows + i * rowStride` stays aligned
/// @param[out] outNorms  Output norms, length `count`
template <typename FPType, daal::internal::CpuType cpu>
static void rowNormsSquared(const FPType * rows, DAAL_INT count, size_t nCols, size_t rowStride, FPType * outNorms)
{
    for (DAAL_INT i = 0; i < count; i++) outNorms[i] = rowNormSquaredAligned<FPType, cpu>(rows + i * rowStride, nCols);
}

/// Round `nCols` up so every row of a row-major batch starts on a `DAAL_MALLOC_DEFAULT_ALIGNMENT`
/// boundary when the base pointer does.
///
/// @tparam FPType Floating-point type
/// @param[in] nCols Feature count
/// @return Padded stride in elements (>= `nCols`); equal to `nCols` when the
///         row size is already an aligned multiple.
template <typename FPType>
static inline size_t alignedRowStride(size_t nCols)
{
    constexpr size_t alignBytes = DAAL_MALLOC_DEFAULT_ALIGNMENT;
    constexpr size_t elemPerAln = alignBytes / sizeof(FPType);
    return ((nCols + elemPerAln - 1) / elemPerAln) * elemPerAln;
}

/// Fill the symmetric `nRows x nRows` distance matrix with a metric functor; the upper triangle
/// is computed and mirrored, and the diagonal is zero.
///
/// @tparam FPType   Floating-point type
/// @tparam cpu      CPU dispatch tag
/// @tparam DistFunc Metric functor exposing `pointDist(a, b, nCols)`
///
/// @param[in]  data     Row-major input buffer of size `nRows x nCols`
/// @param[in]  nRows    Number of rows
/// @param[in]  nCols    Number of features per row
/// @param[in]  distFunc Distance functor instance
/// @param[out] outDist  Row-major output, length `nRows x nRows`
template <typename FPType, daal::internal::CpuType cpu, typename DistFunc>
static void fillFullDistMatrix(const FPType * data, size_t nRows, size_t nCols, const DistFunc & distFunc, FPType * outDist)
{
    // Rows per parallel task.
    constexpr size_t blockSize = 256;
    const size_t nBlocks       = (nRows + blockSize - 1) / blockSize;

    daal::threader_for(nBlocks, 1, [&](size_t iBlock) {
        const size_t i_begin = iBlock * blockSize;
        const size_t i_end   = (i_begin + blockSize > nRows) ? nRows : i_begin + blockSize;
        for (size_t i = i_begin; i < i_end; i++)
        {
            const FPType * row_i = data + i * nCols;
            FPType * dist_row    = outDist + i * nRows;
            for (size_t j = i; j < nRows; j++)
            {
                const FPType d         = distFunc.template pointDist<cpu>(row_i, data + j * nCols, nCols);
                dist_row[j]            = d;
                outDist[j * nRows + i] = d;
            }
            dist_row[i] = FPType(0);
        }
    });
}

// =========================================================================
// Distance functors used by the tree builds and the Boruvka queries. Each provides
//   - pointDist<cpu>(a, b, nCols): point-to-point distance
//   - bboxLowerBound<cpu>(q, lo, hi, nCols): lower bound of the distance from q to a box
//   - blockDist<cpu>(pivot, rows, rowNorms2, count, nCols, rowStride, out): pivot-to-rows
//     distances; `rowNorms2` is used by Euclidean only, `rowStride >= nCols` is the padded stride.
// Only `EuclideanDist::blockDist` relies on aligned rows (see `alignedRowStride`).
// =========================================================================

/// Euclidean (L2) distance functor.
///
/// `blockDist` uses `||x - p||^2 = ||x||^2 + ||p||^2 - 2 <x, p>` with one xxgemv call.
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
    /// @return `||a - b||_2`
    template <daal::internal::CpuType cpu>
    static FPType pointDist(const FPType * a, const FPType * b, size_t nCols)
    {
        FPType sum = FPType(0);
        PRAGMA_OMP_SIMD_ARGS(reduction(+ : sum))
        for (size_t d = 0; d < nCols; d++)
        {
            const FPType diff = a[d] - b[d];
            sum += diff * diff;
        }
        return daal::internal::MathInst<FPType, cpu>::sSqrt(sum);
    }

    /// L2 distance from a query point to the nearest point of a box (0 inside the box).
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
    static FPType bboxLowerBound(const FPType * query, const FPType * lo, const FPType * hi, size_t nCols)
    {
        FPType sum = FPType(0);
        // OpenMP: reduction body uses `?:` rather than `if (...) x = ...`.
        PRAGMA_OMP_SIMD_ARGS(reduction(+ : sum))
        for (size_t d = 0; d < nCols; d++)
        {
            const FPType belowLo = (query[d] < lo[d]) ? (lo[d] - query[d]) : FPType(0);
            const FPType aboveHi = (query[d] > hi[d]) ? (query[d] - hi[d]) : FPType(0);
            const FPType excess  = belowLo + aboveHi;
            sum += excess * excess;
        }
        return daal::internal::MathInst<FPType, cpu>::sSqrt(sum);
    }

    /// Vectorized pivot-to-block Euclidean distance via BLAS xxgemv.
    ///
    /// `rowNorms2[i]` must equal `||scratchRows[i]||^2`; negative squared values from round-off
    /// are clamped to zero.
    ///
    /// @tparam cpu CPU dispatch tag for BLAS / vSqrt selection
    ///
    /// @param[in]  pivotPt     Pivot row, length `nCols`
    /// @param[in]  scratchRows Row-major batch, size `count x rowStride`
    /// @param[in]  rowNorms2   Precomputed `||scratchRows[i]||^2`, length `count`
    /// @param[in]  count       Number of rows in the batch
    /// @param[in]  nCols       Number of features
    /// @param[in]  rowStride   Row stride of `scratchRows` in elements (`>= nCols`);
    ///                         padding columns must be zero
    /// @param[out] outDists    Output distances, length `count`
    template <daal::internal::CpuType cpu>
    static void blockDist(const FPType * pivotPt, const FPType * scratchRows, const FPType * rowNorms2, DAAL_INT count, size_t nCols,
                          size_t rowStride, FPType * outDists)
    {
        const FPType pivotNorm2 = rowNormSquared<FPType, cpu>(pivotPt, nCols);

        // outDists = scratchRows * pivotPt  (count vector)
        // Row-major scratchRows is a column-major rowStride x count matrix, so trans=T with m = nCols
        // skips the padding rows.
        const char trans    = 'T';
        const DAAL_INT m    = static_cast<DAAL_INT>(nCols);
        const DAAL_INT n    = count;
        const FPType alpha  = FPType(1);
        const FPType beta   = FPType(0);
        const DAAL_INT lda  = static_cast<DAAL_INT>(rowStride);
        const DAAL_INT incx = 1;
        const DAAL_INT incy = 1;
        daal::internal::BlasInst<FPType, cpu>::xxgemv(&trans, &m, &n, &alpha, scratchRows, &lda, pivotPt, &incx, &beta, outDists, &incy);

        // `rowNorms2` and `outDists` come from TArrayScalable, aligned to DAAL_MALLOC_DEFAULT_ALIGNMENT.
        PRAGMA_OMP_SIMD_ARGS(aligned(rowNorms2, outDists : DAAL_MALLOC_DEFAULT_ALIGNMENT))
        for (DAAL_INT i = 0; i < count; i++)
        {
            const FPType d2 = rowNorms2[i] + pivotNorm2 - FPType(2) * outDists[i];
            outDists[i]     = (d2 < FPType(0)) ? FPType(0) : d2;
        }
        daal::internal::MathInst<FPType, cpu>::vSqrt(count, outDists, outDists);
    }
};

/// Manhattan (L1) distance functor.
///
/// @tparam FPType Floating-point type
template <typename FPType>
struct ManhattanDist
{
    /// Compute `||a - b||_1`.
    ///
    /// @tparam cpu CPU dispatch tag
    ///
    /// @param[in]  a     First row, length `nCols`
    /// @param[in]  b     Second row, length `nCols`
    /// @param[in]  nCols Number of features
    template <daal::internal::CpuType cpu>
    static FPType pointDist(const FPType * a, const FPType * b, size_t nCols)
    {
        FPType sum = FPType(0);
        PRAGMA_OMP_SIMD_ARGS(reduction(+ : sum))
        for (size_t d = 0; d < nCols; d++)
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
    static FPType bboxLowerBound(const FPType * query, const FPType * lo, const FPType * hi, size_t nCols)
    {
        FPType sum = FPType(0);
        // OpenMP: reduction body uses `?:` rather than `if (...) x = ...`.
        PRAGMA_OMP_SIMD_ARGS(reduction(+ : sum))
        for (size_t d = 0; d < nCols; d++)
        {
            const FPType belowLo = (query[d] < lo[d]) ? (lo[d] - query[d]) : FPType(0);
            const FPType aboveHi = (query[d] > hi[d]) ? (query[d] - hi[d]) : FPType(0);
            sum += belowLo + aboveHi;
        }
        return sum;
    }

    /// Pivot-to-block Manhattan distance.
    ///
    /// @tparam cpu CPU dispatch tag (unused for this metric)
    ///
    /// @param[in]  pivotPt     Pivot row, length `nCols`
    /// @param[in]  scratchRows Row-major batch, size `count x rowStride`
    /// @param[in]  rowNorms2   Unused (kept for interface symmetry with Euclidean)
    /// @param[in]  count       Number of rows
    /// @param[in]  nCols       Number of features
    /// @param[in]  rowStride   Row stride of `scratchRows` in elements (`>= nCols`)
    /// @param[out] outDists    Output distances, length `count`
    template <daal::internal::CpuType cpu>
    static void blockDist(const FPType * pivotPt, const FPType * scratchRows, const FPType * /*rowNorms2*/, DAAL_INT count, size_t nCols,
                          size_t rowStride, FPType * outDists)
    {
        for (DAAL_INT i = 0; i < count; i++) outDists[i] = pointDist<cpu>(pivotPt, scratchRows + i * rowStride, nCols);
    }
};

/// Minkowski distance functor of arbitrary degree `p > 0`.
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

    /// Compute `(sum_d |a_d - b_d|^p) ^ (1/p)`.
    ///
    /// @tparam cpu CPU dispatch tag
    ///
    /// @param[in]  a     First row, length `nCols`
    /// @param[in]  b     Second row, length `nCols`
    /// @param[in]  nCols Number of features
    template <daal::internal::CpuType cpu>
    FPType pointDist(const FPType * a, const FPType * b, size_t nCols) const
    {
        const FPType pFP    = static_cast<FPType>(p);
        const FPType invpFP = static_cast<FPType>(invp);
        FPType sum          = FPType(0);
        for (size_t d = 0; d < nCols; d++)
        {
            const FPType diff = a[d] - b[d];
            const FPType absd = (diff < FPType(0)) ? -diff : diff;
            sum += daal::internal::MathInst<FPType, cpu>::sPowx(absd, pFP);
        }
        return daal::internal::MathInst<FPType, cpu>::sPowx(sum, invpFP);
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
    FPType bboxLowerBound(const FPType * query, const FPType * lo, const FPType * hi, size_t nCols) const
    {
        const FPType pFP    = static_cast<FPType>(p);
        const FPType invpFP = static_cast<FPType>(invp);
        FPType sum          = FPType(0);
        for (size_t d = 0; d < nCols; d++)
        {
            const FPType belowLo = (query[d] < lo[d]) ? (lo[d] - query[d]) : FPType(0);
            const FPType aboveHi = (query[d] > hi[d]) ? (query[d] - hi[d]) : FPType(0);
            const FPType excess  = belowLo + aboveHi;
            sum += daal::internal::MathInst<FPType, cpu>::sPowx(excess, pFP);
        }
        return daal::internal::MathInst<FPType, cpu>::sPowx(sum, invpFP);
    }

    /// Pivot-to-block Minkowski distance.
    ///
    /// Same rationale as ManhattanDist::blockDist -- no BLAS factorization.
    ///
    /// @tparam cpu CPU dispatch tag (unused for this metric)
    ///
    /// @param[in]  pivotPt     Pivot row, length `nCols`
    /// @param[in]  scratchRows Row-major batch, size `count x rowStride`
    /// @param[in]  rowNorms2   Unused
    /// @param[in]  count       Number of rows
    /// @param[in]  nCols       Number of features
    /// @param[in]  rowStride   Row stride of `scratchRows` in elements (`>= nCols`)
    /// @param[out] outDists    Output distances, length `count`
    template <daal::internal::CpuType cpu>
    void blockDist(const FPType * pivotPt, const FPType * scratchRows, const FPType * /*rowNorms2*/, DAAL_INT count, size_t nCols, size_t rowStride,
                   FPType * outDists) const
    {
        for (DAAL_INT i = 0; i < count; i++) outDists[i] = pointDist<cpu>(pivotPt, scratchRows + i * rowStride, nCols);
    }
};

/// Chebyshev (L-infinity) distance functor.
///
/// @tparam FPType Floating-point type
template <typename FPType>
struct ChebyshevDist
{
    /// Compute `max_d |a_d - b_d|`.
    ///
    /// @tparam cpu CPU dispatch tag
    ///
    /// @param[in]  a     First row, length `nCols`
    /// @param[in]  b     Second row, length `nCols`
    /// @param[in]  nCols Number of features
    template <daal::internal::CpuType cpu>
    static FPType pointDist(const FPType * a, const FPType * b, size_t nCols)
    {
        FPType mx = FPType(0);
        // `omp simd` reductions need the `?:` form rather than `if`.
        PRAGMA_OMP_SIMD_ARGS(reduction(max : mx))
        for (size_t d = 0; d < nCols; d++)
        {
            const FPType diff = a[d] - b[d];
            const FPType absd = (diff < FPType(0)) ? -diff : diff;
            mx                = (absd > mx) ? absd : mx;
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
    static FPType bboxLowerBound(const FPType * query, const FPType * lo, const FPType * hi, size_t nCols)
    {
        FPType mx = FPType(0);
        // OpenMP: reduction body uses `?:` rather than `if (...) x = ...`.
        PRAGMA_OMP_SIMD_ARGS(reduction(max : mx))
        for (size_t d = 0; d < nCols; d++)
        {
            const FPType belowLo = (query[d] < lo[d]) ? (lo[d] - query[d]) : FPType(0);
            const FPType aboveHi = (query[d] > hi[d]) ? (query[d] - hi[d]) : FPType(0);
            const FPType excess  = belowLo + aboveHi;
            mx                   = (excess > mx) ? excess : mx;
        }
        return mx;
    }

    /// Pivot-to-block Chebyshev distance.
    ///
    /// Same rationale as ManhattanDist::blockDist -- no BLAS factorization.
    ///
    /// @tparam cpu CPU dispatch tag (unused for this metric)
    ///
    /// @param[in]  pivotPt     Pivot row, length `nCols`
    /// @param[in]  scratchRows Row-major batch, size `count x rowStride`
    /// @param[in]  rowNorms2   Unused
    /// @param[in]  count       Number of rows
    /// @param[in]  nCols       Number of features
    /// @param[in]  rowStride   Row stride of `scratchRows` in elements (`>= nCols`)
    /// @param[out] outDists    Output distances, length `count`
    template <daal::internal::CpuType cpu>
    static void blockDist(const FPType * pivotPt, const FPType * scratchRows, const FPType * /*rowNorms2*/, DAAL_INT count, size_t nCols,
                          size_t rowStride, FPType * outDists)
    {
        for (DAAL_INT i = 0; i < count; i++) outDists[i] = pointDist<cpu>(pivotPt, scratchRows + i * rowStride, nCols);
    }
};

/// Return the `k`-th smallest entry of `values[0, n)` without modifying or copying the input.
///
/// Keeps a max-heap of the `k` smallest values seen, so the scratch is `k` elements.
///
/// @tparam FPType Floating-point type
/// @tparam cpu    CPU dispatch tag
///
/// @param[in]  values  Input values, length `n`, left untouched
/// @param[in]  n       Number of input values
/// @param[in]  k       Rank to select, `1 <= k <= n`
/// @param[out] heapBuf Caller-owned scratch of at least `k` elements; holds the `k` smallest
///                     values in heap order on return
///
/// @return The `k`-th smallest value of `values[0, n)`
template <typename FPType, daal::internal::CpuType cpu>
static FPType kthSmallestBounded(const FPType * values, size_t n, size_t k, FPType * heapBuf)
{
    const auto less = [](FPType a, FPType b) { return a < b; };
    for (size_t i = 0; i < k; i++) heapBuf[i] = values[i];
    daal::algorithms::internal::makeMaxHeap<cpu>(heapBuf, heapBuf + k, less);

    for (size_t i = k; i < n; i++)
    {
        if (values[i] < heapBuf[0])
        {
            heapBuf[0] = values[i];
            daal::algorithms::internal::internalAdjustMaxHeap<cpu>(heapBuf, heapBuf + k, k, size_t(0), less);
        }
    }
    return heapBuf[0];
}

/// Bounded max-heap of the k nearest neighbors seen so far.
///
/// `dists_[0]` is the largest distance kept; `indices_` moves with `dists_`. Once full, `push()`
/// replaces the top only for a strictly closer neighbor.
///
/// @tparam FPType Floating-point type used for distances
/// @tparam cpu    CPU dispatch tag (selects the scalable allocator)
template <typename FPType, daal::internal::CpuType cpu>
struct KnnHeap
{
    /// Construct an empty heap with capacity `cap`.
    ///
    /// Check `ok()` before use; an allocation failure leaves the heap inert.
    ///
    /// @param[in] cap Maximum number of neighbors to keep
    KnnHeap(DAAL_INT cap) : capacity_(cap), size_(0), distsArr_(cap), indicesArr_(cap)
    {
        dists_   = distsArr_.get();
        indices_ = indicesArr_.get();
    }

    KnnHeap(const KnnHeap &)             = delete;
    KnnHeap & operator=(const KnnHeap &) = delete;

    /// True iff internal allocations succeeded.
    bool ok() const { return dists_ != nullptr && indices_ != nullptr; }

    /// Return the current k-th nearest distance, or `+inf` if the heap isn't full.
    FPType maxDist() const { return (size_ == capacity_) ? dists_[0] : daal::services::internal::MaxVal<FPType>::get(); }

    /// Insert a candidate `(dist, idx)`; ignored if the heap is full and the
    /// distance is not strictly smaller than the current top.
    ///
    /// @param[in] dist Candidate distance
    /// @param[in] idx  Candidate point index
    void push(FPType dist, DAAL_INT idx)
    {
        if (size_ < capacity_)
        {
            dists_[size_]   = dist;
            indices_[size_] = idx;
            size_++;
            DAAL_INT i = size_ - 1;
            while (i > 0)
            {
                DAAL_INT parent = (i - 1) / 2;
                if (dists_[i] > dists_[parent])
                {
                    services::internal::swap<cpu>(dists_[i], dists_[parent]);
                    services::internal::swap<cpu>(indices_[i], indices_[parent]);
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
            DAAL_INT i  = 0;
            while (true)
            {
                DAAL_INT l       = 2 * i + 1;
                DAAL_INT r       = 2 * i + 2;
                DAAL_INT largest = i;
                if (l < size_ && dists_[l] > dists_[largest]) largest = l;
                if (r < size_ && dists_[r] > dists_[largest]) largest = r;
                if (largest != i)
                {
                    services::internal::swap<cpu>(dists_[i], dists_[largest]);
                    services::internal::swap<cpu>(indices_[i], indices_[largest]);
                    i = largest;
                }
                else
                    break;
            }
        }
    }

private:
    DAAL_INT capacity_;
    DAAL_INT size_;
    daal::services::internal::TArrayScalable<FPType, cpu> distsArr_;
    daal::services::internal::TArrayScalable<DAAL_INT, cpu> indicesArr_;
    FPType * dists_;     // distances from points in the heap to the current query (heap root is the largest)
    DAAL_INT * indices_; // indices of points in the heap, kept in lockstep with dists_
};

} // namespace internal
} // namespace hdbscan
} // namespace algorithms
} // namespace daal
