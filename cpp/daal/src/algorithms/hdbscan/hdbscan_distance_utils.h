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

#include "services/daal_defines.h" // DAAL_MALLOC_DEFAULT_ALIGNMENT
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
/// General-purpose helper: the input row pointer alignment is not assumed
/// (call sites include unpadded rows such as `data + i * nCols`), so no
/// `aligned(...)` clause is used. For rows guaranteed to start on a
/// `DAAL_MALLOC_DEFAULT_ALIGNMENT` boundary (e.g. per-row slices of a padded
/// `TArrayScalable` buffer with `alignedRowStride`), use
/// `rowNormSquaredAligned` instead.
///
/// Used by `EuclideanDist::blockDist` to compute `||pivotPt||^2` from an
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

/// Aligned variant of `rowNormSquared`.
///
/// Same math as `rowNormSquared` but the inner reduction carries an
/// `aligned(row : DAAL_MALLOC_DEFAULT_ALIGNMENT)` SIMD clause. The caller
/// must guarantee that `row` starts on a `DAAL_MALLOC_DEFAULT_ALIGNMENT`
/// boundary; passing a misaligned pointer is undefined behavior. Used by
/// `rowNormsSquared` on per-row slices of a padded scratch buffer whose
/// stride was rounded up via `alignedRowStride`.
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

/// Compute squared L2 norms for a row-major block with per-row padding.
///
/// Used by ball-tree node construction to amortize `||x||^2` across the three
/// pivot sweeps performed at the same node. Padded rows keep the pointer to
/// each row on the same alignment boundary as the base pointer; padding cells
/// must be zero, so they contribute nothing to the sum of squares. Each row
/// is dispatched through `rowNormSquaredAligned`, which carries the
/// `aligned(row : DAAL_MALLOC_DEFAULT_ALIGNMENT)` clause on its inner
/// reduction.
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
    // Vectorization lives inside `rowNormSquaredAligned` (inner sum reduction
    // over `nCols` with an aligned clause on each row start); the outer loop
    // is a scalar sequence of independent reductions.
    for (DAAL_INT i = 0; i < count; i++) outNorms[i] = rowNormSquaredAligned<FPType, cpu>(rows + i * rowStride, nCols);
}

/// Round `nCols` up so that every row of a row-major batch starts on a
/// `DAAL_MALLOC_DEFAULT_ALIGNMENT` byte boundary. Callers that own an aligned
/// base pointer (e.g. `TArrayScalable::get()`) can then reuse the same
/// alignment guarantee for every row.
///
/// Two consumers currently expose the guarantee to a SIMD clause:
///   - `EuclideanDist::blockDist` finalize pass carries
///     `aligned(rowNorms2, outDists : ...)` on its `omp simd`, and BLAS
///     xxgemv picks up the aligned stride via `lda = rowStride`.
///   - `rowNormsSquared` delegates each row to `rowNormSquaredAligned`, whose
///     inner reduction carries `aligned(row : ...)`. `rowNormSquared` (no
///     `Aligned` suffix) is used for unpadded call sites (`pivotNorm2` in
///     `EuclideanDist::blockDist`) and does not carry an aligned clause.
/// The non-Euclidean `blockDist` variants run a scalar outer loop over
/// `count` that delegates to `pointDist`, whose input row alignment is not
/// guaranteed (see the alignment policy comment below); alignment is still a
/// correctness guarantee for their inner reductions (they never cross a row
/// boundary).
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

/// Fill a symmetric `nRows x nRows` distance matrix in row-major layout using a
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
/// @param[in]  data     Row-major input buffer of size `nRows x nCols`
/// @param[in]  nRows    Number of rows
/// @param[in]  nCols    Number of features per row
/// @param[in]  distFunc Distance functor instance
/// @param[out] outDist  Row-major output, length `nRows x nRows`
template <typename FPType, daal::internal::CpuType cpu, typename DistFunc>
static void fillFullDistMatrix(const FPType * data, size_t nRows, size_t nCols, const DistFunc & distFunc, FPType * outDist)
{
    // Row block size for the outer parallelization. Set to 256 to match the
    // block sizes used elsewhere in DAAL at runtime (covariance / distance
    // primitives in service_kernel_math.h); large enough to amortize scheduler
    // overhead per task while keeping the inner j-loop cache-friendly. Not
    // exposed through the parameters system today; if profiling shows
    // sensitivity for very large nCols, revisit by promoting to a tunable in
    // Parameter.
    constexpr size_t blockSize = 256;
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
                const FPType d         = distFunc.template pointDist<cpu>(row_i, data + j * nCols, nCols);
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
// Each functor provides three methods:
//   - pointDist<cpu>(a, b, nCols)                 -- full point-to-point distance
//   - bboxLowerBound<cpu>(q, lo, hi, nCols)       -- minimum distance from query to a bbox
//   - blockDist<cpu>(pivot, rows, rowNorms2, count, nCols, rowStride, out)
//                                                 -- pivot-to-block distances; Euclidean uses
//                                                   BLAS xxgemv + cached row norms^2,
//                                                   non-Euclidean fall back to a vectorized loop.
//                                                   `rowNorms2` may be nullptr for non-Euclidean.
//                                                   `rowStride` (>= nCols) is the row stride of
//                                                   `rows` in elements; padded past nCols to
//                                                   preserve base-pointer alignment on every row.
//
// The kd-tree "distance to a splitting hyperplane" reduces to `|diff|` for
// every L_p metric supported here (splits are axis-aligned) and is applied at
// a single call site in knnQuery; it is inlined there rather than added to
// each functor.
//
// All methods are templated on `cpu` so each per-CPU instantiation gets its
// own ISA-specific math/BLAS bindings. Used to parameterize HDBSCAN tree
// builds and Boruvka MRD queries by metric without code duplication.
//
// Alignment policy for the SIMD inner loops:
//   - `pointDist` receives arbitrary row starts (`data + i*nCols`), which
//     are not guaranteed to be on a `DAAL_MALLOC_DEFAULT_ALIGNMENT`
//     boundary, so no `aligned(...)` clause is used inside pointDist. This
//     also applies to non-Euclidean `blockDist` bodies whose inner
//     accumulator delegates to `pointDist` per row.
//   - `bboxLowerBound` receives `bboxLo + nodeIdx*nCols` where `bboxLo` is a
//     `TArray` base but the row stride is `nCols` (unpadded), so per-row
//     alignment is also not guaranteed. No `aligned(...)` clause is used.
//   - `blockDist` for Euclidean operates on padded `scratchRows` (per-row
//     start alignment is guaranteed via `alignedRowStride`, see
//     `rowNormsSquared` and `gatherRows`) plus `TArrayScalable`-backed
//     `rowNorms2` / `outDists`. The vectorizable pass -- the finalize loop
//     that combines `rowNorms2`, `pivotNorm2`, and `outDists` -- carries an
//     `aligned(rowNorms2, outDists : DAAL_MALLOC_DEFAULT_ALIGNMENT)` clause,
//     so the padding cost is realized as SIMD alignment there. The xxgemv
//     itself does not need a SIMD clause; MKL BLAS picks up the alignment
//     internally via `lda = rowStride`.
//   - The `pivotNorm2 = rowNormSquared(pivotPt, nCols)` in
//     `EuclideanDist::blockDist` reads from an unpadded row of the caller's
//     data buffer, so its internal reduction does not carry an aligned
//     clause. This is the only vectorized read in `blockDist` that touches
//     an unpadded pointer.
// =========================================================================

/// Euclidean (L2) distance functor.
///
/// All methods are static (stateless metric). blockDist uses the identity
/// `||x - p||^2 = ||x||^2 + ||p||^2 - 2 * <x, p>` to delegate the inner
/// products to a single BLAS xxgemv call, then sqrts in a vectorized pass.
///
/// @tparam FPType Floating-point type
template <typename FPType>
struct EuclideanDist
{
    /// Compute the L2 distance between two rows.
    ///
    /// Uses `MathInst::sSqrt` for the final sqrt to match the CPU-specific
    /// math dispatch used elsewhere in DAAL (e.g. em_gmm, zscore, pca, cordistance).
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
    /// Identity used: `||x_i - p||^2 = ||x_i||^2 + ||p||^2 - 2 * <x_i, p>`.
    /// The inner products are computed as a single xxgemv on the row-major
    /// scratch buffer; `rowNorms2[i]` must equal `||scratchRows[i]||^2` (see
    /// rowNormsSquared). Negative squared distances from rounding noise are
    /// clamped to zero before sqrt.
    ///
    /// @tparam cpu CPU dispatch tag for BLAS / vSqrt selection
    ///
    /// @param[in]  pivotPt     Pivot row, length `nCols`
    /// @param[in]  scratchRows Row-major batch, size `count x rowStride`
    /// @param[in]  rowNorms2   Precomputed `||scratchRows[i]||^2`, length `count`
    /// @param[in]  count       Number of rows in the batch
    /// @param[in]  nCols       Number of features
    /// @param[in]  rowStride   Row stride of `scratchRows` in elements (`>= nCols`);
    ///                         padded past `nCols` so each row starts on the same
    ///                         alignment boundary as the base pointer. Padding
    ///                         columns must be zero so they don't contribute to
    ///                         the inner product.
    /// @param[out] outDists    Output distances, length `count`
    template <daal::internal::CpuType cpu>
    static void blockDist(const FPType * pivotPt, const FPType * scratchRows, const FPType * rowNorms2, DAAL_INT count, size_t nCols,
                          size_t rowStride, FPType * outDists)
    {
        const FPType pivotNorm2 = rowNormSquared<FPType, cpu>(pivotPt, nCols);

        // outDists = scratchRows * pivotPt  (count vector)
        // GEMV: y = alpha * op(A) * x + beta * y. The row-major
        // scratchRows[count x rowStride] is a column-major rowStride x count
        // matrix A; we want y[i] = <scratchRows[i, :nCols], pivotPt> =
        // sum_d A[d,i]*pivotPt[d] for d < nCols. The trailing padding columns
        // of the row-major layout become extra leading rows of A that never
        // contribute (we set m=nCols, ignoring rows [nCols, rowStride)).
        // So trans='T' with m=nCols (rows of A picked from the top), n=count
        // (cols of A): x has length m, y has length n, lda = rowStride.
        const char trans    = 'T';
        const DAAL_INT m    = static_cast<DAAL_INT>(nCols);
        const DAAL_INT n    = count;
        const FPType alpha  = FPType(1);
        const FPType beta   = FPType(0);
        const DAAL_INT lda  = static_cast<DAAL_INT>(rowStride);
        const DAAL_INT incx = 1;
        const DAAL_INT incy = 1;
        daal::internal::BlasInst<FPType, cpu>::xxgemv(&trans, &m, &n, &alpha, scratchRows, &lda, pivotPt, &incx, &beta, outDists, &incy);

        // `rowNorms2` and `outDists` are `TArrayScalable`-backed at every call
        // site (see the ball-tree build in `hdbscan_ball_tree_batch_impl.i`);
        // their starts are aligned to `DAAL_MALLOC_DEFAULT_ALIGNMENT`.
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
/// Stateless metric; same interface as EuclideanDist. blockDist falls through
/// to a vectorized scalar loop because L1 admits no GEMM identity.
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
    /// L1 has no factorization that lets us batch via BLAS, but routing through
    /// a single blockDist entry point keeps ball-tree callers symmetric across
    /// metrics; pointDist is vectorized internally.
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

    /// Compute `(sum_d |a_d - b_d|^p) ^ (1/p)`.
    ///
    /// Per-element `|d|^p` goes through `MathInst<FPType, cpu>::sPowx` so each
    /// per-CPU instantiation binds to its ISA-specific power routine, matching
    /// the dispatch convention used elsewhere in DAAL (e.g. em_gmm, zscore).
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
/// Stateless. blockDist falls through to a scalar loop.
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
        // OpenMP requires reduction-body updates to be expression-form (via `?:`)
        // rather than branch-form (`if (...) x = ...`) so the compiler can safely
        // fold each lane into the max-reduction pattern under `omp simd`. That
        // form is also why the pragma comes through PRAGMA_OMP_SIMD_MINMAX_ARGS:
        // plain clang cannot treat a select-based FP min/max as a reduction, and
        // says so as an error under `-Werror` (see service_defines.h).
        PRAGMA_OMP_SIMD_MINMAX_ARGS(reduction(max : mx))
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
        PRAGMA_OMP_SIMD_MINMAX_ARGS(reduction(max : mx))
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
    ///
    /// Useful as a pruning radius for tree traversals.
    FPType maxDist() const { return (size_ > 0) ? dists_[0] : daal::services::internal::MaxVal<FPType>::get(); }

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
