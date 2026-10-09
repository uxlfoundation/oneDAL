/* file: hdbscan_centers_impl.i */
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

#include "src/algorithms/hdbscan/hdbscan_kernel.h"
#include "src/algorithms/service_error_handling.h"
#include "src/algorithms/service_kernel_math.h"
#include "src/algorithms/service_threading.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_memory.h"
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

using daal::internal::CpuType;
using daal::internal::ReadRows;
using daal::internal::WriteOnlyRows;
using daal::services::internal::TArray;
using daal::services::internal::TArrayCalloc;

/// Create the pairwise distance of `pairwiseDistance` between the rows of `nt` and themselves.
///
/// Euclidean and cosine are the final distances after `finalize`; manhattan is Minkowski with
/// degree 1, which `finalize` leaves as is.
///
/// @tparam algorithmFPType Floating-point type
/// @tparam cpu             CPU dispatch tag
///
/// @param[in] nt               Rows the distances are taken between
/// @param[in] pairwiseDistance Distance metric of the fit
/// @param[in] minkowskiDegree  Exponent `p` for the Minkowski distance
template <typename algorithmFPType, CpuType cpu>
static algorithms::internal::PairwiseDistances<algorithmFPType, cpu> * createPairwiseDistances(
    const NumericTable & nt, algorithms::internal::PairwiseDistanceType pairwiseDistance, double minkowskiDegree)
{
    using namespace algorithms::internal;
    switch (pairwiseDistance)
    {
    case PairwiseDistanceType::cosine: return new CosineDistances<algorithmFPType, cpu>(nt, nt);
    case PairwiseDistanceType::manhattan: return new MinkowskiDistances<algorithmFPType, cpu>(nt, nt, true, 1.0);
    case PairwiseDistanceType::minkowski: return new MinkowskiDistances<algorithmFPType, cpu>(nt, nt, true, minkowskiDegree);
    case PairwiseDistanceType::chebyshev: return new ChebyshevDistances<algorithmFPType, cpu>(nt, nt);
    case PairwiseDistanceType::euclidean:
    default: return new EuclideanDistances<algorithmFPType, cpu>(nt, nt, true);
    }
}

/// Probability-weighted mean of every cluster's members; empty clusters get zero rows.
///
/// @tparam algorithmFPType Floating-point type
/// @tparam cpu             CPU dispatch tag
///
/// @param[in]  data      Row-major data, `nRows x nCols`
/// @param[in]  labels    Cluster id per point, length `nRows`; -1 is noise
/// @param[in]  weights   Membership probability per point, length `nRows`
/// @param[in]  nRows     Number of points
/// @param[in]  nCols     Number of features
/// @param[in]  nClusters Number of clusters
/// @param[out] centroids Row-major centroids, `nClusters x nCols`
template <typename algorithmFPType, CpuType cpu>
static services::Status computeCentroids(const algorithmFPType * data, const int * labels, const algorithmFPType * weights, size_t nRows,
                                         size_t nCols, size_t nClusters, algorithmFPType * centroids)
{
    TArrayCalloc<double, cpu> weightSumsArr(nClusters);
    TArrayCalloc<double, cpu> sumsArr(nClusters * nCols);
    double * weightSums = weightSumsArr.get();
    double * sums       = sumsArr.get();
    DAAL_CHECK_MALLOC(weightSums && sums);

    for (size_t i = 0; i < nRows; i++)
    {
        const int label = labels[i];
        if (label < 0 || static_cast<size_t>(label) >= nClusters) continue;
        const double w = weights[i];
        weightSums[label] += w;
        double * rowOut               = sums + label * nCols;
        const algorithmFPType * rowIn = data + i * nCols;
        PRAGMA_OMP_SIMD
        for (size_t d = 0; d < nCols; d++) rowOut[d] += w * rowIn[d];
    }

    for (size_t k = 0; k < nClusters; k++)
    {
        const double inv = (weightSums[k] > 0) ? 1.0 / weightSums[k] : 0.0;
        PRAGMA_OMP_SIMD
        for (size_t d = 0; d < nCols; d++) centroids[k * nCols + d] = algorithmFPType(sums[k * nCols + d] * inv);
    }
    return services::Status();
}

/// Medoid of every cluster: the member `i` minimizing `sum_j dist(i, j) * weights[j]` over the
/// members `j`. Ties go to the lowest row; empty clusters get zero rows.
///
/// @tparam algorithmFPType Floating-point type
/// @tparam cpu             CPU dispatch tag
///
/// @param[in]  data             Row-major data, `nRows x nCols`
/// @param[in]  labels           Cluster id per point, length `nRows`; -1 is noise
/// @param[in]  weights          Membership probability per point, length `nRows`
/// @param[in]  nRows            Number of points
/// @param[in]  nCols            Number of features
/// @param[in]  nClusters        Number of clusters
/// @param[in]  pairwiseDistance Distance metric of the fit
/// @param[in]  minkowskiDegree  Exponent `p` for the Minkowski distance
/// @param[out] medoids          Row-major medoids, `nClusters x nCols`
template <typename algorithmFPType, CpuType cpu>
static services::Status computeMedoids(const algorithmFPType * data, const int * labels, const algorithmFPType * weights, size_t nRows, size_t nCols,
                                       size_t nClusters, algorithms::internal::PairwiseDistanceType pairwiseDistance, double minkowskiDegree,
                                       algorithmFPType * medoids)
{
    // Members of each cluster in row order, CSR style.
    TArrayCalloc<size_t, cpu> offsetsArr(nClusters + 1);
    size_t * offsets = offsetsArr.get();
    DAAL_CHECK_MALLOC(offsets);
    for (size_t i = 0; i < nRows; i++)
    {
        if (labels[i] >= 0 && static_cast<size_t>(labels[i]) < nClusters) offsets[labels[i] + 1]++;
    }
    for (size_t k = 0; k < nClusters; k++) offsets[k + 1] += offsets[k];
    const size_t memberCount = offsets[nClusters];

    services::internal::service_memset<algorithmFPType, cpu>(medoids, algorithmFPType(0), nClusters * nCols);
    if (memberCount == 0) return services::Status();

    TArray<size_t, cpu> membersArr(memberCount);
    TArray<size_t, cpu> fillArr(nClusters);
    size_t * members = membersArr.get();
    size_t * fill    = fillArr.get();
    DAAL_CHECK_MALLOC(members && fill);
    for (size_t k = 0; k < nClusters; k++) fill[k] = offsets[k];
    for (size_t i = 0; i < nRows; i++)
    {
        if (labels[i] >= 0 && static_cast<size_t>(labels[i]) < nClusters) members[fill[labels[i]]++] = i;
    }

    // The members of each cluster are contiguous here, so a cluster's distance block is one batch
    // against a contiguous range of rows.
    TArray<algorithmFPType, cpu> sortedArr(memberCount * nCols);
    TArray<bool, cpu> zeroRowArr(memberCount);
    algorithmFPType * sorted = sortedArr.get();
    bool * zeroRow           = zeroRowArr.get();
    DAAL_CHECK_MALLOC(sorted && zeroRow);
    daal::threader_for(memberCount, 1, [&](size_t m) {
        const algorithmFPType * src = data + members[m] * nCols;
        bool isZero                 = true;
        for (size_t d = 0; d < nCols; d++)
        {
            sorted[m * nCols + d] = src[d];
            isZero                = isZero && src[d] == algorithmFPType(0);
        }
        zeroRow[m] = isZero;
    });

    services::Status status;
    auto nt = daal::internal::HomogenNumericTableCPU<algorithmFPType, cpu>::create(sorted, nCols, memberCount, &status);
    DAAL_CHECK_STATUS_VAR(status);
    services::SharedPtr<algorithms::internal::PairwiseDistances<algorithmFPType, cpu> > dist(
        createPairwiseDistances<algorithmFPType, cpu>(*nt, pairwiseDistance, minkowskiDegree));
    DAAL_CHECK_MALLOC(dist.get());
    DAAL_CHECK_STATUS(status, dist->init());

    // One task per (cluster, block of rowBlock members).
    const size_t rowBlock = 128;
    const size_t colBlock = 256;
    TArrayCalloc<size_t, cpu> taskOffsetsArr(nClusters + 1);
    size_t * taskOffsets = taskOffsetsArr.get();
    DAAL_CHECK_MALLOC(taskOffsets);
    for (size_t k = 0; k < nClusters; k++) taskOffsets[k + 1] = taskOffsets[k] + (offsets[k + 1] - offsets[k] + rowBlock - 1) / rowBlock;
    const size_t nTasks = taskOffsets[nClusters];
    TArray<size_t, cpu> taskClusterArr(nTasks);
    size_t * taskCluster = taskClusterArr.get();
    DAAL_CHECK_MALLOC(taskCluster);
    for (size_t k = 0; k < nClusters; k++)
    {
        for (size_t task = taskOffsets[k]; task < taskOffsets[k + 1]; task++) taskCluster[task] = k;
    }

    TArrayCalloc<double, cpu> scoresArr(memberCount);
    double * scores = scoresArr.get();
    DAAL_CHECK_MALLOC(scores);
    const bool isCosine = pairwiseDistance == algorithms::internal::PairwiseDistanceType::cosine;
    daal::TlsMem<algorithmFPType, cpu> tlsTile(rowBlock * colBlock);
    SafeStatus safeStat;

    daal::threader_for(nTasks, 1, [&](size_t task) {
        algorithmFPType * tile = tlsTile.local();
        DAAL_CHECK_MALLOC_THR(tile);

        const size_t k      = taskCluster[task];
        const size_t first  = offsets[k];
        const size_t last   = offsets[k + 1];
        const size_t iBegin = first + (task - taskOffsets[k]) * rowBlock;
        const size_t iCount = services::internal::min<cpu, size_t>(rowBlock, last - iBegin);

        for (size_t jBegin = first; jBegin < last; jBegin += colBlock)
        {
            const size_t jCount = services::internal::min<cpu, size_t>(colBlock, last - jBegin);
            services::Status s  = dist->computeBatch(sorted + iBegin * nCols, sorted + jBegin * nCols, iBegin, iCount, jBegin, jCount, tile);
            if (s) s = dist->finalize(iCount * jCount, tile);
            DAAL_CHECK_STATUS_THR(s);
            for (size_t i = 0; i < iCount; i++)
            {
                const size_t mi = iBegin + i;
                double score    = 0;
                for (size_t j = 0; j < jCount; j++)
                {
                    const size_t mj = jBegin + j;
                    double d        = tile[i * jCount + j];
                    if (mi == mj)
                    {
                        d = 0;
                    }
                    else if (isCosine)
                    {
                        // scikit-learn normalizes a zero row to the zero vector, i.e. distance 1.
                        d = (zeroRow[mi] || zeroRow[mj]) ? 1.0 : services::internal::max<cpu, double>(0.0, d);
                    }
                    score += d * weights[members[mj]];
                }
                scores[mi] += score;
            }
        }
    });
    DAAL_CHECK_SAFE_STATUS();

    for (size_t k = 0; k < nClusters; k++)
    {
        if (offsets[k] == offsets[k + 1]) continue;
        size_t best = offsets[k];
        for (size_t q = offsets[k] + 1; q < offsets[k + 1]; q++)
        {
            if (scores[q] < scores[best]) best = q;
        }
        const algorithmFPType * src = data + members[best] * nCols;
        for (size_t d = 0; d < nCols; d++) medoids[k * nCols + d] = src[d];
    }
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status HDBSCANCentersKernel<algorithmFPType, cpu>::compute(const NumericTable * ntData, const NumericTable * ntAssignments,
                                                                     const NumericTable * ntWeights, size_t nClusters, NumericTable * ntCentroids,
                                                                     NumericTable * ntMedoids,
                                                                     algorithms::internal::PairwiseDistanceType pairwiseDistance,
                                                                     double minkowskiDegree)
{
    const size_t nRows = ntData->getNumberOfRows();
    const size_t nCols = ntData->getNumberOfColumns();
    if (nClusters == 0 || nRows == 0) return services::Status();

    ReadRows<algorithmFPType, cpu> dataRows(const_cast<NumericTable *>(ntData), 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(dataRows);
    ReadRows<int, cpu> labelRows(const_cast<NumericTable *>(ntAssignments), 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(labelRows);
    ReadRows<algorithmFPType, cpu> weightRows(const_cast<NumericTable *>(ntWeights), 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(weightRows);
    const algorithmFPType * data    = dataRows.get();
    const int * labels              = labelRows.get();
    const algorithmFPType * weights = weightRows.get();

    if (ntCentroids)
    {
        WriteOnlyRows<algorithmFPType, cpu> centroidRows(ntCentroids, 0, nClusters);
        DAAL_CHECK_BLOCK_STATUS(centroidRows);
        services::Status status;
        DAAL_CHECK_STATUS(status, (computeCentroids<algorithmFPType, cpu>(data, labels, weights, nRows, nCols, nClusters, centroidRows.get())));
    }
    if (ntMedoids)
    {
        WriteOnlyRows<algorithmFPType, cpu> medoidRows(ntMedoids, 0, nClusters);
        DAAL_CHECK_BLOCK_STATUS(medoidRows);
        services::Status status;
        DAAL_CHECK_STATUS(status, (computeMedoids<algorithmFPType, cpu>(data, labels, weights, nRows, nCols, nClusters, pairwiseDistance,
                                                                        minkowskiDegree, medoidRows.get())));
    }
    return services::Status();
}

} // namespace internal
} // namespace hdbscan
} // namespace algorithms
} // namespace daal
