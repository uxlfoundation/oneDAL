/* file: kmeans_csr_helper.h */
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

#ifndef _KMEANS_CSR_HELPER_H__
#define _KMEANS_CSR_HELPER_H__

#include "src/externals/service_blas.h"
#include "src/externals/service_memory.h"
#include "src/services/service_defines.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace internal
{
using namespace daal::internal;

/// Transposes the `nClusters x p` row-major centroids into `p x nClusters`.
///
/// @param centroids   Centroids, `nClusters x p`, row-major
/// @param nClusters   Number of centroids
/// @param p           Number of features
/// @param centroidsT  Output, `p x nClusters`, row-major
template <typename algorithmFPType, CpuType cpu>
void transposeCentroids(const algorithmFPType * const centroids, const size_t nClusters, const size_t p, algorithmFPType * const centroidsT)
{
    const DAAL_INT rows       = static_cast<DAAL_INT>(nClusters);
    const DAAL_INT cols       = static_cast<DAAL_INT>(p);
    const algorithmFPType one = algorithmFPType(1);
    BlasInst<algorithmFPType, cpu>::xomatcopy("R", "T", &rows, &cols, &one, centroids, &cols, centroidsT, &rows);
}

/// Inner products of the rows of a CSR block with every centroid, each summed over the
/// stored values of the row in order. `mkl_sparse_?_mm` runs in parallel under the TBB
/// layer even after `mkl_set_num_threads_local(1)`, so its sums vary from run to run.
///
/// @param nRows       Number of rows in the block
/// @param nClusters   Number of centroids
/// @param values      Stored values of the block
/// @param colIdx      One-based column index of each stored value
/// @param rowIdx      One-based offsets of the rows in `values`, `nRows + 1` entries
/// @param centroidsT  Transposed centroids, `p x nClusters`, row-major
/// @param products    Output, `nRows x nClusters`, row-major
// TODO: go back to mkl_sparse_?_mm once MKL keeps it sequential under
// mkl_set_num_threads_local(1) with the TBB threading layer.
template <typename algorithmFPType, CpuType cpu>
void csrRowsTimesCentroids(const size_t nRows, const size_t nClusters, const algorithmFPType * const values, const size_t * const colIdx,
                           const size_t * const rowIdx, const algorithmFPType * const centroidsT, algorithmFPType * const products)
{
    for (size_t i = 0; i < nRows; i++)
    {
        algorithmFPType * const out = products + i * nClusters;
        services::internal::service_memset_seq<algorithmFPType, cpu>(out, algorithmFPType(0), nClusters);
        for (size_t k = rowIdx[i] - 1; k < rowIdx[i + 1] - 1; k++)
        {
            const algorithmFPType value          = values[k];
            const algorithmFPType * const column = centroidsT + (colIdx[k] - 1) * nClusters;
            PRAGMA_OMP_SIMD
            for (size_t j = 0; j < nClusters; j++)
            {
                out[j] += value * column[j];
            }
        }
    }
}

} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal

#endif
