/* file: hdbscan_dist_matrix_impl.i */
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
#include "src/externals/service_blas.h"

namespace daal
{
namespace algorithms
{
namespace hdbscan
{
namespace internal
{

using daal::internal::BlasInst;
using daal::internal::CpuType;

template <typename algorithmFPType, CpuType cpu>
services::Status HDBSCANDistMatrixKernel<algorithmFPType, cpu>::compute(const algorithmFPType * data, algorithmFPType * distMatrix, size_t nRows,
                                                                        size_t nCols)
{
    const char transa           = 't';
    const char transb           = 'n';
    const DAAL_INT m            = static_cast<DAAL_INT>(nRows);
    const DAAL_INT n            = static_cast<DAAL_INT>(nRows);
    const DAAL_INT k            = static_cast<DAAL_INT>(nCols);
    const algorithmFPType alpha = algorithmFPType(1);
    const algorithmFPType beta  = algorithmFPType(0);
    const DAAL_INT lda          = static_cast<DAAL_INT>(nCols);
    const DAAL_INT ldb          = static_cast<DAAL_INT>(nCols);
    const DAAL_INT ldc          = static_cast<DAAL_INT>(nRows);

    BlasInst<algorithmFPType, cpu>::xxgemm(&transa, &transb, &m, &n, &k, &alpha, data, &lda, data, &ldb, &beta, distMatrix, &ldc);

    return services::Status();
}

} // namespace internal
} // namespace hdbscan
} // namespace algorithms
} // namespace daal
