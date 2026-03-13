/* file: cholesky_impl.i */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

/*
//++
//  Implementation of cholesky algorithm
//--
*/

#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_lapack.h"

using namespace daal::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace cholesky
{
namespace internal
{

/**
 *  \brief Kernel for Cholesky calculation
 */
template <typename algorithmFPType, Method method, CpuType cpu>
Status CholeskyKernel<algorithmFPType, method, cpu>::compute(NumericTable * aTable, NumericTable * r, const daal::algorithms::Parameter * par)
{
    const size_t dim = aTable->getNumberOfColumns(); /* Dimension of input feature vectors */

    WriteOnlyRows<algorithmFPType, cpu> rowsR;

    algorithmFPType * L = nullptr;

    rowsR.set(*r, 0, dim);
    DAAL_CHECK_BLOCK_STATUS(rowsR);
    L = rowsR.get();

    Status s;

    ReadRows<algorithmFPType, cpu> rowsA(*aTable, 0, dim);
    DAAL_CHECK_BLOCK_STATUS(rowsA);
    s = copyMatrix(rowsA.get(), L, dim);

    return s.ok() ? performCholesky(L, dim) : s;
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status CholeskyKernel<algorithmFPType, method, cpu>::copyMatrix(const algorithmFPType * pA, algorithmFPType * pL, size_t dim) const
{
    const size_t blockSize = 256;
    const size_t n         = dim;
    size_t nBlocks         = n / blockSize;
    if (nBlocks * blockSize < n)
    {
        nBlocks++;
    }

    threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
        size_t endBlock = (iBlock + 1) * blockSize;
        endBlock        = endBlock > n ? n : endBlock;

        for (size_t i = iBlock * blockSize; i < endBlock; i++)
        {
            PRAGMA_OMP_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t j = 0; j <= i; j++)
            {
                pL[i * dim + j] = pA[i * dim + j];
            }
            PRAGMA_OMP_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t j = (i + 1); j < dim; j++)
            {
                pL[i * dim + j] = algorithmFPType(0);
            }
        }
    });

    return Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status CholeskyKernel<algorithmFPType, method, cpu>::performCholesky(algorithmFPType * pL, size_t dim)
{
    DAAL_INT info;
    DAAL_INT dims = static_cast<DAAL_INT>(dim);
    char uplo     = 'U';

    LapackInst<algorithmFPType, cpu>::xpotrf(&uplo, &dims, pL, &dims, &info);

    if (info > 0) return Status(Error::create(services::ErrorInputMatrixHasNonPositiveMinor, services::Minor, (int)info));

    return info < 0 ? Status(services::ErrorCholeskyInternal) : Status();
}

} // namespace internal
} // namespace cholesky
} // namespace algorithms
} // namespace daal
