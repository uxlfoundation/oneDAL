/* file: cordistance_impl.i */
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

#include "src/services/service_defines.h"
using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace correlation_distance
{
namespace internal
{
/// Compute sum of each row of the input matrix x and store the result in sum
///
/// \tparam algorithmFPType  Data type to use in intermediate computations for the correlation distance algorithm, double or float
/// \tparam cpu              Type of CPU, e.g. daal::CpuType::sse2, daal::CpuType::avx2, etc.
///
/// \param[in] nRows            Number of rows in the input matrix x
/// \param[in] nColumns         Number of columns in the input matrix x
/// \param[in] x                Pointer to the input matrix x of size nRows * nColumns
/// \param[out] sum              Pointer to the output array of size nRows, where the sum of each row of x will be stored
template <typename algorithmFPType, CpuType cpu>
void sumByRows(const size_t nRows, const size_t nColumns, const algorithmFPType * x, algorithmFPType * sum)
{
    for (size_t i = 0; i < nRows; i++)
    {
        algorithmFPType s = (algorithmFPType)0.0;
        PRAGMA_OMP_SIMD(reduction(+:s))
        for (size_t j = 0; j < nColumns; j++)
        {
            s += x[i * nColumns + j];
        }
        sum[i] = s;
    }
}

/// Compute the diagonal block of the correlation distance matrix
///
/// \tparam algorithmFPType  Data type to use in intermediate computations for the correlation distance algorithm, double or float
/// \tparam cpu              Type of CPU, e.g. daal::CpuType::sse2, daal::CpuType::avx2, etc.
/// \tparam upper            If true, compute the upper triangle of the block, otherwise compute the lower triangle
///
/// \param[in] blockSize     Size of the block to compute
/// \param[in] nColumns      Number of columns in the input matrix x
/// \param[in] x             Pointer to the input matrix x of size blockSize * nColumns
/// \param[in,out] sum       On input, pointer to the array of size blockSize where the sum of each row of x is be stored.
///                          On output, it contains the diagonal elements of the block.
/// \param[out] block        Pointer to the output block of size blockSize * blockSize, where the computed correlation distance block will be stored
template <typename algorithmFPType, CpuType cpu, bool upper = true>
void computeDiagonalBlock(const size_t blockSize, const size_t nColumns, const algorithmFPType * x, algorithmFPType * sum, algorithmFPType * block)
{
    /* calculate sum^t * sum */
    const algorithmFPType one(1.0);
    const algorithmFPType zero(0.0);
    const DAAL_INT bsz = blockSize;
    const DAAL_INT ione = 1;

    BlasInst<algorithmFPType, cpu>::xxgemm("N", "T", &bsz, &bsz, &ione, &one, sum, &bsz, sum, &bsz, &zero, block, &bsz);

    /* calculate x * x^t - 1/p * sum^t * sum */
    const algorithmFPType beta = -one / (algorithmFPType)nColumns;
    const DAAL_INT ncol = nColumns;

    BlasInst<algorithmFPType, cpu>::xxgemm("T", "N", &bsz, &bsz, &ncol, &one, x, &ncol, x, &ncol, &beta, block, &bsz);

    algorithmFPType * diag = sum; // reuse sum array for diagonal elements

    for (size_t i = 0; i < blockSize; i++)
    {
        if (block[i * blockSize + i] > (algorithmFPType)0.0)
        {
            block[i * blockSize + i] = (algorithmFPType)1.0 / daal::internal::MathInst<algorithmFPType, cpu>::sSqrt(block[i * blockSize + i]);
        }
        diag[i] = block[i * blockSize + i];
    }

    if constexpr (upper)
    {
        for (size_t i = 0; i < blockSize; i++)
        {
            PRAGMA_FORCE_SIMD
            for (size_t j = i + 1; j < blockSize; j++)
            {
                block[i * blockSize + j] = one - block[i * blockSize + j] * diag[i] * diag[j];
            }
        }
    }
    else
    {
        for (size_t i = 0; i < blockSize; i++)
        {
            PRAGMA_VECTOR_ALWAYS
            for (size_t j = 0; j < i; j++)
            {
                block[i * blockSize + j] = one - block[i * blockSize + j] * diag[i] * diag[j];
            }
        }
    }
}

}
}
}
}
