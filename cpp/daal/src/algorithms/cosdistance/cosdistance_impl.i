/* file: cosdistance_impl.i */
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
namespace cosine_distance
{
namespace internal
{
/// Compute the diagonal block of the cosine distance matrix
///
/// \tparam algorithmFPType  Data type to use in intermediate computations for the cosine distance algorithm, double or float
/// \tparam cpu              Type of CPU, e.g. daal::CpuType::sse2, daal::CpuType::avx2, etc.
/// \tparam upper            If true, compute the upper triangle of the block, otherwise compute the lower triangle
///
/// \param[in] blockSize     Size of the block to compute
/// \param[in] nColumns      Number of columns in the input matrix x
/// \param[in] x             Pointer to the input matrix x of size blockSize * nColumns
/// \param[out] block        Pointer to the output block of size blockSize * blockSize, where the computed cosine distance block will be stored
template <typename algorithmFPType, CpuType cpu, bool upper = true>
void computeDiagonalBlock(const size_t blockSize, const size_t nColumns, const algorithmFPType * x, algorithmFPType * block)
{
    /* calculate x^t * x */
    algorithmFPType diag[blockSizeDefault];
    constexpr algorithmFPType one(1.0);
    constexpr algorithmFPType zero(0.0);
    const algorithmFPType & alpha(one);
    const DAAL_INT bsz  = blockSize;
    const DAAL_INT ncol = nColumns;

    BlasInst<algorithmFPType, cpu>::xxgemm("T", "N", &bsz, &bsz, &ncol, &alpha, x, &ncol, x, &ncol, &zero, block, &bsz);

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
            PRAGMA_OMP_SIMD
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

} // namespace internal
} // namespace cosine_distance
} // namespace algorithms
} // namespace daal
