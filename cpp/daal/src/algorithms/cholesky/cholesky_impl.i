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
template <typename algorithmFPType, CpuType cpu>
bool isFull(NumericTableIface::StorageLayout rLayout);

/**
 *  \brief Kernel for Cholesky calculation
 */
template <typename algorithmFPType, Method method, CpuType cpu>
Status CholeskyKernel<algorithmFPType, method, cpu>::compute(NumericTable * aTable, NumericTable * r, const daal::algorithms::Parameter * par)
{
    const size_t dim = aTable->getNumberOfColumns(); /* Dimension of input feature vectors */

    const NumericTableIface::StorageLayout iLayout = aTable->getDataLayout();
    const NumericTableIface::StorageLayout rLayout = r->getDataLayout();

    WriteOnlyRows<algorithmFPType, cpu> rowsR;
    WriteOnlyPacked<algorithmFPType, cpu> packedR;

    algorithmFPType * L = nullptr;
    if (isFull<algorithmFPType, cpu>(rLayout))
    {
        rowsR.set(*r, 0, dim);
        DAAL_CHECK_BLOCK_STATUS(rowsR);
        L = rowsR.get();
    }
    else
    {
        packedR.set(r);
        DAAL_CHECK_BLOCK_STATUS(packedR);
        L = packedR.get();
    }

    Status s;
    if (isFull<algorithmFPType, cpu>(iLayout))
    {
        ReadRows<algorithmFPType, cpu> rowsA(*aTable, 0, dim);
        DAAL_CHECK_BLOCK_STATUS(rowsA);
        s = copyMatrix(iLayout, rowsA.get(), rLayout, L, dim);
    }
    else
    {
        ReadPacked<algorithmFPType, cpu> packedA(*aTable);
        DAAL_CHECK_BLOCK_STATUS(packedA);
        s = copyMatrix(iLayout, packedA.get(), rLayout, L, dim);
    }
    return s.ok() ? performCholesky(rLayout, L, dim) : s;
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status CholeskyKernel<algorithmFPType, method, cpu>::copyMatrix(NumericTableIface::StorageLayout iLayout, const algorithmFPType * pA,
                                                                NumericTableIface::StorageLayout rLayout, algorithmFPType * pL, size_t dim) const
{
    if (isFull<algorithmFPType, cpu>(rLayout))
    {
        if (!copyToFullMatrix(iLayout, pA, pL, dim))
            return Status(ErrorIncorrectTypeOfInputNumericTable);
        else
            return Status();
    }
    else
    {
        return copyToLowerTrianglePacked(iLayout, pA, pL, dim);
    }
}

template <typename algorithmFPType, Method method, CpuType cpu>
Status CholeskyKernel<algorithmFPType, method, cpu>::performCholesky(NumericTableIface::StorageLayout rLayout, algorithmFPType * pL, size_t dim)
{
    DAAL_INT info;
    DAAL_INT dims = static_cast<DAAL_INT>(dim);
    char uplo     = 'U';

    if (isFull<algorithmFPType, cpu>(rLayout))
    {
        LapackInst<algorithmFPType, cpu>::xpotrf(&uplo, &dims, pL, &dims, &info);
    }
    else if (rLayout == NumericTableIface::lowerPackedTriangularMatrix)
    {
        LapackInst<algorithmFPType, cpu>::xpptrf(&uplo, &dims, pL, &info);
    }
    else
    {
        return Status(ErrorIncorrectTypeOfOutputNumericTable);
    }

    if (info > 0) return Status(Error::create(services::ErrorInputMatrixHasNonPositiveMinor, services::Minor, (int)info));

    return info < 0 ? Status(services::ErrorCholeskyInternal) : Status();
}

template <typename algorithmFPType, CpuType cpu>
bool isFull(NumericTableIface::StorageLayout layout)
{
    int layoutInt = int(layout);
    return !((packed_mask & layoutInt) && (NumericTableIface::csrArray != layoutInt));
}

template <typename algorithmFPType, Method method, CpuType cpu>
bool CholeskyKernel<algorithmFPType, method, cpu>::copyToFullMatrix(NumericTableIface::StorageLayout iLayout, const algorithmFPType * pA,
                                                                    algorithmFPType * pL, size_t dim) const
{
    const size_t blockSize = 256;
    const size_t n         = dim;
    size_t nBlocks         = n / blockSize;
    if (nBlocks * blockSize < n)
    {
        nBlocks++;
    }

    if (isFull<algorithmFPType, cpu>(iLayout))
    {
        threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
            size_t endBlock = (iBlock + 1) * blockSize;
            endBlock        = endBlock > n ? n : endBlock;

            for (size_t i = iBlock * blockSize; i < endBlock; i++)
            {
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j <= i; j++)
                {
                    pL[i * dim + j] = pA[i * dim + j];
                }
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = (i + 1); j < dim; j++)
                {
                    pL[i * dim + j] = algorithmFPType(0);
                }
            }
        });
    }
    else if (iLayout == NumericTableIface::lowerPackedSymmetricMatrix)
    {
        threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
            size_t endBlock = (iBlock + 1) * blockSize;
            endBlock        = endBlock > n ? n : endBlock;

            for (size_t i = iBlock * blockSize; i < endBlock; i++)
            {
                const size_t ind = (i + 1) * i / 2;

                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j <= i; j++)
                {
                    pL[i * dim + j] = pA[ind + j];
                }
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = (i + 1); j < dim; j++)
                {
                    pL[i * dim + j] = algorithmFPType(0);
                }
            }
        });
    }
    else if (iLayout == NumericTableIface::upperPackedSymmetricMatrix)
    {
        threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
            size_t endBlock = (iBlock + 1) * blockSize;
            endBlock        = endBlock > n ? n : endBlock;

            for (size_t j = iBlock * blockSize; j < endBlock; j++)
            {
                const size_t ind = (2 * dim - j + 1) * j / 2;

                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t i = 0; i < j; i++)
                {
                    pL[i * dim + j] = algorithmFPType(0);
                }
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t i = j; i < dim; i++)
                {
                    pL[i * dim + j] = pA[ind + i - j];
                }
            }
        });
    }
    else
    {
        return false;
    }
    return true;
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status CholeskyKernel<algorithmFPType, method, cpu>::copyToLowerTrianglePacked(NumericTableIface::StorageLayout iLayout,
                                                                                         const algorithmFPType * pA, algorithmFPType * pL,
                                                                                         size_t dim) const
{
    Status status;
    const size_t blockSize = 512;
    const size_t n         = dim;
    size_t nBlocks         = n / blockSize;
    if (nBlocks * blockSize < n)
    {
        nBlocks++;
    }

    if (isFull<algorithmFPType, cpu>(iLayout))
    {
        threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
            size_t endBlock = (iBlock + 1) * blockSize;
            endBlock        = endBlock > n ? n : endBlock;

            for (size_t i = iBlock * blockSize; i < endBlock; i++)
            {
                const size_t ind = (i + 1) * i / 2;

                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j <= i; j++)
                {
                    pL[ind + j] = pA[i * dim + j];
                }
            }
        });
    }
    else if (iLayout == NumericTableIface::lowerPackedSymmetricMatrix)
    {
        size_t size = (dim * (dim + 1) / 2) * sizeof(algorithmFPType);
        int result  = 0;
        result      = services::internal::daal_memcpy_s(pL, size, pA, size);
        if (result)
        {
            status |= Status(ErrorMemoryCopyFailedInternal);
        }
    }
    else if (iLayout == NumericTableIface::upperPackedSymmetricMatrix)
    {
        threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
            size_t endBlock = (iBlock + 1) * blockSize;
            endBlock        = endBlock > n ? n : endBlock;

            for (size_t j = iBlock * blockSize; j < endBlock; j++)
            {
                const size_t ind = (j + 1) * j / 2;

                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t i = 0; i <= j; i++)
                {
                    pL[ind + i] = pA[(dim * i - i * (i - 1) / 2 - i) + j];
                }
            }
        });
    }
    else
    {
        status |= Status(ErrorIncorrectTypeOfInputNumericTable);
    }
    return status;
}

} // namespace internal
} // namespace cholesky
} // namespace algorithms
} // namespace daal
