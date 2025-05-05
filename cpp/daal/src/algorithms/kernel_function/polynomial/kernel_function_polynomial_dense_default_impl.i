/* file: kernel_function_polynomial_dense_default_impl.i */
/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef __KERNEL_FUNCTION_POLYNOMIAL_DENSE_DEFAULT_IMPL_I__
#define __KERNEL_FUNCTION_POLYNOMIAL_DENSE_DEFAULT_IMPL_I__

#include "src/algorithms/kernel_function/polynomial/kernel_function_types_polynomial.h"

#include "src/externals/service_blas.h"
#include "src/externals/service_stat.h"
#include "src/threading/threading.h"
#include "src/algorithms/service_error_handling.h"
#include "src/externals/service_math.h"

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace polynomial
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
services::Status KernelImplPolynomial<defaultDense, algorithmFPType, cpu>::computeInternalVectorVector(const NumericTable * a1,
                                                                                                       const NumericTable * a2, NumericTable * r,
                                                                                                       const KernelParameter * par)
{
    if (par->kernelType != KernelType::linear)
    {
        return services::ErrorMethodNotImplemented;
    }

    //prepareData
    const size_t nFeatures = a1->getNumberOfColumns();

    ReadRows<algorithmFPType, cpu> mtA1(*const_cast<NumericTable *>(a1), par->rowIndexX, 1);
    DAAL_CHECK_BLOCK_STATUS(mtA1);
    const algorithmFPType * dataA1 = mtA1.get();

    ReadRows<algorithmFPType, cpu> mtA2(*const_cast<NumericTable *>(a2), par->rowIndexY, 1);
    DAAL_CHECK_BLOCK_STATUS(mtA2);
    const algorithmFPType * dataA2 = mtA2.get();

    WriteOnlyRows<algorithmFPType, cpu> mtR(r, par->rowIndexResult, 1);
    DAAL_CHECK_BLOCK_STATUS(mtR);
    algorithmFPType * dataR = mtR.get();

    //compute
    dataR[0] = 0.0;
    PRAGMA_FORCE_SIMD
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < nFeatures; i++)
    {
        dataR[0] += dataA1[i] * dataA2[i];
    }
    dataR[0] = dataR[0] * par->scale + par->shift;

    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status KernelImplPolynomial<defaultDense, algorithmFPType, cpu>::computeInternalMatrixVector(const NumericTable * a1,
                                                                                                       const NumericTable * a2, NumericTable * r,
                                                                                                       const KernelParameter * par)
{
    //prepareData
    const size_t nVectors1 = a1->getNumberOfRows();
    const size_t nFeatures = a1->getNumberOfColumns();

    ReadRows<algorithmFPType, cpu> mtA1(*const_cast<NumericTable *>(a1), 0, nVectors1);
    DAAL_CHECK_BLOCK_STATUS(mtA1);
    const algorithmFPType * dataA1 = mtA1.get();

    ReadRows<algorithmFPType, cpu> mtA2(*const_cast<NumericTable *>(a2), par->rowIndexY, 1);
    DAAL_CHECK_BLOCK_STATUS(mtA2);
    const algorithmFPType * dataA2 = mtA2.get();

    WriteOnlyRows<algorithmFPType, cpu> mtR(r, 0, nVectors1);
    DAAL_CHECK_BLOCK_STATUS(mtR);
    algorithmFPType * dataR = mtR.get();

    //compute
    const algorithmFPType k = (algorithmFPType)(par->scale);
    const algorithmFPType b = (algorithmFPType)(par->shift);

    services::internal::service_memset_seq<algorithmFPType, cpu>(dataR, b, nVectors1);
    for (size_t i = 0; i < nVectors1; i++)
    {
        PRAGMA_FORCE_SIMD
        PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < nFeatures; j++)
        {
            dataR[i] += dataA1[i * nFeatures + j] * dataA2[j];
        }
        dataR[i] = k * dataR[i];
    }

    if (par->kernelType == KernelType::sigmoid)
    {
        daal::internal::MathInst<algorithmFPType, cpu>::vTanh(nVectors1, dataR, dataR);
    }
    if (par->kernelType == KernelType::polynomial)
    {
        daal::internal::MathInst<algorithmFPType, cpu>::vPowx(nVectors1, dataR, par->degree, dataR);
    }

    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status KernelImplPolynomial<defaultDense, algorithmFPType, cpu>::computeInternalMatrixMatrix(const NumericTable * a1,
                                                                                                       const NumericTable * a2, NumericTable * r,
                                                                                                       const KernelParameter * par)
{
    SafeStatus safeStat;

    char trans   = 'T';
    char notrans = 'N';

    const size_t nFeatures = a1->getNumberOfColumns();
    const size_t nVectors1 = a1->getNumberOfRows();
    const size_t nVectors2 = a2->getNumberOfRows();

    algorithmFPType alpha = (algorithmFPType)(par->scale);
    algorithmFPType beta  = 0.0;
    algorithmFPType one   = 1.0;
    algorithmFPType shift = (algorithmFPType)(par->shift);
    const size_t degree   = (par->kernelType == KernelType::sigmoid) ? 1 : static_cast<size_t>(par->degree);

    const bool isSOARes = r->getDataLayout() & NumericTableIface::soa;

    DAAL_OVERFLOW_CHECK_BY_ADDING(size_t, nVectors1, nVectors2);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nVectors1 + nVectors2, sizeof(algorithmFPType));

    const size_t blockSize = 128;
    const size_t nBlocks1  = nVectors1 / blockSize + !!(nVectors1 % blockSize);
    const size_t nBlocks2  = nVectors2 / blockSize + !!(nVectors2 % blockSize);

    TlsMem<algorithmFPType, cpu> tlsMklBuff(blockSize * blockSize);

    daal::conditional_threader_for((nBlocks1 > 2), nBlocks1, [&, isSOARes](const size_t iBlock1) {
        DAAL_INT nRowsInBlock1 = (iBlock1 != nBlocks1 - 1) ? blockSize : nVectors1 - iBlock1 * blockSize;
        DAAL_INT startRow1     = iBlock1 * blockSize;

        ReadRows<algorithmFPType, cpu> mtA1(*const_cast<NumericTable *>(a1), startRow1, nRowsInBlock1);
        DAAL_CHECK_BLOCK_STATUS_THR(mtA1);
        const algorithmFPType * const dataA1 = const_cast<algorithmFPType *>(mtA1.get());

        WriteOnlyRows<algorithmFPType, cpu> mtRRows;
        if (!isSOARes)
        {
            mtRRows.set(r, startRow1, nRowsInBlock1);
            DAAL_CHECK_MALLOC_THR(mtRRows.get());
        }

        daal::conditional_threader_for((nBlocks2 > 2), nBlocks2, [&, nVectors2, nBlocks2](const size_t iBlock2) {
            DAAL_INT nRowsInBlock2 = (iBlock2 != nBlocks2 - 1) ? blockSize : nVectors2 - iBlock2 * blockSize;
            DAAL_INT startRow2     = iBlock2 * blockSize;

            ReadRows<algorithmFPType, cpu> mtA2(*const_cast<NumericTable *>(a2), startRow2, nRowsInBlock2);
            DAAL_CHECK_BLOCK_STATUS_THR(mtA2);
            const algorithmFPType * const dataA2 = const_cast<algorithmFPType *>(mtA2.get());

            if (!isSOARes)
            {
                algorithmFPType * const dataR = mtRRows.get() + startRow2;
                BlasInst<algorithmFPType, cpu>::xxgemm(&trans, &notrans, &nRowsInBlock2, &nRowsInBlock1, (DAAL_INT *)&nFeatures, &alpha, dataA2,
                                                       (DAAL_INT *)&nFeatures, dataA1, (DAAL_INT *)&nFeatures, &beta, dataR, (DAAL_INT *)&nVectors2);

                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t i = 0; i < nRowsInBlock1; ++i)
                {
                    for (size_t j = 0; j < nRowsInBlock2; ++j)
                    {
                        if (degree != 0)
                        {
                            dataR[i * nVectors2 + j] += shift;
                            const algorithmFPType factor = dataR[i * nVectors2 + j];
                            for (size_t k = 0; k < degree - 1; ++k)
                            {
                                dataR[i * nVectors2 + j] *= factor;
                            }
                        }
                        else
                        {
                            dataR[i * nVectors2 + j] = one;
                        }
                    }
                    if (par->kernelType == KernelType::sigmoid)
                    {
                        daal::internal::MathInst<algorithmFPType, cpu>::vTanh(nRowsInBlock2, dataR + i * nVectors2, dataR + i * nVectors2);
                    }
                }
            }
            else
            {
                algorithmFPType * const mklBuff = tlsMklBuff.local();
                DAAL_CHECK_MALLOC_THR(mklBuff);
                DAAL_INT ldc2 = blockSize;

                BlasInst<algorithmFPType, cpu>::xxgemm(&trans, &notrans, &nRowsInBlock1, &nRowsInBlock2, (DAAL_INT *)&nFeatures, &alpha, dataA1,
                                                       (DAAL_INT *)&nFeatures, dataA2, (DAAL_INT *)&nFeatures, &beta, mklBuff, &ldc2);

                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t i = 0; i < blockSize * blockSize; ++i)
                {
                    if (degree != 0)
                    {
                        mklBuff[i] += shift;
                        const algorithmFPType factor = mklBuff[i];
                        for (size_t k = 0; k < degree - 1; ++k)
                        {
                            mklBuff[i] *= factor;
                        }
                    }
                    else
                    {
                        mklBuff[i] = one;
                    }
                }

                if (par->kernelType == KernelType::sigmoid)
                {
                    daal::internal::MathInst<algorithmFPType, cpu>::vTanh(blockSize * blockSize, mklBuff, mklBuff);
                }

                for (size_t i = 0; i < nRowsInBlock2; ++i)
                {
                    WriteOnlyColumns<algorithmFPType, cpu> mtrColumns(r, startRow2 + i, startRow1, nRowsInBlock1);
                    DAAL_CHECK_BLOCK_STATUS_THR(mtrColumns);
                    Helper<algorithmFPType, cpu>::copy(mtrColumns.get(), mklBuff + i * ldc2, nRowsInBlock1);
                }
            }
        });
    });

    return safeStat.detach();
}

} // namespace internal
} // namespace polynomial
} // namespace kernel_function
} // namespace algorithms
} // namespace daal

#endif
