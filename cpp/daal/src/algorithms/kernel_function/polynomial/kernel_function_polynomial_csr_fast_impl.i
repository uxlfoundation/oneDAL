/* file: kernel_function_polynomial_csr_fast_impl.i */
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

#ifndef __KERNEL_FUNCTION_POLYNOMIAL_CSR_FAST_IMPL_I__
#define __KERNEL_FUNCTION_POLYNOMIAL_CSR_FAST_IMPL_I__

#include "src/algorithms/kernel_function/polynomial/kernel_function_types_polynomial.h"
#include "src/algorithms/kernel_function/kernel_function_csr_impl.i"

#include "src/threading/threading.h"
#include "src/externals/service_spblas.h"
#include "src/externals/service_blas.h"
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
services::Status KernelImplPolynomial<fastCSR, algorithmFPType, cpu>::computeInternalVectorVector(const NumericTable * a1, const NumericTable * a2,
                                                                                                  NumericTable * r, const KernelParameter * par)
{
    if (par->kernelType != KernelType::linear)
    {
        return services::ErrorMethodNotImplemented;
    }

    //prepareData
    ReadRowsCSR<algorithmFPType, cpu> mtA1(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a1)), par->rowIndexX, 1);
    DAAL_CHECK_BLOCK_STATUS(mtA1);
    const size_t * rowOffsetsA1 = mtA1.rows();

    ReadRowsCSR<algorithmFPType, cpu> mtA2(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a2)), par->rowIndexY, 1);
    DAAL_CHECK_BLOCK_STATUS(mtA2);
    const size_t * rowOffsetsA2 = mtA2.rows();

    WriteOnlyRows<algorithmFPType, cpu> mtR(r, par->rowIndexResult, 1);
    DAAL_CHECK_BLOCK_STATUS(mtR);
    algorithmFPType * dataR = mtR.get();

    //compute
    dataR[0] = computeDotProduct(rowOffsetsA1[0] - 1, rowOffsetsA1[1] - 1, mtA1.values(), mtA1.cols(), rowOffsetsA2[0] - 1, rowOffsetsA2[1] - 1,
                                 mtA2.values(), mtA2.cols());
    dataR[0] = dataR[0] * par->scale + par->shift;

    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status KernelImplPolynomial<fastCSR, algorithmFPType, cpu>::computeInternalMatrixVector(const NumericTable * a1, const NumericTable * a2,
                                                                                                  NumericTable * r, const KernelParameter * par)
{
    //prepareData
    const size_t nVectors1 = a1->getNumberOfRows();

    ReadRowsCSR<algorithmFPType, cpu> mtA1(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a1)), 0, nVectors1);
    DAAL_CHECK_BLOCK_STATUS(mtA1);
    const size_t * rowOffsetsA1 = mtA1.rows();

    ReadRowsCSR<algorithmFPType, cpu> mtA2(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a2)), par->rowIndexY, 1);
    DAAL_CHECK_BLOCK_STATUS(mtA2);
    const size_t * rowOffsetsA2 = mtA2.rows();

    WriteOnlyRows<algorithmFPType, cpu> mtR(r, 0, nVectors1);
    DAAL_CHECK_BLOCK_STATUS(mtR);
    algorithmFPType * dataR = mtR.get();

    algorithmFPType k = (algorithmFPType)(par->scale);
    algorithmFPType b = (algorithmFPType)(par->shift);

    //compute
    for (size_t i = 0; i < nVectors1; i++)
    {
        dataR[i] = computeDotProduct(rowOffsetsA1[i] - 1, rowOffsetsA1[i + 1] - 1, mtA1.values(), mtA1.cols(), rowOffsetsA2[0] - 1,
                                     rowOffsetsA2[1] - 1, mtA2.values(), mtA2.cols());
        dataR[i] = dataR[i] * k + b;
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
services::Status KernelImplPolynomial<fastCSR, algorithmFPType, cpu>::computeInternalMatrixMatrix(const NumericTable * a1, const NumericTable * a2,
                                                                                                  NumericTable * r, const KernelParameter * par)
{
    //prepareData
    const size_t nVectors1 = a1->getNumberOfRows();
    const size_t nVectors2 = a2->getNumberOfRows();
    const size_t nFeatures = a1->getNumberOfColumns();

    const algorithmFPType k    = (algorithmFPType)(par->scale);
    const algorithmFPType b    = (algorithmFPType)(par->shift);
    const size_t degree        = (par->kernelType == KernelType::sigmoid) ? 1 : static_cast<size_t>(par->degree);
    const algorithmFPType zero = algorithmFPType(0.0);
    const algorithmFPType one  = algorithmFPType(1.0);

    if (a1 == a2)
    {
        ReadRowsCSR<algorithmFPType, cpu> mtA1(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a1)), 0, nVectors1);
        DAAL_CHECK_BLOCK_STATUS(mtA1);
        const algorithmFPType * dataA1 = mtA1.values();
        const size_t * colIndicesA1    = mtA1.cols();
        const size_t * rowOffsetsA1    = mtA1.rows();

        WriteOnlyRows<algorithmFPType, cpu> mtR(r, 0, nVectors1);
        DAAL_CHECK_BLOCK_STATUS(mtR);
        algorithmFPType * dataR = mtR.get();

        SpBlasInst<algorithmFPType, cpu>::xsyrk_a_at(dataA1, colIndicesA1, rowOffsetsA1, nVectors1, a1->getNumberOfColumns(), dataR, nVectors2);

        if (k != one || b != zero)
        {
            daal::threader_for_optional(nVectors1, nVectors1, [=](size_t i) {
                PRAGMA_FORCE_SIMD
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j <= i; j++)
                {
                    const algorithmFPType factor = dataR[i * nVectors1 + j] * k + b;
                    dataR[i * nVectors1 + j]     = factor;
                    for (size_t k = 0; k < degree - 1; ++k)
                    {
                        dataR[i * nVectors1 + j] *= factor;
                    }
                }
                if (par->kernelType == KernelType::sigmoid)
                {
                    daal::internal::MathInst<algorithmFPType, cpu>::vTanh(i + 1, dataR + i * nVectors1, dataR + i * nVectors1);
                }
            });
        }

        daal::threader_for_optional(nVectors1, nVectors1, [=](size_t i) {
            PRAGMA_FORCE_SIMD
            PRAGMA_VECTOR_ALWAYS
            for (size_t j = i + 1; j < nVectors1; j++)
            {
                dataR[i * nVectors1 + j] = dataR[j * nVectors1 + i];
            }
        });
        return services::Status();
    }

    const size_t blockSize = 256;
    const size_t nBlocks1  = nVectors1 / blockSize + !!(nVectors1 % blockSize);
    const size_t nBlocks2  = nVectors2 / blockSize + !!(nVectors2 % blockSize);

    const bool isSOARes = r->getDataLayout() & NumericTableIface::soa;

    TlsMem<algorithmFPType, cpu> tlsMklBuff(blockSize * blockSize);
    SafeStatus safeStat;
    daal::conditional_threader_for(nBlocks1 > 2, nBlocks1, [&, isSOARes](const size_t iBlock1) {
        const size_t nRowsInBlock1 = (iBlock1 != nBlocks1 - 1) ? blockSize : nVectors1 - iBlock1 * blockSize;
        const size_t startRow1     = iBlock1 * blockSize;

        ReadRowsCSR<algorithmFPType, cpu> mtA1(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a1)), startRow1, nRowsInBlock1, true);
        DAAL_CHECK_BLOCK_STATUS_THR(mtA1);
        const algorithmFPType * dataA1 = mtA1.values();
        const size_t * colIndicesA1    = mtA1.cols();
        const size_t * rowOffsetsA1    = mtA1.rows();

        WriteOnlyRows<algorithmFPType, cpu> mtRRows;
        if (!isSOARes)
        {
            mtRRows.set(r, startRow1, nRowsInBlock1);
            DAAL_CHECK_MALLOC_THR(mtRRows.get());
        }
        daal::conditional_threader_for(nBlocks2 > 2, nBlocks2, [&, nVectors2, nBlocks2](const size_t iBlock2) {
            const size_t nRowsInBlock2 = (iBlock2 != nBlocks2 - 1) ? blockSize : nVectors2 - iBlock2 * blockSize;
            const size_t startRow2     = iBlock2 * blockSize;

            ReadRowsCSR<algorithmFPType, cpu> mtA2(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a2)), startRow2, nRowsInBlock2,
                                                   true);
            DAAL_CHECK_BLOCK_STATUS_THR(mtA2);
            const algorithmFPType * dataA2 = mtA2.values();
            const size_t * colIndicesA2    = mtA2.cols();
            const size_t * rowOffsetsA2    = mtA2.rows();

            if (!isSOARes)
            {
                algorithmFPType * const dataR = mtRRows.get();

                const size_t ldc = nVectors2;
                SpBlasInst<algorithmFPType, cpu>::xgemm_a_bt(dataA1, colIndicesA1, rowOffsetsA1, dataA2, colIndicesA2, rowOffsetsA2, nRowsInBlock1,
                                                             nRowsInBlock2, nFeatures, dataR + startRow2, ldc);

                if (k != one || b != zero)
                {
                    for (size_t i = 0; i < nRowsInBlock1; ++i)
                    {
                        for (size_t j = 0; j < nRowsInBlock2; ++j)
                        {
                            const algorithmFPType factor   = dataR[i * ldc + j + startRow2] * k + b;
                            dataR[i * ldc + j + startRow2] = factor;
                            for (size_t k = 0; k < degree - 1; ++k)
                            {
                                dataR[i * ldc + j + startRow2] *= factor;
                            }
                        }
                        if (par->kernelType == KernelType::sigmoid)
                        {
                            daal::internal::MathInst<algorithmFPType, cpu>::vTanh(nRowsInBlock2, dataR + i * ldc + startRow2,
                                                                                  dataR + i * ldc + startRow2);
                        }
                    }
                }
            }
            else
            {
                const size_t ldc                = blockSize;
                algorithmFPType * const mklBuff = tlsMklBuff.local();

                SpBlasInst<algorithmFPType, cpu>::xgemm_a_bt(dataA2, colIndicesA2, rowOffsetsA2, dataA1, colIndicesA1, rowOffsetsA1, nRowsInBlock2,
                                                             nRowsInBlock1, nFeatures, mklBuff, ldc);

                if (k != one || b != zero)
                {
                    for (size_t i = 0; i < nRowsInBlock2; ++i)
                    {
                        for (size_t j = 0; j < nRowsInBlock1; ++j)
                        {
                            const algorithmFPType factor = mklBuff[i * ldc + j] * k + b;
                            mklBuff[i * ldc + j]         = factor;
                            for (size_t k = 0; k < degree - 1; ++k)
                            {
                                mklBuff[i * ldc + j] *= factor;
                            }
                        }
                        if (par->kernelType == KernelType::sigmoid)
                        {
                            daal::internal::MathInst<algorithmFPType, cpu>::vTanh(nRowsInBlock1, mklBuff + i * ldc, mklBuff + i * ldc);
                        }
                    }
                }

                for (size_t j = 0; j < nRowsInBlock2; ++j)
                {
                    WriteOnlyColumns<algorithmFPType, cpu> mtRColumns(r, startRow2 + j, startRow1, nRowsInBlock1);
                    DAAL_CHECK_BLOCK_STATUS_THR(mtRColumns);
                    algorithmFPType * const dataRBlock   = mtRColumns.get();
                    algorithmFPType * const mklBuffBlock = &mklBuff[j * blockSize];
                    internal::Helper<algorithmFPType, cpu>::copy(dataRBlock, mklBuffBlock, nRowsInBlock1);
                }
            }
        });
    });

    return services::Status();
}

} // namespace internal
} // namespace polynomial
} // namespace kernel_function
} // namespace algorithms
} // namespace daal

#endif
