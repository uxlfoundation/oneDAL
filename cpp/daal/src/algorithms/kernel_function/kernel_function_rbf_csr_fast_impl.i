/* file: kernel_function_rbf_csr_fast_impl.i */
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
//  RBF kernel functions implementation
//--
*/

#ifndef __KERNEL_FUNCTION_RBF_CSR_FAST_IMPL_I__
#define __KERNEL_FUNCTION_RBF_CSR_FAST_IMPL_I__

#include "src/algorithms/kernel_function/kernel_function_types_rbf.h"
#include "src/data_management/service_numeric_table.h"
#include "src/threading/threading.h"
#include "src/externals/service_spblas.h"
#include "src/algorithms/kernel_function/kernel_function_rbf_helper.h"
#include "src/algorithms/kernel_function/kernel_function_csr_impl.i"
#include "src/algorithms/service_error_handling.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace rbf
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
services::Status KernelImplRBF<fastCSR, algorithmFPType, cpu>::computeInternalVectorVector(const NumericTable * a1, const NumericTable * a2,
                                                                                           NumericTable * r, const KernelParameter * par)
{
    //prepareData
    ReadRowsCSR<algorithmFPType, cpu> mtA1(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a1)), par->rowIndexX, 1);
    DAAL_CHECK_BLOCK_STATUS(mtA1);
    const algorithmFPType * dataA1 = mtA1.values();
    const size_t * rowOffsetsA1    = mtA1.rows();

    ReadRowsCSR<algorithmFPType, cpu> mtA2(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a2)), par->rowIndexY, 1);
    DAAL_CHECK_BLOCK_STATUS(mtA2);
    const algorithmFPType * dataA2 = mtA2.values();
    const size_t * rowOffsetsA2    = mtA2.rows();

    WriteOnlyRows<algorithmFPType, cpu> mtR(r, par->rowIndexResult, 1);
    DAAL_CHECK_BLOCK_STATUS(mtR);

    //compute
    const algorithmFPType coeff = (algorithmFPType)(-0.5 / (par->sigma * par->sigma));
    const size_t startIndex1    = rowOffsetsA1[0] - 1;
    const size_t startIndex2    = rowOffsetsA2[0] - 1;
    const size_t endIndex1      = rowOffsetsA1[1] - 1;
    const size_t endIndex2      = rowOffsetsA2[1] - 1;
    algorithmFPType factor      = computeDotProduct(startIndex1, endIndex1, dataA1, mtA1.cols(), startIndex2, endIndex2, dataA2, mtA2.cols());
    factor *= -2.0;

    for (size_t index = startIndex1; index < endIndex1; index++)
    {
        factor += dataA1[index] * dataA1[index];
    }
    for (size_t index = startIndex2; index < endIndex2; index++)
    {
        factor += dataA2[index] * dataA2[index];
    }
    factor *= coeff;
    daal::internal::MathInst<algorithmFPType, cpu>::vExp(1, &factor, mtR.get());

    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status KernelImplRBF<fastCSR, algorithmFPType, cpu>::computeInternalMatrixVector(const NumericTable * a1, const NumericTable * a2,
                                                                                           NumericTable * r, const KernelParameter * par)
{
    //prepareData
    const size_t nVectors1 = a1->getNumberOfRows();

    ReadRowsCSR<algorithmFPType, cpu> mtA1(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a1)), 0, nVectors1);
    DAAL_CHECK_BLOCK_STATUS(mtA1);
    const algorithmFPType * dataA1 = mtA1.values();
    const size_t * rowOffsetsA1    = mtA1.rows();

    ReadRowsCSR<algorithmFPType, cpu> mtA2(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a2)), par->rowIndexY, 1);
    DAAL_CHECK_BLOCK_STATUS(mtA2);
    const algorithmFPType * dataA2 = mtA2.values();
    const size_t * rowOffsetsA2    = mtA2.rows();

    WriteOnlyRows<algorithmFPType, cpu> mtR(r, 0, nVectors1);
    DAAL_CHECK_BLOCK_STATUS(mtR);
    algorithmFPType * dataR = mtR.get();

    //compute
    const algorithmFPType coeff = (algorithmFPType)(-0.5 / (par->sigma * par->sigma));
    const size_t startIndex2    = rowOffsetsA2[0] - 1;
    const size_t endIndex2      = rowOffsetsA2[1] - 1;

    algorithmFPType factor = 0.0;
    for (size_t index = startIndex2; index < endIndex2; index++)
    {
        factor += dataA2[index] * dataA2[index];
    }
    for (size_t i = 0; i < nVectors1; i++)
    {
        size_t startIndex1 = rowOffsetsA1[i] - 1;
        size_t endIndex1   = rowOffsetsA1[i + 1] - 1;
        dataR[i]           = computeDotProduct(startIndex1, endIndex1, dataA1, mtA1.cols(), startIndex2, endIndex2, dataA2, mtA2.cols());
        dataR[i]           = -2.0 * dataR[i] + factor;
        for (size_t index = startIndex1; index < endIndex1; index++)
        {
            dataR[i] += dataA1[index] * dataA1[index];
        }
        dataR[i] *= coeff;

        // make all values less than threshold as threshold value
        // to fix slow work on vExp on large negative inputs
        if (dataR[i] < MathInst<algorithmFPType, cpu>::vExpThreshold())
        {
            dataR[i] = MathInst<algorithmFPType, cpu>::vExpThreshold();
        }
    }
    daal::internal::MathInst<algorithmFPType, cpu>::vExp(nVectors1, dataR, dataR);

    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status KernelImplRBF<fastCSR, algorithmFPType, cpu>::computeInternalMatrixMatrix(const NumericTable * a1, const NumericTable * a2,
                                                                                           NumericTable * r, const KernelParameter * par)
{
    //prepareData
    const size_t nVectors1 = a1->getNumberOfRows();
    const size_t nVectors2 = a2->getNumberOfRows();
    const size_t nFeatures = a1->getNumberOfColumns();

    //compute
    const algorithmFPType coeff  = (algorithmFPType)(-0.5 / (par->sigma * par->sigma));
    const algorithmFPType zero   = 0.0;
    const algorithmFPType negTwo = -2.0;

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

        daal::threader_for_optional(nVectors1, nVectors1, [=](size_t i) {
            for (size_t k = 0; k < i; k++)
            {
                dataR[i * nVectors1 + k] = coeff * (dataR[i * nVectors1 + i] + dataR[k * nVectors1 + k] + negTwo * dataR[i * nVectors1 + k]);
            }
        });
        daal::threader_for_optional(nVectors1, nVectors1, [=](size_t i) {
            dataR[i * nVectors1 + i] = zero;
            daal::internal::MathInst<algorithmFPType, cpu>::vExp(i + 1, dataR + i * nVectors1, dataR + i * nVectors1);
        });
        daal::threader_for_optional(nVectors1, nVectors1, [=](size_t i) {
            for (size_t k = i + 1; k < nVectors1; k++)
            {
                dataR[i * nVectors1 + k] = dataR[k * nVectors1 + i];
            }
        });
        return services::Status();
    }

    const bool isSOARes = r->getDataLayout() & NumericTableIface::soa;

    DAAL_OVERFLOW_CHECK_BY_ADDING(size_t, nVectors1, nVectors2);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nVectors1 + nVectors2, sizeof(algorithmFPType));

    const size_t blockSize = 256;
    const size_t nBlocks1  = nVectors1 / blockSize + !!(nVectors1 % blockSize);
    const size_t nBlocks2  = nVectors2 / blockSize + !!(nVectors2 % blockSize);

    const algorithmFPType expExpThreshold = MathInst<algorithmFPType, cpu>::vExpThreshold();

    SafeStatus safeStat;
    daal::tls<KernelRBFTask<algorithmFPType, cpu> *> tslTask([=, &safeStat]() {
        auto tlsData = KernelRBFTask<algorithmFPType, cpu>::create(blockSize, false);
        if (!tlsData)
        {
            safeStat.add(services::ErrorMemoryAllocationFailed);
        }
        return tlsData;
    });

    daal::conditional_threader_for((nBlocks1 > 2), nBlocks1, [&, isSOARes](const size_t iBlock1) {
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
        daal::conditional_threader_for((nBlocks2 > 2), nBlocks2, [&, nVectors2, nBlocks2](const size_t iBlock2) {
            const size_t nRowsInBlock2 = (iBlock2 != nBlocks2 - 1) ? blockSize : nVectors2 - iBlock2 * blockSize;
            const size_t startRow2     = iBlock2 * blockSize;

            KernelRBFTask<algorithmFPType, cpu> * const tlsLocal = tslTask.local();

            algorithmFPType * const mklBuff   = tlsLocal->mklBuff;
            algorithmFPType * const sqrDataA1 = tlsLocal->sqrDataA1;
            algorithmFPType * const sqrDataA2 = tlsLocal->sqrDataA2;

            for (size_t i = 0; i < nRowsInBlock1; ++i)
            {
                sqrDataA1[i] = zero;
                for (size_t j = rowOffsetsA1[i] - 1; j < rowOffsetsA1[i + 1] - 1; j++)
                {
                    sqrDataA1[i] += dataA1[j] * dataA1[j];
                }
            }

            ReadRowsCSR<algorithmFPType, cpu> mtA2(dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(a2)), startRow2, nRowsInBlock2,
                                                   true);
            DAAL_CHECK_BLOCK_STATUS_THR(mtA2);
            const algorithmFPType * dataA2 = mtA2.values();
            const size_t * colIndicesA2    = mtA2.cols();
            const size_t * rowOffsetsA2    = mtA2.rows();

            for (size_t i = 0; i < nRowsInBlock2; ++i)
            {
                sqrDataA2[i] = zero;
                for (size_t j = rowOffsetsA2[i] - 1; j < rowOffsetsA2[i + 1] - 1; j++)
                {
                    sqrDataA2[i] += dataA2[j] * dataA2[j];
                }
            }

            if (!isSOARes)
            {
                algorithmFPType * const dataR = mtRRows.get();
                SpBlasInst<algorithmFPType, cpu>::xgemm_a_bt(dataA1, colIndicesA1, rowOffsetsA1, dataA2, colIndicesA2, rowOffsetsA2, nRowsInBlock1,
                                                             nRowsInBlock2, nFeatures, mklBuff, blockSize);

                for (size_t i = 0; i < nRowsInBlock1; ++i)
                {
                    const algorithmFPType sqrA1i         = sqrDataA1[i];
                    algorithmFPType * const dataRBlock   = &dataR[i * nVectors2 + startRow2];
                    algorithmFPType * const mklBuffBlock = &mklBuff[i * blockSize];
                    HelperKernelRBF<algorithmFPType, cpu>::postGemmPart(mklBuffBlock, sqrDataA2, sqrA1i, coeff, expExpThreshold, nRowsInBlock2,
                                                                        dataRBlock);
                }
            }
            else
            {
                SpBlasInst<algorithmFPType, cpu>::xgemm_a_bt(dataA2, colIndicesA2, rowOffsetsA2, dataA1, colIndicesA1, rowOffsetsA1, nRowsInBlock2,
                                                             nRowsInBlock1, nFeatures, mklBuff, blockSize);

                for (size_t j = 0; j < nRowsInBlock2; ++j)
                {
                    const algorithmFPType sqrA2i = sqrDataA2[j];
                    WriteOnlyColumns<algorithmFPType, cpu> mtRColumns(r, startRow2 + j, startRow1, nRowsInBlock1);
                    DAAL_CHECK_BLOCK_STATUS_THR(mtRColumns);
                    algorithmFPType * const dataRBlock   = mtRColumns.get();
                    algorithmFPType * const mklBuffBlock = &mklBuff[j * blockSize];
                    HelperKernelRBF<algorithmFPType, cpu>::postGemmPart(mklBuffBlock, sqrDataA1, sqrA2i, coeff, expExpThreshold, nRowsInBlock1,
                                                                        dataRBlock);
                }
            }
        });
    });

    tslTask.reduce([](KernelRBFTask<algorithmFPType, cpu> * tlsLocal) { delete tlsLocal; });

    return services::Status();
}

} // namespace internal
} // namespace rbf
} // namespace kernel_function
} // namespace algorithms
} // namespace daal

#endif
