/* file: kernel_function_rbf_dense_default_impl.i */
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

#ifndef __KERNEL_FUNCTION_RBF_DENSE_DEFAULT_IMPL_I__
#define __KERNEL_FUNCTION_RBF_DENSE_DEFAULT_IMPL_I__

#include "algorithms/kernel_function/kernel_function_types_rbf.h"
#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_math.h"
#include "src/externals/service_blas.h"
#include "src/externals/service_profiler.h"
#include "src/threading/threading.h"
#include "src/algorithms/kernel_function/kernel_function_rbf_helper.h"

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
services::Status KernelImplRBF<defaultDense, algorithmFPType, cpu>::computeInternalVectorVector(const NumericTable * a1, const NumericTable * a2,
                                                                                                NumericTable * r, const KernelParameter * par)
{
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

    //compute
    const algorithmFPType invSqrSigma = (algorithmFPType)(1.0 / (par->sigma * par->sigma));
    algorithmFPType factor            = 0.0;
    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < nFeatures; i++)
    {
        algorithmFPType diff = (dataA1[i] - dataA2[i]);
        factor += diff * diff;
    }
    factor *= -0.5 * invSqrSigma;
    daal::internal::MathInst<algorithmFPType, cpu>::vExp(1, &factor, mtR.get());
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status KernelImplRBF<defaultDense, algorithmFPType, cpu>::computeInternalMatrixVector(const NumericTable * a1, const NumericTable * a2,
                                                                                                NumericTable * r, const KernelParameter * par)
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
    const algorithmFPType invSqrSigma = (algorithmFPType)(1.0 / (par->sigma * par->sigma));
    for (size_t i = 0; i < nVectors1; i++)
    {
        algorithmFPType factor = 0.0;
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < nFeatures; j++)
        {
            algorithmFPType diff = (dataA1[i * nFeatures + j] - dataA2[j]);
            factor += diff * diff;
        }
        dataR[i] = -0.5 * invSqrSigma * factor;

        if (dataR[i] < MathInst<algorithmFPType, cpu>::vExpThreshold())
        {
            dataR[i] = MathInst<algorithmFPType, cpu>::vExpThreshold();
        }
    }
    daal::internal::MathInst<algorithmFPType, cpu>::vExp(nVectors1, dataR, dataR);
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status KernelImplRBF<defaultDense, algorithmFPType, cpu>::computeInternalMatrixMatrix(const NumericTable * a1, const NumericTable * a2,
                                                                                                NumericTable * r, const KernelParameter * par)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(KernelRBF.MatrixMatrix);

    SafeStatus safeStat;

    const size_t nVectors1   = a1->getNumberOfRows();
    const size_t nVectors2   = a2->getNumberOfRows();
    const size_t nFeatures   = a1->getNumberOfColumns();
    const bool isEqualMatrix = a1 == a2;

    const algorithmFPType coeff = static_cast<algorithmFPType>(-0.5 / (par->sigma * par->sigma));

    char trans = 'T', notrans = 'N';
    DAAL_INT one         = 1;
    algorithmFPType zero = algorithmFPType(0.0), onef = algorithmFPType(1.0);

    const bool isSOARes = r->getDataLayout() & NumericTableIface::soa;

    DAAL_OVERFLOW_CHECK_BY_ADDING(size_t, nVectors1, nVectors2);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nVectors1 + nVectors2, sizeof(algorithmFPType));

    const size_t blockSize                = 256;
    const size_t nBlocks1                 = nVectors1 / blockSize + !!(nVectors1 % blockSize);
    const size_t nBlocks2                 = nVectors2 / blockSize + !!(nVectors2 % blockSize);
    const algorithmFPType expExpThreshold = MathInst<algorithmFPType, cpu>::vExpThreshold();

    daal::tls<KernelRBFTask<algorithmFPType, cpu> *> tslTask([=, &safeStat]() {
        auto tlsData = KernelRBFTask<algorithmFPType, cpu>::create(blockSize, isEqualMatrix);
        if (!tlsData)
        {
            safeStat.add(services::ErrorMemoryAllocationFailed);
        }
        return tlsData;
    });

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

            KernelRBFTask<algorithmFPType, cpu> * tlsLocal = tslTask.local();
            algorithmFPType * const mklBuff                = tlsLocal->mklBuff;
            algorithmFPType * const sqrDataA1              = tlsLocal->sqrDataA1;
            algorithmFPType * const sqrDataA2              = tlsLocal->sqrDataA2;

            if (!isEqualMatrix)
            {
                for (size_t i = 0; i < nRowsInBlock1; ++i)
                {
                    const algorithmFPType * dataA1i = dataA1 + i * nFeatures;
                    sqrDataA1[i]                    = BlasInst<algorithmFPType, cpu>::xxdot((DAAL_INT *)&nFeatures, dataA1i, &one, dataA1i, &one);
                }
            }

            ReadRows<algorithmFPType, cpu> mtA2(*const_cast<NumericTable *>(a2), startRow2, nRowsInBlock2);
            DAAL_CHECK_BLOCK_STATUS_THR(mtA2);
            const algorithmFPType * const dataA2 = const_cast<algorithmFPType *>(mtA2.get());

            for (size_t i = 0; i < nRowsInBlock2; ++i)
            {
                const algorithmFPType * dataA2i = dataA2 + i * nFeatures;
                sqrDataA2[i]                    = BlasInst<algorithmFPType, cpu>::xxdot((DAAL_INT *)&nFeatures, dataA2i, &one, dataA2i, &one);
            }

            DAAL_INT lda = nFeatures;
            DAAL_INT ldb = nFeatures;
            DAAL_INT ldc = blockSize;
            if (!isSOARes)
            {
                BlasInst<algorithmFPType, cpu>::xxgemm(&trans, &notrans, &nRowsInBlock2, &nRowsInBlock1, (DAAL_INT *)&nFeatures, &onef, dataA2, &ldb,
                                                       dataA1, &lda, &zero, mklBuff, &ldc);

                algorithmFPType * const dataR = mtRRows.get();
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
                BlasInst<algorithmFPType, cpu>::xxgemm(&trans, &notrans, &nRowsInBlock1, &nRowsInBlock2, (DAAL_INT *)&nFeatures, &onef, dataA1, &lda,
                                                       dataA2, &ldb, &zero, mklBuff, &ldc);

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
