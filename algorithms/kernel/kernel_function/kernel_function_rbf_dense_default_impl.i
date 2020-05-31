/* file: kernel_function_rbf_dense_default_impl.i */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
#include "service/kernel/data_management/service_numeric_table.h"
#include "externals/service_math.h"
#include "externals/service_blas.h"
#include "externals/service_ittnotify.h"
#include "algorithms/threading/threading.h"

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
                                                                                                NumericTable * r, const ParameterBase * par)
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
    const Parameter * rbfPar          = static_cast<const Parameter *>(par);
    const algorithmFPType invSqrSigma = (algorithmFPType)(1.0 / (rbfPar->sigma * rbfPar->sigma));
    algorithmFPType factor            = 0.0;
    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < nFeatures; i++)
    {
        algorithmFPType diff = (dataA1[i] - dataA2[i]);
        factor += diff * diff;
    }
    factor *= -0.5 * invSqrSigma;
    daal::internal::Math<algorithmFPType, cpu>::vExp(1, &factor, mtR.get());
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status KernelImplRBF<defaultDense, algorithmFPType, cpu>::computeInternalMatrixVector(const NumericTable * a1, const NumericTable * a2,
                                                                                                NumericTable * r, const ParameterBase * par)
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
    const Parameter * rbfPar          = static_cast<const Parameter *>(par);
    const algorithmFPType invSqrSigma = (algorithmFPType)(1.0 / (rbfPar->sigma * rbfPar->sigma));
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

        if (dataR[i] < Math<algorithmFPType, cpu>::vExpThreshold())
        {
            dataR[i] = Math<algorithmFPType, cpu>::vExpThreshold();
        }
    }
    daal::internal::Math<algorithmFPType, cpu>::vExp(nVectors1, dataR, dataR);
    return services::Status();
}

template <typename algorithmFPType, CpuType cpu>
services::Status KernelImplRBF<defaultDense, algorithmFPType, cpu>::computeInternalMatrixMatrix(const NumericTable * a1, const NumericTable * a2,
                                                                                                NumericTable * r, const ParameterBase * par)
{
    DAAL_ITTNOTIFY_SCOPED_TASK(KernelRBF.computeMatrixMatrix);

    SafeStatus safeStat;

    const size_t nVectors1 = a1->getNumberOfRows();
    const size_t nVectors2 = a2->getNumberOfRows();
    const size_t nFeatures = a1->getNumberOfColumns();

    const Parameter * rbfPar    = static_cast<const Parameter *>(par);
    const algorithmFPType coeff = (algorithmFPType)(-0.5 / (rbfPar->sigma * rbfPar->sigma));

    char trans = 'T', notrans = 'N';
    algorithmFPType zero = 0.0, negTwo = -2.0;

    DAAL_OVERFLOW_CHECK_BY_ADDING(size_t, nVectors1, nVectors2);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nVectors1 + nVectors2, sizeof(algorithmFPType));

    if (_aBuf.size() < nVectors1 + nVectors2)
    {
        printf("RESIZE\n");
        _aBuf.reset(nVectors1 + nVectors2);
        DAAL_CHECK(_aBuf.get(), services::ErrorMemoryAllocationFailed);
    }
    algorithmFPType * buffer = _aBuf.get();

    algorithmFPType * sqrDataA1 = buffer;
    // algorithmFPType * sqrDataA2 = buffer + nVectors1;

    const size_t blockSize = 128;
    const size_t nBlocks1  = nVectors1 / blockSize + !!(nVectors1 % blockSize);
    const size_t nBlocks2  = nVectors2 / blockSize + !!(nVectors2 % blockSize);

    // printf("nBlocks1 %lu nBlocks2 %lu\n", nBlocks1, nBlocks2);

    struct TslData
    {
    public:
        algorithmFPType * mklBuff;
        algorithmFPType * sqrDataA2;

        static TslData * create(const size_t blockSize, const size_t nVectors2)
        {
            auto object = new TslData(blockSize, nVectors2);
            if (!object)
            {
                return nullptr;
            }
            if (!(object->mklBuff) && !(object->sqrDataA2))
            {
                return nullptr;
            }
            return object;
        }

    private:
        TslData(const size_t blockSize, const size_t nVectors2)
        {
            _array.reset(blockSize * blockSize + nVectors2);

            mklBuff   = &_array[0];
            sqrDataA2 = &_array[blockSize * blockSize];
        }

        TArrayScalable<algorithmFPType, cpu> _array;
    };

    daal::tls<TslData *> tslData([=, &safeStat]() {
        auto tlsData = TslData::create(blockSize, nVectors2);
        if (!tlsData)
        {
            safeStat.add(services::ErrorMemoryAllocationFailed);
        }
        return tlsData;
    });

    daal::threader_for(nBlocks1, nBlocks1, [&](const size_t iBlock1) {
        DAAL_INT nRowsInBlock1 = (iBlock1 != nBlocks1 - 1) ? blockSize : nVectors1 - iBlock1 * blockSize;
        DAAL_INT startRow1     = iBlock1 * blockSize;
        DAAL_INT endRow1       = startRow1 + nRowsInBlock1;

        ReadRows<algorithmFPType, cpu> mtA1(*const_cast<NumericTable *>(a1), startRow1, nRowsInBlock1);
        DAAL_CHECK_BLOCK_STATUS_THR(mtA1);
        const algorithmFPType * dataA1 = const_cast<algorithmFPType *>(mtA1.get());

        WriteRows<algorithmFPType, cpu> mtR(r, startRow1, nRowsInBlock1);
        DAAL_CHECK_BLOCK_STATUS_THR(mtR);
        algorithmFPType * dataR = mtR.get();

        daal::threader_for(nBlocks2, nBlocks2, [&, nVectors2, nBlocks2](const size_t iBlock2) {
            DAAL_INT nRowsInBlock2 = (iBlock2 != nBlocks2 - 1) ? blockSize : nVectors2 - iBlock2 * blockSize;
            DAAL_INT startRow2     = iBlock2 * blockSize;
            DAAL_INT endRow2       = startRow2 + nRowsInBlock2;

            TslData * localTslData = tslData.local();

            algorithmFPType * mklBuff   = localTslData->mklBuff;
            algorithmFPType * sqrDataA2 = localTslData->sqrDataA2;

            ReadRows<algorithmFPType, cpu> mtA2(*const_cast<NumericTable *>(a2), startRow2, nRowsInBlock2);
            DAAL_CHECK_BLOCK_STATUS_THR(mtA2);
            const algorithmFPType * dataA2 = const_cast<algorithmFPType *>(mtA2.get());

            for (size_t i = 0; i < nRowsInBlock2; ++i)
            {
                sqrDataA2[i] = zero;
                for (size_t j = 0; j < nFeatures; ++j)
                {
                    sqrDataA2[i] += dataA2[i * nFeatures + j] * dataA2[i * nFeatures + j];
                }
            }

            Blas<algorithmFPType, cpu>::xxgemm(&trans, &notrans, &nRowsInBlock2, &nRowsInBlock1, (DAAL_INT *)&nFeatures, &negTwo, dataA2,
                                               (DAAL_INT *)&nFeatures, dataA1, (DAAL_INT *)&nFeatures, &zero, mklBuff, (DAAL_INT *)&blockSize);

            for (size_t i = 0; i < nRowsInBlock1; ++i)
            {
                algorithmFPType sqrDataA1 = zero;
                for (size_t j = 0; j < nFeatures; ++j)
                {
                    sqrDataA1 += dataA1[i * nFeatures + j] * dataA1[i * nFeatures + j];
                }

                PRAGMA_IVDEP
                PRAGMA_VECTOR_ALWAYS
                for (size_t j = 0; j < nRowsInBlock2; ++j)
                {
                    algorithmFPType rbf = (mklBuff[i * blockSize + j] + sqrDataA1 + sqrDataA2[j]) * coeff;
                    rbf                 = services::internal::max<cpu, algorithmFPType>(rbf, Math<algorithmFPType, cpu>::vExpThreshold());
                    rbf                 = Math<algorithmFPType, cpu>::sExp(rbf);
                    dataR[i * nVectors2 + iBlock2 * blockSize + j] = rbf;
                }
            }
        });
    });

    tslData.reduce([](TslData * localTslData) { delete localTslData; });

    return services::Status();
}

} // namespace internal
} // namespace rbf
} // namespace kernel_function
} // namespace algorithms
} // namespace daal

#endif
