/* file: implicit_als_train_csr_default_distr_impl.i */
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
//  Implementation of implicit ALS training algorithm for distributed processing mode
//--
*/

#ifndef __IMPLICIT_ALS_TRAIN_CSR_DEFAULT_DISTR_IMPL_I__
#define __IMPLICIT_ALS_TRAIN_CSR_DEFAULT_DISTR_IMPL_I__

#include "src/data_management/service_numeric_table.h"
#include "src/externals/service_memory.h"
#include "src/algorithms/service_error_handling.h"
#include "src/externals/service_blas.h"
#include "src/services/service_data_utils.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace internal
{
using namespace daal::services::internal;
using namespace daal::internal;
using namespace daal::services;

template <typename algorithmFPType, CpuType cpu>
Status ImplicitALSTrainDistrStep1Kernel<algorithmFPType, cpu>::compute(implicit_als::PartialModel * partialModel,
                                                                       data_management::NumericTable * crossProduct, const Parameter * parameter)
{
    const size_t maxBlockSize = 100 * 1024 * 1024;

    const size_t nFactors = parameter->nFactors;
    size_t nRowsInBlock   = maxBlockSize / nFactors;

    NumericTablePtr pFactors = partialModel->getFactors();
    const size_t nRows       = pFactors->getNumberOfRows();
    size_t nBlocks           = nRows / nRowsInBlock;
    if (nBlocks * nRowsInBlock < nRows) nBlocks++;
    if (nBlocks == 1) nRowsInBlock = nRows;

    WriteOnlyRows<algorithmFPType, cpu> mtCrossProduct(*crossProduct, 0, nFactors);
    DAAL_CHECK_BLOCK_STATUS(mtCrossProduct);
    algorithmFPType * cp = mtCrossProduct.get();

    const algorithmFPType zero(0.0);
    for (size_t i = 0; i < nFactors * nFactors; i++)
    {
        cp[i] = zero;
    }

    /* SYRK parameters */
    char uplo             = 'U';
    char trans            = 'N';
    algorithmFPType alpha = 1.0;
    algorithmFPType beta  = 1.0;

    ReadRows<algorithmFPType, cpu> mtFactors;
    for (size_t block = 0; block < nBlocks; block++)
    {
        const size_t iStart = block * nRowsInBlock;
        size_t iEnd         = iStart + nRowsInBlock;
        if (iEnd > nRows) iEnd = nRows;
        const size_t nRowsToCP = iEnd - iStart;
        mtFactors.set(*pFactors, iStart, nRowsToCP);
        DAAL_CHECK_BLOCK_STATUS(mtFactors);
        const algorithmFPType * srcFactorsBlock = mtFactors.get();
        BlasInst<algorithmFPType, cpu>::xsyrk(&uplo, &trans, (DAAL_INT *)&nFactors, (DAAL_INT *)&nRowsToCP, &alpha,
                                              const_cast<algorithmFPType *>(srcFactorsBlock), (DAAL_INT *)&nFactors, &beta, cp,
                                              (DAAL_INT *)&nFactors);
    }
    return Status();
}

template <typename algorithmFPType, CpuType cpu>
Status ImplicitALSTrainDistrStep2Kernel<algorithmFPType, cpu>::compute(size_t nParts, data_management::NumericTable ** partialCrossProducts,
                                                                       data_management::NumericTable * crossProduct, const Parameter * parameter)
{
    const size_t nFactors = parameter->nFactors;

    WriteOnlyRows<algorithmFPType, cpu> mtCrossProduct(*crossProduct, 0, nFactors);
    DAAL_CHECK_BLOCK_STATUS(mtCrossProduct);
    algorithmFPType * cp = mtCrossProduct.get();
    const algorithmFPType zero(0.0);
    for (size_t i = 0; i < nFactors * nFactors; i++) cp[i] = zero;

    ReadRows<algorithmFPType, cpu> mtPartialCrossProduct;
    for (size_t i = 0; i < nParts; i++)
    {
        mtPartialCrossProduct.set(*partialCrossProducts[i], 0, nFactors);
        DAAL_CHECK_BLOCK_STATUS(mtPartialCrossProduct);
        const algorithmFPType * partialCP = mtPartialCrossProduct.get();
        for (size_t j = 0; j < nFactors * nFactors; j++) cp[j] += partialCP[j];
    }
    return Status();
}

template <typename algorithmFPType, CpuType cpu>
Status ImplicitALSTrainDistrStep3Kernel<algorithmFPType, cpu>::compute(implicit_als::PartialModel * srcPartialModel,
                                                                       data_management::NumericTable * offsetTable,
                                                                       data_management::KeyValueDataCollection * dstPartialModels,
                                                                       const Parameter * parameter)
{
    int offset = 0;
    {
        ReadRows<int, cpu> mtOffset(offsetTable, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(mtOffset);
        const int * offsetData = mtOffset.get();
        offset                 = offsetData[0];
    }

    const size_t nPartialModels = dstPartialModels->size();
    const size_t nFactors       = parameter->nFactors;

    SafeStatus safeStat;
    daal::threader_for(nPartialModels, 0, [&](size_t k) {
        PartialModel * dstPartialModel = static_cast<PartialModel *>((*dstPartialModels)[k].get());
        NumericTablePtr pSrcFactors    = srcPartialModel->getFactors();
        NumericTablePtr pDstFactors    = dstPartialModel->getFactors();
        const size_t dstNRows          = pDstFactors->getNumberOfRows();
        const size_t nCols             = pDstFactors->getNumberOfColumns();
        const size_t sizeOfBlock       = 512;
        const size_t nBlocks           = (dstNRows + sizeOfBlock - 1) / sizeOfBlock;

        daal::threader_for(nBlocks, 0, [&](size_t i) {
            const size_t nRows = i < nBlocks - 1 ? sizeOfBlock : dstNRows - (i * sizeOfBlock);
            int result         = 0;

            ReadRows<algorithmFPType, cpu> mtSrcFactors;

            ReadRows<int, cpu> dstIndices(*dstPartialModel->getIndices(), i * sizeOfBlock, nRows);
            DAAL_CHECK_BLOCK_STATUS_THR(dstIndices);

            WriteOnlyRows<algorithmFPType, cpu> mtDstFactors(*pDstFactors, i * sizeOfBlock, nRows);
            DAAL_CHECK_BLOCK_STATUS_THR(mtDstFactors);

            algorithmFPType * dstFactor = mtDstFactors.get();
            const int * indices         = dstIndices.get();

            for (size_t j = 0; j < nRows; j++)
            {
                mtSrcFactors.set(*pSrcFactors, indices[j] - offset, 1);
                DAAL_CHECK_BLOCK_STATUS_THR(mtSrcFactors);
                const algorithmFPType * srcFactor = mtSrcFactors.get();

                result |= daal::services::internal::daal_memcpy_s(dstFactor + j * nCols, nFactors * sizeof(algorithmFPType), srcFactor,
                                                                  nFactors * sizeof(algorithmFPType));
            }
            if (result) safeStat.add(services::Status(services::ErrorMemoryCopyFailedInternal));
        });
    });

    return safeStat.detach();
}

template <typename algorithmFPType, CpuType cpu>
struct AlsTls
{
    DAAL_NEW_DELETE();
    AlsTls(size_t nBlocks, const Parameter & parameter) : _nBlocks(nBlocks), _prm(parameter), _lhs(parameter.nFactors * parameter.nFactors) {}
    bool isValid() const { return _lhs.get(); }

    Status run(NumericTable & dstFactors, ReadRowsCSR<algorithmFPType, cpu> & mtData, size_t i, const algorithmFPType * xtx,
               NumericTable ** aSrcFactors, const size_t * nColFactorsRows, const int ** indices);

protected:
    Status formSystem(ReadRowsCSR<algorithmFPType, cpu> & mtData, size_t i, NumericTable ** aSrcFactors, const size_t * nColFactorsRows,
                      const int ** indices);

protected:
    WriteOnlyRows<algorithmFPType, cpu> _mtDstFactors;
    TArray<algorithmFPType, cpu> _lhs;
    ReadRows<algorithmFPType, cpu> _mtSrcFactors;
    const Parameter & _prm;
    size_t _nBlocks;
};

template <typename algorithmFPType, CpuType cpu>
Status AlsTls<algorithmFPType, cpu>::run(NumericTable & dstFactors, ReadRowsCSR<algorithmFPType, cpu> & mtData, size_t i, const algorithmFPType * xtx,
                                         NumericTable ** aSrcFactors, const size_t * nColFactorsRows, const int ** indices)
{
    int result = 0;

    _mtDstFactors.set(dstFactors, i, 1);
    DAAL_CHECK_BLOCK_STATUS(_mtDstFactors);
    algorithmFPType * rhs = _mtDstFactors.get();
    service_memset<algorithmFPType, cpu>(rhs, 0.0, _prm.nFactors);
    result = daal::services::internal::daal_memcpy_s(_lhs.get(), _prm.nFactors * _prm.nFactors * sizeof(algorithmFPType), xtx,
                                                     _prm.nFactors * _prm.nFactors * sizeof(algorithmFPType));

    Status s = formSystem(mtData, i, aSrcFactors, nColFactorsRows, indices);
    s |= (result) ? services::Status(services::ErrorMemoryCopyFailedInternal) : s;

    /* Solve system of normal equations */
    if (s.ok() && !ImplicitALSTrainKernelBase<algorithmFPType, cpu>::solve(_prm.nFactors, _lhs.get(), rhs)) return Status(ErrorALSInternal);
    return s;
}

template <typename algorithmFPType, CpuType cpu>
Status ImplicitALSTrainDistrStep4Kernel<algorithmFPType, fastCSR, cpu>::compute(data_management::KeyValueDataCollection * srcPartialModels,
                                                                                data_management::NumericTable * dataTable,
                                                                                data_management::NumericTable * cpTable,
                                                                                implicit_als::PartialModel * dstPartialModel,
                                                                                const Parameter * parameter)
{
    const size_t nBlocks = srcPartialModels->size();
    TArray<size_t, cpu> nFactorsRows(nBlocks);
    TArray<const int *, cpu> indices(nBlocks);
    TArray<ReadRows<int, cpu>, cpu> mtIndices(nBlocks);
    TArray<NumericTable *, cpu> aSrcFactors(nBlocks);

    DAAL_CHECK_MALLOC(nFactorsRows.get() && indices.get() && mtIndices.get() && aSrcFactors.get());

    /* Initialize arrays of partial models data */
    for (size_t i = 0; i < nBlocks; i++)
    {
        PartialModel * srcPartialModel = static_cast<PartialModel *>((*srcPartialModels)[i].get());
        aSrcFactors[i]                 = srcPartialModel->getFactors().get();
        NumericTable * pIndices        = srcPartialModel->getIndices().get();
        const size_t nRows             = pIndices->getNumberOfRows();
        nFactorsRows[i]                = nRows;
        mtIndices[i].set(*pIndices, 0, nRows);
        DAAL_CHECK_BLOCK_STATUS(mtIndices[i]);
        indices[i] = mtIndices[i].get();
    }

    /* Compute resulting partial factors */
    daal::tls<AlsTls<algorithmFPType, cpu> *> alsTls([=]() {
        auto ptr = new AlsTls<algorithmFPType, cpu>(nBlocks, *parameter);
        if (ptr && !ptr->isValid())
        {
            delete ptr;
            ptr = nullptr;
        }
        return ptr;
    });

    ReadRows<algorithmFPType, cpu> mtXTX(cpTable, 0, parameter->nFactors);
    DAAL_CHECK_BLOCK_STATUS(mtXTX);
    const algorithmFPType * xtx = mtXTX.get();

    const size_t nRows                    = dataTable->getNumberOfRows();
    const CSRNumericTableIface * csrIface = dynamic_cast<const CSRNumericTableIface *>(dataTable);
    DAAL_CHECK(csrIface, ErrorEmptyCSRNumericTable);
    ReadRowsCSR<algorithmFPType, cpu> mtData(*const_cast<CSRNumericTableIface *>(csrIface), 0, nRows);
    DAAL_CHECK_BLOCK_STATUS(mtData);

    NumericTablePtr pDstFactors = dstPartialModel->getFactors();
    SafeStatus safeStat;
    daal::threader_for(nRows, nRows, [&](size_t i) {
        AlsTls<algorithmFPType, cpu> * alsTlsLocal = alsTls.local();
        DAAL_CHECK_THR(alsTlsLocal, ErrorMemoryAllocationFailed);
        safeStat |= alsTlsLocal->run(*pDstFactors, mtData, i, xtx, aSrcFactors.get(), nFactorsRows.get(), indices.get());
    });

    alsTls.reduce([=](AlsTls<algorithmFPType, cpu> * alsTlsLocal) { delete alsTlsLocal; });
    return safeStat.detach();
}

template <typename algorithmFPType, CpuType cpu>
Status AlsTls<algorithmFPType, cpu>::formSystem(ReadRowsCSR<algorithmFPType, cpu> & mtData, size_t i, NumericTable ** aSrcFactors,
                                                const size_t * nColFactorsRows, const int ** indices)
{
    algorithmFPType * rhs = _mtDstFactors.get();
    algorithmFPType * lhs = _lhs.get();
    const size_t startIdx = mtData.rows()[i] - 1;
    const size_t endIdx   = mtData.rows()[i + 1] - 1;

    /* Update the linear system of normal equations */
    for (size_t j = startIdx; j < endIdx; j++)
    {
        algorithmFPType c1 = algorithmFPType(_prm.alpha) * mtData.values()[j];
        algorithmFPType c  = c1 + 1.0;
        DAAL_ASSERT(mtData.cols()[j] <= services::internal::MaxVal<int>::get())
        int colIndex = (int)mtData.cols()[j] - 1;

        int blockIndex = -1;
        /* find block that contains needed index */
        for (size_t block = 0; block < _nBlocks; block++)
        {
            if (indices[block] && indices[block][0] <= colIndex && colIndex <= indices[block][nColFactorsRows[block] - 1])
            {
                blockIndex = block;
                break;
            }
        }
        if (blockIndex == -1) return Status(ErrorALSInconsistentSparseDataBlocks);

        const int * blockIndices = indices[blockIndex];
        /* find index in the block using binary search */
        size_t hiIndex = nColFactorsRows[blockIndex] - 1;
        size_t loIndex = 0;
        size_t meIndex = ((loIndex + hiIndex) >> 1);
        while (colIndex != blockIndices[meIndex])
        {
            if (colIndex < blockIndices[meIndex])
                hiIndex = meIndex - 1;
            else if (colIndex > blockIndices[meIndex])
                loIndex = meIndex + 1;
            meIndex = ((loIndex + hiIndex) >> 1);
            if (loIndex >= hiIndex) break;
        }
        if (colIndex != blockIndices[meIndex]) return Status(ErrorALSInconsistentSparseDataBlocks);

        _mtSrcFactors.set(*aSrcFactors[blockIndex], meIndex, 1);
        DAAL_CHECK_BLOCK_STATUS(_mtSrcFactors);
        ImplicitALSTrainKernelBase<algorithmFPType, cpu>::updateSystem(_prm.nFactors, _mtSrcFactors.get(), &c1, &c, lhs, rhs);
    }

    /* Add regularization term */
    const algorithmFPType gamma = algorithmFPType(_prm.lambda) * (endIdx - startIdx);
    for (size_t k = 0; k < _prm.nFactors; k++)
    {
        lhs[k * _prm.nFactors + k] += gamma;
    }
    return Status();
}

} // namespace internal
} // namespace training
} // namespace implicit_als
} // namespace algorithms
} // namespace daal

#endif
