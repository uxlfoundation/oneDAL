/* file: linear_model_predict_dense_default_batch_impl.i */
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
//  Common functions for linear regression predictions calculation
//--
*/

#ifndef __LINEAR_MODEL_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__
#define __LINEAR_MODEL_PREDICT_DENSE_DEFAULT_BATCH_IMPL_I__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "src/threading/threading.h"
#include "services/daal_defines.h"
#include "src/services/service_defines.h"
#include "src/externals/service_blas.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/service_error_handling.h"

namespace daal
{
namespace algorithms
{
namespace linear_model
{
namespace prediction
{
namespace internal
{
using namespace daal::internal;
using namespace daal::services;

template <typename algorithmFPType, CpuType cpu>
services::Status PredictKernel<algorithmFPType, defaultDense, cpu>::computeBlockOfResponses(size_t startRow, size_t numRows, NumericTable * dataTable,
                                                                                            size_t numBetas, const algorithmFPType * beta,
                                                                                            size_t numResponses, algorithmFPType * responseBlock,
                                                                                            bool findBeta0)
{
    Status st;
    const size_t numFeatures = dataTable->getNumberOfColumns();
    /* Retrieve data blocks associated with input and resulting tables */
    ReadRows<algorithmFPType, cpu> dataRows(dataTable, startRow, numRows);
    DAAL_CHECK_BLOCK_STATUS(dataRows);
    const algorithmFPType * dataBlock = dataRows.get();

    /* GEMM parameters */
    char trans           = 'T';
    char notrans         = 'N';
    algorithmFPType one  = 1.0;
    algorithmFPType zero = 0.0;

    BlasInst<algorithmFPType, cpu>::xxgemm(&trans, &notrans, (DAAL_INT *)&numResponses, (DAAL_INT *)&numRows, (DAAL_INT *)&numFeatures, &one,
                                           beta + 1, (DAAL_INT *)&numBetas, dataBlock, (DAAL_INT *)&numFeatures, &zero, responseBlock,
                                           (DAAL_INT *)&numResponses);

    if (findBeta0)
    {
        /* Add intercept term to linear regression results */
        DAAL_INT iZero = 0;
        for (size_t j = 0; j < numResponses; ++j)
        {
            BlasInst<algorithmFPType, cpu>::xxaxpy((DAAL_INT *)&numRows, &one, beta + j * numBetas, &iZero, responseBlock + j,
                                                   (DAAL_INT *)&numResponses);
        }
    }
    return st;
}

template <typename algorithmFPType, CpuType cpu>
services::Status PredictKernel<algorithmFPType, defaultDense, cpu>::computeBlockOfResponsesSOA(
    size_t startRow, size_t numRows, NumericTable * dataTable, size_t numBetas, const algorithmFPType * beta, size_t numResponses,
    algorithmFPType * responseBlock, bool findBeta0, bool isHomogeneous, TlsMem<algorithmFPType, cpu> & tlsData)
{
    Status st;
    char trans                       = 'T';
    char notrans                     = 'N';
    algorithmFPType one              = 1.0;
    const size_t numFeatures         = dataTable->getNumberOfColumns();
    const size_t numRowsInData       = dataTable->getNumberOfRows();
    const size_t numBlockSizeColumns = blockSizeColumns;
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, numRows, numResponses);
    services::internal::service_memset_seq<algorithmFPType, cpu>(responseBlock, algorithmFPType(0.0), numRows * numResponses);

    const size_t numBlocks = numFeatures / blockSizeColumns + !!(numFeatures % blockSizeColumns);
    for (size_t iBlock = 0; iBlock < numBlocks; ++iBlock)
    {
        const size_t startColumn       = iBlock * blockSizeColumns;
        const size_t numColumnsInBlock = (iBlock == numBlocks - 1) ? numFeatures - startColumn : blockSizeColumns;

        if (isHomogeneous)
        {
            ReadColumns<algorithmFPType, cpu> xBlock(dataTable, startColumn, startRow, numRows);
            DAAL_CHECK_BLOCK_STATUS(xBlock);
            const algorithmFPType * data = xBlock.get();
            BlasInst<algorithmFPType, cpu>::xxgemm(&trans, &trans, (DAAL_INT *)&numResponses, (DAAL_INT *)&numRows, (DAAL_INT *)&numColumnsInBlock,
                                                   &one, beta + 1 + startColumn, (DAAL_INT *)&numBetas, data, (DAAL_INT *)&numRowsInData, &one,
                                                   responseBlock, (DAAL_INT *)&numResponses);
        }
        else
        {
            algorithmFPType * tlsLocal = tlsData.local();
            DAAL_CHECK_MALLOC(tlsLocal);
            for (size_t i = 0; i < numColumnsInBlock; ++i)
            {
                ReadColumns<algorithmFPType, cpu> xBlock(dataTable, i + startColumn, startRow, numRows);
                DAAL_CHECK_BLOCK_STATUS(xBlock);
                services::internal::tmemcpy<algorithmFPType, cpu>(tlsLocal + i * blockSizeRows, xBlock.get(), numRows);
            }
            BlasInst<algorithmFPType, cpu>::xxgemm(&trans, &trans, (DAAL_INT *)&numResponses, (DAAL_INT *)&numRows, (DAAL_INT *)&numColumnsInBlock,
                                                   &one, beta + 1 + startColumn, (DAAL_INT *)&numBetas, tlsLocal, (DAAL_INT *)&numBlockSizeColumns,
                                                   &one, responseBlock, (DAAL_INT *)&numResponses);
        }
    }
    if (findBeta0)
    {
        const DAAL_INT zero = 0;
        for (size_t j = 0; j < numResponses; ++j)
        {
            BlasInst<algorithmFPType, cpu>::xxaxpy((DAAL_INT *)&numRows, &one, beta + j * numBetas, &zero, responseBlock + j,
                                                   (DAAL_INT *)&numResponses);
        }
    }
    return st;
}

template <typename algorithmFPType, CpuType cpu>
services::Status PredictKernel<algorithmFPType, defaultDense, cpu>::compute_impl(const NumericTable * a, const NumericTable * b, NumericTable * r,
                                                                                 bool intercept_flag)
{
    /* Get numeric tables with input data */
    NumericTable * dataTable = const_cast<NumericTable *>(a);

    /* Get sizes of input data */
    const size_t numVectors  = dataTable->getNumberOfRows();
    const size_t numFeatures = dataTable->getNumberOfColumns();

    bool isHomogeneous = false;
    if (dataTable->getDataLayout() & NumericTableIface::soa)
    {
        SOANumericTable * soaDataPtr = dynamic_cast<SOANumericTable *>(dataTable);
        DAAL_CHECK(soaDataPtr, services::ErrorNullNumericTable);
        isHomogeneous = soaDataPtr->isHomogeneousFloatOrDouble();
        auto f        = (*(soaDataPtr->getDictionary()))[0];
        isHomogeneous &= data_management::features::getIndexNumType<algorithmFPType>() == f.indexType;

        for (size_t i = 1; i < numFeatures && isHomogeneous; ++i)
        {
            algorithmFPType * fisrtArrayPtr = (algorithmFPType *)(soaDataPtr->getArray(i - 1));
            algorithmFPType * lastArrayPtr  = (algorithmFPType *)(soaDataPtr->getArray(i));
            if ((lastArrayPtr - fisrtArrayPtr) != numVectors)
            {
                isHomogeneous = false;
            }
        }
    }

    /* Get linear regression coefficients */
    NumericTable * betaTable  = const_cast<NumericTable *>(b);
    const size_t numResponses = betaTable->getNumberOfRows();
    const size_t numBetas     = betaTable->getNumberOfColumns();

    /* Retrieve data associated with coefficients */
    ReadRows<algorithmFPType, cpu> betaRows(betaTable, 0, numResponses);
    DAAL_CHECK_BLOCK_STATUS(betaRows)
    const algorithmFPType * beta = betaRows.get();

    /* Calculate number of blocks */
    const size_t numBlocks = numVectors / blockSizeRows + !!(numVectors % blockSizeRows);

    SafeStatus safeStat;
    TlsMem<algorithmFPType, cpu> tlsData(blockSizeRows * blockSizeColumns);

    daal::threader_for(numBlocks, 1, [&](int iBlock) {
        const size_t startRow       = iBlock * blockSizeRows;
        const size_t numRowsInBlock = (iBlock == numBlocks - 1) ? numVectors - startRow : blockSizeRows;

        WriteOnlyRows<algorithmFPType, cpu> responseRows(r, startRow, numRowsInBlock);
        DAAL_CHECK_BLOCK_STATUS_THR(responseRows);
        algorithmFPType * responseBlock = responseRows.get();

        /* Calculate predictions */
        if (dataTable->getDataLayout() & NumericTableIface::soa)
        {
            DAAL_CHECK_STATUS_THR(computeBlockOfResponsesSOA(startRow, numRowsInBlock, dataTable, numBetas, beta, numResponses, responseBlock,
                                                             intercept_flag, isHomogeneous, tlsData));
        }
        else
        {
            DAAL_CHECK_STATUS_THR(
                computeBlockOfResponses(startRow, numRowsInBlock, dataTable, numBetas, beta, numResponses, responseBlock, intercept_flag));
        }
    }); /* daal::threader_for */
    return safeStat.detach();
} /* void PredictKernel<algorithmFPType, defaultDense, cpu>::compute_impl */

template <typename algorithmFPType, CpuType cpu>
services::Status PredictKernel<algorithmFPType, defaultDense, cpu>::compute(const NumericTable * a, const linear_model::Model * m, NumericTable * r)
{
    linear_model::Model * model = const_cast<linear_model::Model *>(m);
    return compute_impl(a, model->getBeta().get(), r, model->getInterceptFlag());
}

} /* namespace internal */
} /* namespace prediction */
} /* namespace linear_model */
} /* namespace algorithms */
} /* namespace daal */

#endif
