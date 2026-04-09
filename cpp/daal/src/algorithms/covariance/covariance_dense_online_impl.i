/* file: covariance_dense_online_impl.i */
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
//  Covariance matrix computation algorithm implementation in distributed mode
//--
*/

#ifndef __COVARIANCE_DENSE_ONLINE_IMPL_I__
#define __COVARIANCE_DENSE_ONLINE_IMPL_I__

#include "src/algorithms/covariance/covariance_kernel.h"
#include "src/algorithms/covariance/covariance_impl.i"

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace internal
{
using namespace daal::internal;

template <typename algorithmFPType, Method method, CpuType cpu, typename SumsArrayType>
services::Status prepareSums(NumericTable * dataTable, const bool isNormalized, algorithmFPType *& userSums,
                             ReadRows<algorithmFPType, cpu> & userSumsBlock, SumsArrayType & userSumsArray)
{
    if (method == defaultDense || (method == sumDense && isNormalized))
    {
        userSumsArray.reset(dataTable->getNumberOfColumns());
        DAAL_CHECK_MALLOC(userSumsArray.get());

        userSums = userSumsArray.get();
    }
    else /* method == sumDense */
    {
        NumericTable * userSumsTable = dataTable->basicStatistics.get(NumericTable::sum).get();
        userSumsBlock.set(userSumsTable, 0, userSumsTable->getNumberOfRows());
        DAAL_CHECK_BLOCK_STATUS(userSumsBlock);

        userSums = const_cast<algorithmFPType *>(userSumsBlock.get());
    }

    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status CovarianceDenseOnlineKernel<algorithmFPType, method, cpu>::compute(NumericTable * dataTable, NumericTable * nObservationsTable,
                                                                                    NumericTable * crossProductTable, NumericTable * sumTable,
                                                                                    const Parameter * parameter,
                                                                                    const Hyperparameter * hyperparameter)
{
    const size_t nFeatures  = dataTable->getNumberOfColumns();
    const size_t nVectors   = dataTable->getNumberOfRows();
    const bool isNormalized = dataTable->isNormalized(NumericTableIface::standardScoreNormalized);

    DEFINE_TABLE_BLOCK(WriteRows, sumBlock, sumTable);
    DEFINE_TABLE_BLOCK(WriteRows, crossProductBlock, crossProductTable);
    DEFINE_TABLE_BLOCK(WriteRows, nObservationsBlock, nObservationsTable);

    algorithmFPType * sums          = sumBlock.get();
    algorithmFPType * crossProduct  = crossProductBlock.get();
    algorithmFPType * nObservations = nObservationsBlock.get();

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nFeatures, nFeatures);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nFeatures * nFeatures, sizeof(algorithmFPType));

    services::Status status;
    if (method == singlePassDense)
    {
        status |= updateDenseCrossProductAndSums<algorithmFPType, method, cpu>(isNormalized, nFeatures, nVectors, dataTable, crossProduct, sums,
                                                                               nObservations, parameter, hyperparameter);
        DAAL_CHECK_STATUS_VAR(status);
    }
    else
    {
        algorithmFPType partialNObservations = 0.0;

        algorithmFPType * userSums;
        ReadRows<algorithmFPType, cpu> userSumsBlock;
        TArrayCalloc<algorithmFPType, cpu> userSumsArray;

        status |= prepareSums<algorithmFPType, method, cpu>(dataTable, isNormalized, userSums, userSumsBlock, userSumsArray);
        DAAL_CHECK_STATUS_VAR(status);

        TArrayCalloc<algorithmFPType, cpu> partialCrossProductArray(nFeatures * nFeatures);
        DAAL_CHECK_MALLOC(partialCrossProductArray.get());
        algorithmFPType * partialCrossProduct = partialCrossProductArray.get();

        status |= updateDenseCrossProductAndSums<algorithmFPType, method, cpu>(isNormalized, nFeatures, nVectors, dataTable, partialCrossProduct,
                                                                               userSums, &partialNObservations, parameter, hyperparameter);
        DAAL_CHECK_STATUS_VAR(status);

        mergeCrossProductAndSums<algorithmFPType, cpu>(nFeatures, partialCrossProduct, userSums, &partialNObservations, crossProduct, sums,
                                                       nObservations, hyperparameter);
    }

    return status;
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status CovarianceDenseOnlineKernel<algorithmFPType, method, cpu>::finalizeCompute(NumericTable * nObservationsTable,
                                                                                            NumericTable * crossProductTable, NumericTable * sumTable,
                                                                                            NumericTable * covTable, NumericTable * meanTable,
                                                                                            const Parameter * parameter,
                                                                                            const Hyperparameter * hyperparameter)
{
    return finalizeCovariance<algorithmFPType, cpu>(nObservationsTable, crossProductTable, sumTable, covTable, meanTable, parameter, hyperparameter);
}

} // namespace internal
} // namespace covariance
} // namespace algorithms
} // namespace daal

#endif
