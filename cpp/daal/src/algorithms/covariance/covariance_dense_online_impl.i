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
#include <iostream>

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace internal
{
template <typename algorithmFPType, Method method, CpuType cpu, typename SumsArrayType>
services::Status prepareSums(NumericTable * dataTable, const bool isNormalized, algorithmFPType *& userSums,
                             ReadRows<algorithmFPType, cpu> & userSumsBlock, SumsArrayType & userSumsArray)
{
    std::cout << "prepareSums" << std::endl;
    if (method == defaultDense || (method == sumDense && isNormalized))
    {
        std::cout << "prepareSums1" << std::endl;
        userSumsArray.reset(dataTable->getNumberOfColumns());
        DAAL_CHECK_MALLOC(userSumsArray.get());
        std::cout << "prepareSums2" << std::endl;
        userSums = userSumsArray.get();
        std::cout << "prepareSums3" << std::endl;
    }
    else /* method == sumDense */
    {
        std::cout << "prepareSums4" << std::endl;
        NumericTable * userSumsTable = dataTable->basicStatistics.get(NumericTable::sum).get();
        userSumsBlock.set(userSumsTable, 0, userSumsTable->getNumberOfRows());
        DAAL_CHECK_BLOCK_STATUS(userSumsBlock);
        std::cout << "prepareSums5" << std::endl;
        userSums = const_cast<algorithmFPType *>(userSumsBlock.get());
        std::cout << "prepareSums6" << std::endl;
    }

    return services::Status();
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status CovarianceDenseOnlineKernel<algorithmFPType, method, cpu>::compute(NumericTable * dataTable, NumericTable * nObservationsTable,
                                                                                    NumericTable * crossProductTable, NumericTable * sumTable,
                                                                                    const Parameter * parameter,
                                                                                    const Hyperparameter * hyperparameter)
{
    std::cout << "compute 1" << std::endl;
    const size_t nFeatures  = dataTable->getNumberOfColumns();
    const size_t nVectors   = dataTable->getNumberOfRows();
    const bool isNormalized = dataTable->isNormalized(NumericTableIface::standardScoreNormalized);
    std::cout << "compute 2" << std::endl;
    DEFINE_TABLE_BLOCK(WriteRows, sumBlock, sumTable);
    DEFINE_TABLE_BLOCK(WriteRows, crossProductBlock, crossProductTable);
    DEFINE_TABLE_BLOCK(WriteRows, nObservationsBlock, nObservationsTable);
    std::cout << "compute 3" << std::endl;
    algorithmFPType * sums          = sumBlock.get();
    algorithmFPType * crossProduct  = crossProductBlock.get();
    algorithmFPType * nObservations = nObservationsBlock.get();
    std::cout << "compute 4" << std::endl;
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nFeatures, nFeatures);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nFeatures * nFeatures, sizeof(algorithmFPType));
    std::cout << "compute 5" << std::endl;
    services::Status status;
    if (method == singlePassDense)
    {
        std::cout << "compute 6" << std::endl;
        status |= updateDenseCrossProductAndSums<algorithmFPType, method, cpu>(isNormalized, nFeatures, nVectors, dataTable, crossProduct, sums,
                                                                               nObservations, parameter, hyperparameter);
        DAAL_CHECK_STATUS_VAR(status);
        std::cout << "compute 7" << std::endl;
    }
    else
    {
        std::cout << "compute 8" << std::endl;
        algorithmFPType partialNObservations = 0.0;

        algorithmFPType * userSums;
        ReadRows<algorithmFPType, cpu> userSumsBlock;
        TArrayCalloc<algorithmFPType, cpu> userSumsArray;
        std::cout << "compute 9" << std::endl;
        status |= prepareSums<algorithmFPType, method, cpu>(dataTable, isNormalized, userSums, userSumsBlock, userSumsArray);
        DAAL_CHECK_STATUS_VAR(status);
        std::cout << "compute 10" << std::endl;
        TArrayCalloc<algorithmFPType, cpu> partialCrossProductArray(nFeatures * nFeatures);
        DAAL_CHECK_MALLOC(partialCrossProductArray.get());
        algorithmFPType * partialCrossProduct = partialCrossProductArray.get();
        std::cout << "compute 11" << std::endl;
        status |= updateDenseCrossProductAndSums<algorithmFPType, method, cpu>(isNormalized, nFeatures, nVectors, dataTable, partialCrossProduct,
                                                                               userSums, &partialNObservations, parameter, hyperparameter);
        DAAL_CHECK_STATUS_VAR(status);
        std::cout << "compute 12" << std::endl;
        mergeCrossProductAndSums<algorithmFPType, cpu>(nFeatures, partialCrossProduct, userSums, &partialNObservations, crossProduct, sums,
                                                       nObservations, hyperparameter);
        std::cout << "compute 13" << std::endl;
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
