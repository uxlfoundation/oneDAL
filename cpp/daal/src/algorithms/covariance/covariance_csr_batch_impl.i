/* file: covariance_csr_batch_impl.i */
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
//  Covariance matrix computation algorithm implementation in batch mode
//--
*/

#ifndef __COVARIANCE_CSR_BATCH_IMPL_I__
#define __COVARIANCE_CSR_BATCH_IMPL_I__

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
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status CovarianceCSRBatchKernel<algorithmFPType, method, cpu>::compute(NumericTable * dataTable, NumericTable * covTable,
                                                                                 NumericTable * meanTable, const Parameter * parameter,
                                                                                 const Hyperparameter * hyperparameter)
{
    algorithmFPType nObservations = 0.0;

    const size_t nFeatures              = dataTable->getNumberOfColumns();
    const size_t nVectors               = dataTable->getNumberOfRows();
    CSRNumericTableIface * csrDataTable = dynamic_cast<CSRNumericTableIface *>(dataTable);

    DEFINE_TABLE_BLOCK_EX(ReadRowsCSR, dataBlock, csrDataTable, 0, nVectors);
    DEFINE_TABLE_BLOCK(WriteOnlyRows, sumBlock, meanTable);
    DEFINE_TABLE_BLOCK(WriteOnlyRows, crossProductBlock, covTable);

    algorithmFPType * sums         = sumBlock.get();
    algorithmFPType * crossProduct = crossProductBlock.get();
    algorithmFPType * data         = const_cast<algorithmFPType *>(dataBlock.values());
    size_t * colIndices            = dataBlock.cols();
    size_t * rowOffsets            = dataBlock.rows();

    services::Status status;

    status |= prepareSums<algorithmFPType, method, cpu>(dataTable, sums);
    DAAL_CHECK_STATUS_VAR(status);

    status |= prepareCrossProduct<algorithmFPType, cpu>(nFeatures, crossProduct);
    DAAL_CHECK_STATUS_VAR(status);

    status |= updateCSRCrossProductAndSums<algorithmFPType, method, cpu>(nFeatures, nVectors, data, colIndices, rowOffsets, crossProduct, sums,
                                                                         &nObservations, hyperparameter);
    DAAL_CHECK_STATUS_VAR(status);

    const algorithmFPType invNRows = 1.0 / (algorithmFPType)nVectors;
    for (size_t i = 0; i < nFeatures; i++)
    {
        for (size_t j = 0; j < i; j++)
        {
            crossProduct[i * nFeatures + j] -= sums[i] * sums[j] * invNRows;
            crossProduct[j * nFeatures + i] = crossProduct[i * nFeatures + j];
        }
        crossProduct[i * nFeatures + i] -= sums[i] * sums[i] * invNRows;
    }

    status |= finalizeCovariance<algorithmFPType, cpu>(nFeatures, nObservations, crossProduct, sums, crossProduct, sums, parameter, hyperparameter);

    return status;
}

} // namespace internal
} // namespace covariance
} // namespace algorithms
} // namespace daal

#endif
