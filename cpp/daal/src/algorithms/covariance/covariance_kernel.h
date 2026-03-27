/* file: covariance_kernel.h */
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
//  Declaration of template structs that calculate Covariance matrix.
//--
*/

#ifndef __COVARIANCE_KERNEL_H__
#define __COVARIANCE_KERNEL_H__

#include "src/algorithms/kernel.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/algorithm_base_common.h"
#include "algorithms/covariance/covariance_types.h"
#include "src/algorithms/covariance/covariance_hyperparameter_impl.h"

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services;
template <typename algorithmFPType, Method method, CpuType cpu>
class CovarianceDenseBatchKernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(NumericTable * dataTable, NumericTable * covTable, NumericTable * meanTable, const Parameter * parameter,
                             const Hyperparameter * hyperparameter = nullptr);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class CovarianceCSRBatchKernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(NumericTable * dataTable, NumericTable * covTable, NumericTable * meanTable, const Parameter * parameter,
                             const Hyperparameter * hyperparameter = nullptr);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class CovarianceDenseOnlineKernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(NumericTable * dataTable, NumericTable * nObsTable, NumericTable * crossProductTable, NumericTable * sumTable,
                             const Parameter * parameter, const Hyperparameter * hyperparameter = nullptr);

    services::Status finalizeCompute(NumericTable * nObsTable, NumericTable * crossProductTable, NumericTable * sumTable, NumericTable * covTable,
                                     NumericTable * meanTable, const Parameter * parameter, const Hyperparameter * hyperparameter = nullptr);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class CovarianceCSROnlineKernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(NumericTable * dataTable, NumericTable * nObsTable, NumericTable * crossProductTable, NumericTable * sumTable,
                             const Parameter * parameter, const Hyperparameter * hyperparameter = nullptr);

    services::Status finalizeCompute(NumericTable * nObsTable, NumericTable * crossProductTable, NumericTable * sumTable, NumericTable * covTable,
                                     NumericTable * meanTable, const Parameter * parameter, const Hyperparameter * hyperparameter = nullptr);
};

template <typename algorithmFPType, Method method, CpuType cpu>
class CovarianceDistributedKernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(DataCollection * partialResultsCollection, NumericTable * nObsTable, NumericTable * crossProductTable,
                             NumericTable * sumTable, const Parameter * parameter, const Hyperparameter * hyperparameter = nullptr);

    services::Status finalizeCompute(NumericTable * nObsTable, NumericTable * crossProductTable, NumericTable * sumTable, NumericTable * covTable,
                                     NumericTable * meanTable, const Parameter * parameter, const Hyperparameter * hyperparameter = nullptr);
};

} // namespace internal
} // namespace covariance
} // namespace algorithms
} // namespace daal

#endif
