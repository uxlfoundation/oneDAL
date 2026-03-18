/* file: lasso_regression_train_kernel.h */
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
//  Declaration of structure containing kernels for lasso regression
//  training.
//--
*/

#ifndef __LASSO_REGRESSION_TRAIN_KERNEL_H__
#define __LASSO_REGRESSION_TRAIN_KERNEL_H__

#include "data_management/data/numeric_table.h"
#include "algorithms/algorithm_base_common.h"
#include "algorithms/optimization_solver/objective_function/mse_batch.h"
#include "algorithms/lasso_regression/lasso_regression_training_types.h"

namespace daal
{
namespace algorithms
{
namespace lasso_regression
{
namespace training
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::internal;
using namespace daal::services;

template <typename algorithmFPType, Method method, CpuType cpu>
class TrainBatchKernel : public daal::algorithms::Kernel
{
public:
    services::Status compute(const HostAppIfacePtr & pHost, const NumericTablePtr & x, const NumericTablePtr & y, lasso_regression::Model & m,
                             Result & res, const Parameter & par,
                             services::SharedPtr<daal::algorithms::optimization_solver::mse::Batch<algorithmFPType> > & objFunc);
};

} // namespace internal
} // namespace training
} // namespace lasso_regression
} // namespace algorithms
} // namespace daal

#endif
