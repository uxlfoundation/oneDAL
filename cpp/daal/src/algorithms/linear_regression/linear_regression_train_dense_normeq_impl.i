/* file: linear_regression_train_dense_normeq_impl.i */
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
//  Implementation of auxiliary functions for linear regression
//  Normal Equations (normEqDense) method.
//--
*/

#ifndef __LINEAR_REGRESSION_TRAIN_DENSE_NORMEQ_IMPL_I__
#define __LINEAR_REGRESSION_TRAIN_DENSE_NORMEQ_IMPL_I__

#include "src/algorithms/linear_regression/linear_regression_hyperparameter_impl.h"
#include "src/algorithms/linear_regression/linear_regression_train_kernel.h"

#include "services/daal_defines.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace training
{
namespace internal
{
using namespace daal::algorithms::linear_model::normal_equations::training::internal;

template <typename algorithmFPType, CpuType cpu>
Status BatchKernel<algorithmFPType, training::normEqDense, cpu>::compute(const NumericTable & x, const NumericTable & y, NumericTable & xtx,
                                                                         NumericTable & xty, NumericTable & beta, bool interceptFlag) const
{
    return this->compute(x, y, xtx, xty, beta, interceptFlag, nullptr);
}

template <typename algorithmFPType, CpuType cpu>
Status BatchKernel<algorithmFPType, training::normEqDense, cpu>::compute(const NumericTable & x, const NumericTable & y, NumericTable & xtx,
                                                                         NumericTable & xty, NumericTable & beta, bool interceptFlag,
                                                                         const HyperparameterType * hyperparameter) const
{
    services::SharedPtr<linear_model::internal::Hyperparameter> lmHyperparameter;
    DAAL_CHECK_STATUS_VAR(linear_regression::internal::convert(hyperparameter, lmHyperparameter));
    DAAL_CHECK_STATUS_VAR(UpdateKernelType::compute(x, y, xtx, xty, true, interceptFlag, lmHyperparameter.get()));
    return FinalizeKernelType::compute(xtx, xty, xtx, xty, beta, interceptFlag, KernelHelper<algorithmFPType, cpu>(), lmHyperparameter.get());
}

template <typename algorithmFPType, CpuType cpu>
Status OnlineKernel<algorithmFPType, training::normEqDense, cpu>::compute(const NumericTable & x, const NumericTable & y, NumericTable & xtx,
                                                                          NumericTable & xty, bool interceptFlag) const
{
    return this->compute(x, y, xtx, xty, interceptFlag, nullptr);
}

template <typename algorithmFPType, CpuType cpu>
Status OnlineKernel<algorithmFPType, training::normEqDense, cpu>::compute(const NumericTable & x, const NumericTable & y, NumericTable & xtx,
                                                                          NumericTable & xty, bool interceptFlag,
                                                                          const HyperparameterType * hyperparameter) const
{
    services::SharedPtr<linear_model::internal::Hyperparameter> lmHyperparameter;
    DAAL_CHECK_STATUS_VAR(linear_regression::internal::convert(hyperparameter, lmHyperparameter));
    return UpdateKernelType::compute(x, y, xtx, xty, false, interceptFlag, lmHyperparameter.get());
}

template <typename algorithmFPType, CpuType cpu>
Status OnlineKernel<algorithmFPType, training::normEqDense, cpu>::finalizeCompute(const NumericTable & xtx, const NumericTable & xty,
                                                                                  NumericTable & xtxFinal, NumericTable & xtyFinal,
                                                                                  NumericTable & beta, bool interceptFlag) const
{
    return this->finalizeCompute(xtx, xty, xtxFinal, xtyFinal, beta, interceptFlag, nullptr);
}

template <typename algorithmFPType, CpuType cpu>
Status OnlineKernel<algorithmFPType, training::normEqDense, cpu>::finalizeCompute(const NumericTable & xtx, const NumericTable & xty,
                                                                                  NumericTable & xtxFinal, NumericTable & xtyFinal,
                                                                                  NumericTable & beta, bool interceptFlag,
                                                                                  const HyperparameterType * hyperparameter) const
{
    services::SharedPtr<linear_model::internal::Hyperparameter> lmHyperparameter;
    DAAL_CHECK_STATUS_VAR(linear_regression::internal::convert(hyperparameter, lmHyperparameter));
    return FinalizeKernelType::compute(xtx, xty, xtxFinal, xtyFinal, beta, interceptFlag, KernelHelper<algorithmFPType, cpu>(),
                                       lmHyperparameter.get());
}

} // namespace internal
} // namespace training
} // namespace linear_regression
} // namespace algorithms
} // namespace daal

#endif
