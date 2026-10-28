/* file: ridge_regression_train_kernel.h */
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
//  Declaration of structure containing kernels for ridge regression training.
//--
*/

#ifndef __RIDGE_REGRESSION_TRAIN_KERNEL_H__
#define __RIDGE_REGRESSION_TRAIN_KERNEL_H__

#include "data_management/data/numeric_table.h"
#include "algorithms/algorithm_base_common.h"
#include "algorithms/ridge_regression/ridge_regression_training_types.h"
#include "src/algorithms/linear_model/linear_model_train_normeq_kernel.h"

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace training
{
namespace internal
{
using namespace daal::data_management;
using namespace daal::services;
using namespace daal::algorithms::linear_model::normal_equations::training::internal;

template <typename algorithmFPType, training::Method method, CpuType cpu>
class BatchKernel
{};

template <typename algorithmFPType, CpuType cpu>
class KernelHelper : public KernelHelperIface<algorithmFPType, cpu>
{
public:
    KernelHelper(const NumericTable & ridge) : _ridge(ridge) {}
    Status computeBetasImpl(DAAL_INT p, const algorithmFPType * a, algorithmFPType * aCopy, DAAL_INT ny, algorithmFPType * b, bool inteceptFlag,
                            bool isDualSystem) const;
    ErrorID getInternalErrorCode() const;

protected:
    const NumericTable & _ridge;
};

template <typename algorithmFPType, CpuType cpu>
class BatchKernel<algorithmFPType, training::normEqDense, cpu> : public daal::algorithms::Kernel
{
    typedef linear_model::normal_equations::training::internal::UpdateKernel<algorithmFPType, cpu> UpdateKernelType;
    typedef linear_model::normal_equations::training::internal::FinalizeKernel<algorithmFPType, cpu> FinalizeKernelType;
    typedef linear_model::normal_equations::training::internal::DualKernel<algorithmFPType, cpu> DualKernelType;

public:
    Status compute(const NumericTable & x, const NumericTable & y, NumericTable & xtx, NumericTable & xty, NumericTable & beta, bool interceptFlag,
                   const NumericTable & ridge) const;
};

template <typename algorithmFPType, training::Method method, CpuType cpu>
class OnlineKernel
{};

template <typename algorithmFPType, CpuType cpu>
class OnlineKernel<algorithmFPType, training::normEqDense, cpu> : public daal::algorithms::Kernel
{
    typedef linear_model::normal_equations::training::internal::UpdateKernel<algorithmFPType, cpu> UpdateKernelType;
    typedef linear_model::normal_equations::training::internal::FinalizeKernel<algorithmFPType, cpu> FinalizeKernelType;

public:
    Status compute(const NumericTable & x, const NumericTable & y, NumericTable & xtx, NumericTable & xty, bool interceptFlag) const;

    Status finalizeCompute(const NumericTable & xtx, const NumericTable & xty, NumericTable & xtxFinal, NumericTable & xtyFinal, NumericTable & beta,
                           bool interceptFlag, const NumericTable & ridge) const;
};

template <typename algorithmFPType, training::Method method, CpuType cpu>
class DistributedKernel
{};

template <typename algorithmFPType, CpuType cpu>
class DistributedKernel<algorithmFPType, training::normEqDense, cpu> : public daal::algorithms::Kernel
{
    typedef linear_model::normal_equations::training::internal::MergeKernel<algorithmFPType, cpu> MergeKernelType;
    typedef linear_model::normal_equations::training::internal::FinalizeKernel<algorithmFPType, cpu> FinalizeKernelType;

public:
    Status compute(size_t n, NumericTable ** partialxtx, NumericTable ** partialxty, NumericTable & xtx, NumericTable & xty) const;

    Status finalizeCompute(const NumericTable & xtx, const NumericTable & xty, NumericTable & xtxFinal, NumericTable & xtyFinal, NumericTable & beta,
                           bool interceptFlag, const NumericTable & ridge) const;
};

} // namespace internal
} // namespace training
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal

#endif
