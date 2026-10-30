/* file: linear_regression_train_kernel.h */
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
//  Declaration of structure containing kernels for linear regression
//  training.
//--
*/

#ifndef __LINEAR_REGRESSION_TRAIN_KERNEL_H__
#define __LINEAR_REGRESSION_TRAIN_KERNEL_H__

#include "data_management/data/numeric_table.h"
#include "algorithms/algorithm_base_common.h"
#include "algorithms/algorithm_kernel.h"

#include "src/algorithms/linear_model/linear_model_train_normeq_kernel.h"
#include "src/algorithms/linear_model/linear_model_train_qr_kernel.h"

#include "src/algorithms/linear_regression/linear_regression_hyperparameter_impl.h"
#include "algorithms/linear_regression/linear_regression_training_types.h"

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
using namespace daal::data_management;
using namespace daal::services;

using namespace daal::algorithms::linear_regression::internal;
using namespace daal::algorithms::linear_model::normal_equations::training::internal;

template <typename algorithmFPType, training::Method method, CpuType cpu>
class BatchKernel
{};

template <typename algorithmFPType, CpuType cpu>
class KernelHelper : public KernelHelperIface<algorithmFPType, cpu>
{
public:
    Status computeBetasImpl(DAAL_INT p, const algorithmFPType * a, algorithmFPType * aCopy, DAAL_INT ny, algorithmFPType * b, bool inteceptFlag,
                            bool isDualSystem) const;
    ErrorID getInternalErrorCode() const;
};

template <typename algorithmFPType, CpuType cpu>
class BatchKernel<algorithmFPType, training::normEqDense, cpu> : public daal::algorithms::Kernel
{
    typedef linear_model::normal_equations::training::internal::UpdateKernel<algorithmFPType, cpu> UpdateKernelType;
    typedef linear_model::normal_equations::training::internal::FinalizeKernel<algorithmFPType, cpu> FinalizeKernelType;
    typedef linear_model::normal_equations::training::internal::DualKernel<algorithmFPType, cpu> DualKernelType;

public:
    typedef linear_regression::internal::Hyperparameter HyperparameterType;
    Status compute(const NumericTable & x, const NumericTable & y, NumericTable & xtx, NumericTable & xty, NumericTable & beta,
                   bool interceptFlag) const;
    Status compute(const NumericTable & x, const NumericTable & y, NumericTable & xtx, NumericTable & xty, NumericTable & beta, bool interceptFlag,
                   const HyperparameterType * hyperparameter) const;
};

template <typename algorithmFPType, CpuType cpu>
class BatchKernel<algorithmFPType, training::qrDense, cpu> : public daal::algorithms::Kernel
{
    typedef linear_model::qr::training::internal::UpdateKernel<algorithmFPType, cpu> UpdateKernelType;
    typedef linear_model::qr::training::internal::FinalizeKernel<algorithmFPType, cpu> FinalizeKernelType;

public:
    Status compute(const NumericTable & x, const NumericTable & y, NumericTable & r, NumericTable & qty, NumericTable & beta,
                   bool interceptFlag) const;
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
    typedef linear_regression::internal::Hyperparameter HyperparameterType;
    Status compute(const NumericTable & x, const NumericTable & y, NumericTable & xtx, NumericTable & xty, bool interceptFlag) const;
    Status finalizeCompute(const NumericTable & xtx, const NumericTable & xty, NumericTable & xtxFinal, NumericTable & xtyFinal, NumericTable & beta,
                           bool interceptFlag) const;
    Status compute(const NumericTable & x, const NumericTable & y, NumericTable & xtx, NumericTable & xty, bool interceptFlag,
                   const HyperparameterType * hyperparameter) const;
    Status finalizeCompute(const NumericTable & xtx, const NumericTable & xty, NumericTable & xtxFinal, NumericTable & xtyFinal, NumericTable & beta,
                           bool interceptFlag, const HyperparameterType * hyperparameter) const;
};

template <typename algorithmFPType, CpuType cpu>
class OnlineKernel<algorithmFPType, training::qrDense, cpu> : public daal::algorithms::Kernel
{
    typedef linear_model::qr::training::internal::UpdateKernel<algorithmFPType, cpu> UpdateKernelType;
    typedef linear_model::qr::training::internal::FinalizeKernel<algorithmFPType, cpu> FinalizeKernelType;

public:
    Status compute(const NumericTable & x, const NumericTable & y, NumericTable & r, NumericTable & qty, bool interceptFlag) const;
    Status finalizeCompute(const NumericTable & r, const NumericTable & qty, NumericTable & rFinal, NumericTable & qtyFinal, NumericTable & beta,
                           bool interceptFlag) const;
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
    typedef linear_regression::internal::Hyperparameter HyperparameterType;
    Status compute(size_t n, NumericTable ** partialxtx, NumericTable ** partialxty, NumericTable & xtx, NumericTable & xty) const;
    Status finalizeCompute(const NumericTable & xtx, const NumericTable & xty, NumericTable & xtxFinal, NumericTable & xtyFinal, NumericTable & beta,
                           bool interceptFlag) const;
    Status compute(size_t n, NumericTable ** partialxtx, NumericTable ** partialxty, NumericTable & xtx, NumericTable & xty,
                   const HyperparameterType * hyperparameter) const;
    Status finalizeCompute(const NumericTable & xtx, const NumericTable & xty, NumericTable & xtxFinal, NumericTable & xtyFinal, NumericTable & beta,
                           bool interceptFlag, const HyperparameterType * hyperparameter) const;
};

template <typename algorithmFPType, CpuType cpu>
class DistributedKernel<algorithmFPType, training::qrDense, cpu> : public daal::algorithms::Kernel
{
    typedef linear_model::qr::training::internal::MergeKernel<algorithmFPType, cpu> MergeKernelType;
    typedef linear_model::qr::training::internal::FinalizeKernel<algorithmFPType, cpu> FinalizeKernelType;

public:
    Status compute(size_t n, NumericTable ** partialr, NumericTable ** partialqty, NumericTable & r, NumericTable & qty) const;
    Status finalizeCompute(const NumericTable & r, const NumericTable & qty, NumericTable & rFinal, NumericTable & qtyFinal, NumericTable & beta,
                           bool interceptFlag) const;
};

} // namespace internal
} // namespace training
} // namespace linear_regression
} // namespace algorithms
} // namespace daal

#endif
