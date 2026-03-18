/* file: kernel_function_rbf_csr_fast_kernel.h */
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
//  Declaration of template structs that calculate SVM RBF Kernel functions.
//--
*/

#ifndef __KERNEL_FUNCTION_RBF_CSR_FAST_KERNEL_H__
#define __KERNEL_FUNCTION_RBF_CSR_FAST_KERNEL_H__

#include "src/algorithms/kernel_function/kernel_function_csr_base.h"
#include "src/algorithms/kernel_function/kernel_function_rbf_base.h"

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace rbf
{
namespace internal
{
using namespace daal::data_management;

template <typename algorithmFPType, daal::internal::CpuType cpu>
struct KernelImplRBF<fastCSR, algorithmFPType, cpu> : public daal::algorithms::kernel_function::internal::KernelCSRImplBase<algorithmFPType, cpu>
{
    using daal::algorithms::kernel_function::internal::KernelCSRImplBase<algorithmFPType, cpu>::computeDotProduct;

    virtual services::Status computeInternalVectorVector(const NumericTable * a1, const NumericTable * a2, NumericTable * r,
                                                         const KernelParameter * par);
    virtual services::Status computeInternalMatrixVector(const NumericTable * a1, const NumericTable * a2, NumericTable * r,
                                                         const KernelParameter * par);
    virtual services::Status computeInternalMatrixMatrix(const NumericTable * a1, const NumericTable * a2, NumericTable * r,
                                                         const KernelParameter * par);
};

} // namespace internal

} // namespace rbf

} // namespace kernel_function

} // namespace algorithms

} // namespace daal

#endif
