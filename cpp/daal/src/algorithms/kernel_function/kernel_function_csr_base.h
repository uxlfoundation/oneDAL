/* file: kernel_function_csr_base.h */
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
//  Declaration of template structs that calculate SVM Kernel functions.
//--
*/

#ifndef __KERNEL_FUNCTION_CSR_BASE_H__
#define __KERNEL_FUNCTION_CSR_BASE_H__

#include "data_management/data/numeric_table.h"
#include "src/algorithms/kernel_function/kernel_function_types_linear.h"
#include "src/algorithms/kernel_function/kernel_function_types_rbf.h"
#include "src/algorithms/kernel_function/kernel_function_linear.h"
#include "src/algorithms/kernel_function/kernel_function_rbf.h"
#include "src/data_management/service_numeric_table.h"
#include "src/algorithms/kernel.h"
#include "src/algorithms/kernel_function/kernel_function_dense_base.h"

using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
struct KernelCSRImplBase : public Kernel
{
    virtual services::Status computeInternalVectorVector(const NumericTable * a1, const NumericTable * a2, NumericTable * r,
                                                         const KernelParameter * par) = 0;
    virtual services::Status computeInternalMatrixVector(const NumericTable * a1, const NumericTable * a2, NumericTable * r,
                                                         const KernelParameter * par) = 0;
    virtual services::Status computeInternalMatrixMatrix(const NumericTable * a1, const NumericTable * a2, NumericTable * r,
                                                         const KernelParameter * par) = 0;

    services::Status compute(const NumericTable * a1, const NumericTable * a2, NumericTable * r, const KernelParameter * par)
    {
        ComputationMode computationMode = par->computationMode;
        switch (computationMode)
        {
        case vectorVector: return computeInternalVectorVector(a1, a2, r, par);
        case matrixVector: return computeInternalMatrixVector(a1, a2, r, par);
        case matrixMatrix: return computeInternalMatrixMatrix(a1, a2, r, par);
        default: return services::ErrorIncorrectParameter;
        }
    }

protected:
    inline algorithmFPType computeDotProduct(const size_t startIndex1, const size_t endIndex1, const algorithmFPType * dataA1,
                                             const size_t * colIndicesA1, const size_t startIndex2, const size_t endIndex2,
                                             const algorithmFPType * dataA2, const size_t * colIndicesA2);
};

} // namespace internal

} // namespace kernel_function

} // namespace algorithms

} // namespace daal

#endif
