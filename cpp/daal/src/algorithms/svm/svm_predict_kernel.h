/* file: svm_predict_kernel.h */
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
//  Declaration of template structs that contains SVM prediction functions.
//--
*/

#ifndef __SVM_PREDICT_KERNEL_H__
#define __SVM_PREDICT_KERNEL_H__

#include "data_management/data/numeric_table.h"
#include "algorithms/model.h"
#include "services/daal_defines.h"
#include "algorithms/svm/svm_predict_types.h"
#include "src/algorithms/kernel.h"
#include "src/algorithms/kernel_function/kernel_function_types.h"
#include "src/algorithms/kernel_function/kernel_function_iface_impl.h"

// Typedef to avoid namespace conflicts with local 'internal'
namespace daal_kf_internal = ::daal::algorithms::kernel_function::internal;

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace prediction
{
namespace internal
{
template <Method method, typename algorithmFPType, CpuType cpu>
struct SVMPredictImpl : public Kernel
{
    services::Status compute(const data_management::NumericTablePtr & xTable, Model * model, data_management::NumericTable & r,
                             const svm::Parameter * par);

    services::Status computeSequential(const data_management::NumericTablePtr & xTable, const data_management::NumericTablePtr & svCoeffTable,
                                       const data_management::NumericTablePtr & svTable, data_management::NumericTable & r,
                                       services::SharedPtr<daal_kf_internal::KernelIfaceImpl> & kernel, const algorithmFPType bias, const size_t nVectors, const size_t nSV,
                                       const bool isSparse);

    services::Status computeThreading(const data_management::NumericTablePtr & xTable, const data_management::NumericTablePtr & svCoeffTable,
                                      const data_management::NumericTablePtr & svTable, data_management::NumericTable & r,
                                      services::SharedPtr<daal_kf_internal::KernelIfaceImpl> & kernel, const algorithmFPType bias, const size_t nVectors, const size_t nSV,
                                      const bool isSparse, const size_t nRowsPerBlock, const size_t nBlocks);
};

} // namespace internal
} // namespace prediction
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
