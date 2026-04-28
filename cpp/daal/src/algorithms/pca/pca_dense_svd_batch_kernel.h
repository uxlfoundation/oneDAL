/* file: pca_dense_svd_batch_kernel.h */
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
//  Declaration of template structs that calculate PCA SVD.
//--
*/

#ifndef __PCA_DENSE_SVD_BATCH_KERNEL_H__
#define __PCA_DENSE_SVD_BATCH_KERNEL_H__

#include "algorithms/pca/pca_batch.h"
#include "algorithms/pca/pca_types.h"
#include "src/algorithms/svd/svd_dense_default_kernel.h"

#include "src/algorithms/pca/pca_dense_svd_base.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{
using namespace daal::internal;

template <typename algorithmFPType, typename ParameterType, CpuType cpu>
class PCASVDBatchKernel : public PCASVDKernelBase<algorithmFPType, cpu>
{
public:
    PCASVDBatchKernel() {};
    using PCASVDKernelBase<algorithmFPType, cpu>::computeEigenValues;

    using PCADenseBase<algorithmFPType, cpu>::computeExplainedVariancesRatio;

    services::Status compute(InputDataType type, const data_management::NumericTablePtr & data, data_management::NumericTable & eigenvalues,
                             data_management::NumericTable & eigenvectors);

    services::Status compute(InputDataType type, data_management::NumericTable & data, const ParameterType * parameter,
                             data_management::NumericTable & eigenvalues, data_management::NumericTable & eigenvectors,
                             data_management::NumericTable & means, data_management::NumericTable & variances);

    services::Status compute(InputDataType type, data_management::NumericTable & data, data_management::NumericTable & eigenvectors,
                             data_management::NumericTable & singular_values, data_management::NumericTable & means,
                             data_management::NumericTable & variances, data_management::NumericTable * eigenvalues,
                             data_management::NumericTable * explained_variances_ratio, const ParameterType * parameter);

protected:
    services::Status normalizeDataset(const data_management::NumericTablePtr & data, data_management::NumericTablePtr & normalizedData);

    services::Status decompose(const NumericTable * normalizedDataTable, data_management::NumericTable & eigenvalues,
                               data_management::NumericTable & eigenvectors);
};

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal
#endif
