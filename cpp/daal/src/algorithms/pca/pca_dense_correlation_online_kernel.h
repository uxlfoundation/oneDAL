/* file: pca_dense_correlation_online_kernel.h */
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
//  Declaration of template structs that calculate PCA Correlation.
//--
*/

#ifndef __PCA_DENSE_CORRELATION_ONLINE_KERNEL_H__
#define __PCA_DENSE_CORRELATION_ONLINE_KERNEL_H__

#include "algorithms/pca/pca_types.h"
#include "src/services/service_defines.h"
#include "services/error_handling.h"
#include "src/algorithms/pca/pca_dense_correlation_base.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
class PCACorrelationKernel<online, algorithmFPType, cpu> : public PCACorrelationBase<algorithmFPType, cpu>
{
public:
    explicit PCACorrelationKernel() {};

    using PCACorrelationBase<algorithmFPType, cpu>::computeCorrelationEigenvalues;

    using PCACorrelationBase<algorithmFPType, cpu>::computeSingularValues;

    using PCACorrelationBase<algorithmFPType, cpu>::computeVariancesFromCov;

    using PCADenseBase<algorithmFPType, cpu>::computeExplainedVariancesRatio;

    using PCADenseBase<algorithmFPType, cpu>::signFlipEigenvectors;

    services::Status compute(const data_management::NumericTablePtr & pData, PartialResult<correlationDense> * partialResult,
                             const OnlineParameter<algorithmFPType, correlationDense> * parameter);

    services::Status finalize(PartialResult<correlationDense> * partialResult, const OnlineParameter<algorithmFPType, correlationDense> * parameter,
                              data_management::NumericTable & eigenvectors, data_management::NumericTable & eigenvalues);

private:
    services::Status copyIfNeeded(data_management::NumericTable * src, data_management::NumericTable * dst);
    services::Status copyCovarianceResultToPartialResult(covariance::PartialResult * covariancePres, PartialResult<correlationDense> * partialResult);
};

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
