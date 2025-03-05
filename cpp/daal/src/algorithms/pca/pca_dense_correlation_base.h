/* file: pca_dense_correlation_base.h */
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

#ifndef __PCA_DENSE_CORRELATION_BASE_H__
#define __PCA_DENSE_CORRELATION_BASE_H__

#include "algorithms/pca/pca_types.h"
#include "src/algorithms/pca/pca_dense_base.h"
#include "src/algorithms/pca/pca_dense_correlation_base_iface.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
class PCACorrelationBase : public PCACorrelationBaseIface<algorithmFPType>, public PCADenseBase<algorithmFPType, cpu>
{
public:
    explicit PCACorrelationBase() {};

protected:
    services::Status computeCorrelationEigenvalues(const data_management::NumericTable & correlation, data_management::NumericTable & eigenvectors,
                                                   data_management::NumericTable & eigenvalues) DAAL_C11_OVERRIDE;
    services::Status computeEigenvectorsInplace(size_t nFeatures, algorithmFPType * eigenvectors, algorithmFPType * eigenvalues);
    services::Status sortEigenvectorsDescending(size_t nFeatures, algorithmFPType * eigenvectors, algorithmFPType * eigenvalues);
    services::Status computeSingularValues(const data_management::NumericTable & eigenvalues, data_management::NumericTable & variances,
                                           size_t nRows);
    services::Status computeVariancesFromCov(const data_management::NumericTable & correlation, data_management::NumericTable & variances);
    services::Status signFlipEigenvectors(NumericTable & eigenvectors) const DAAL_C11_OVERRIDE;
    services::Status fillTable(NumericTable & table, algorithmFPType val) const DAAL_C11_OVERRIDE;
    services::Status copyVarianceFromCovarianceTable(NumericTable & source, NumericTable & dest) const;
    services::Status correlationFromCovarianceTable(NumericTable & source) const;

private:
    void copyArray(size_t size, const algorithmFPType * source, algorithmFPType * destination);
};

template <ComputeMode mode, typename algorithmFPType, CpuType cpu>
class PCACorrelationKernel : public PCACorrelationBase<algorithmFPType, cpu>
{};

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
