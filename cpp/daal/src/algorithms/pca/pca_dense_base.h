/* file: pca_dense_base.h */
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

#ifndef __PCA_DENSE_BASE_H__
#define __PCA_DENSE_BASE_H__

#include "src/algorithms/kernel.h"
#include "src/algorithms/pca/pca_dense_base_iface.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
class PCADenseBase : public Kernel
{
public:
    services::Status signFlipEigenvectors(NumericTable & eigenvectors) const;
    services::Status fillTable(NumericTable & table, algorithmFPType val) const;
    services::Status copyTable(NumericTable & source, NumericTable & dest) const;
    services::Status computeExplainedVariancesRatio(const data_management::NumericTable & eigenvalues,
                                                    const data_management::NumericTable & variances,
                                                    data_management::NumericTable & explainedVariancesRatio);
    services::Status computeNoiseVariances(const data_management::NumericTable & eigenvalues, const data_management::NumericTable & variances,
                                           double & noiseVariance);

private:
    void signFlipArray(size_t size, algorithmFPType * source) const;
};

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif // __PCA_DENSE_BASE_H__
