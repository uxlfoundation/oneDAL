/* file: pca_dense_correlation_base_iface.h */
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
//  Interface of functions needed to compute dense correlation-based pca
//--
*/

#ifndef __PCA_DENSE_CORRELATION_BASE_IFACE_H__
#define __PCA_DENSE_CORRELATION_BASE_IFACE_H__

#include "src/algorithms/pca/pca_dense_base_iface.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{
template <typename algorithmFPType>
class PCACorrelationBaseIface
{
public:
    virtual services::Status computeCorrelationEigenvalues(const data_management::NumericTable & correlation,
                                                           data_management::NumericTable & eigenvectors,
                                                           data_management::NumericTable & eigenvalues) = 0;
    virtual services::Status signFlipEigenvectors(NumericTable & eigenvectors) const                    = 0;
    virtual services::Status fillTable(NumericTable & table, algorithmFPType val) const                 = 0;
};

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
