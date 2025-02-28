/* file: pca_batchparameter_correlation_fpt.cpp */
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
//  Implementation of PCA algorithm interface.
//--
*/

#include "algorithms/pca/pca_types.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace interface3
{
/** Constructs PCA parameters */
template <typename algorithmFPType>
BatchParameter<algorithmFPType, correlationDense>::BatchParameter(const services::SharedPtr<covariance::BatchImpl> & covariance)
    : covariance(covariance) {};

template <typename algorithmFPType>
services::Status BatchParameter<algorithmFPType, correlationDense>::check() const
{
    DAAL_CHECK(covariance, services::ErrorNullAuxiliaryAlgorithm);
    return services::Status();
}

template DAAL_EXPORT BatchParameter<DAAL_FPTYPE, correlationDense>::BatchParameter(const services::SharedPtr<covariance::BatchImpl> & covariance);
template DAAL_EXPORT services::Status BatchParameter<DAAL_FPTYPE, correlationDense>::check() const;
} // namespace interface3
} // namespace pca
} // namespace algorithms
} // namespace daal
