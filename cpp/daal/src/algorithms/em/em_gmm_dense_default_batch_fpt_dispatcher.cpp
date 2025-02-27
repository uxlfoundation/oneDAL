/* file: em_gmm_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of EM calculation algorithm container.
//--
*/

#include "src/algorithms/em/em_gmm_dense_default_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(em_gmm::BatchContainer, batch, DAAL_FPTYPE, em_gmm::defaultDense)
namespace em_gmm
{
namespace interface1
{

template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, em_gmm::defaultDense>::Batch(const size_t nComponents)
    : parameter(nComponents, services::SharedPtr<covariance::Batch<DAAL_FPTYPE, covariance::defaultDense> >(
                                 new covariance::Batch<DAAL_FPTYPE, covariance::defaultDense>()))
{
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, em_gmm::defaultDense>;

template <>
DAAL_EXPORT BatchType::Batch(const BatchType & other) : input(other.input), parameter(other.parameter)
{
    initialize();
}

} // namespace interface1
} // namespace em_gmm
} // namespace algorithms
} // namespace daal
