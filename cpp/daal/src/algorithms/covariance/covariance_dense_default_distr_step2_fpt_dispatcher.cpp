/* file: covariance_dense_default_distr_step2_fpt_dispatcher.cpp */
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
//  Instantiation of distributed covariance calculation algorithm container.
//--
*/

#include "src/algorithms/covariance/covariance_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(covariance::DistributedContainer, distributed, step2Master, DAAL_FPTYPE, covariance::defaultDense)
namespace covariance
{
namespace interface1
{
template <typename algorithmFPType, Method method>
DAAL_EXPORT Distributed<step2Master, algorithmFPType, method>::Distributed()
{
    initialize();
}

template <typename algorithmFPType, Method method>
DAAL_EXPORT Distributed<step2Master, algorithmFPType, method>::Distributed(const Distributed<step2Master, algorithmFPType, method> & other)
    : DistributedIface<step2Master>(other)
{
    initialize();
}

template class Distributed<step2Master, DAAL_FPTYPE, covariance::defaultDense>;
} // namespace interface1
} // namespace covariance
} // namespace algorithms
} // namespace daal