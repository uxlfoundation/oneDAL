/* file: svd_dense_default_distr_step2_fpt_dispatcher.cpp */
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
//  Implementation of svd calculation algorithm container.
//--
*/

#include "src/algorithms/svd/svd_dense_default_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(svd::DistributedContainer, distributed, step2Master, DAAL_FPTYPE, svd::defaultDense)
namespace svd
{
namespace interface1
{
using DistributedType = Distributed<step2Master, DAAL_FPTYPE, svd::defaultDense>;

template <>
DAAL_EXPORT DistributedType::Distributed()
{
    initialize();
}

template <>
DAAL_EXPORT DistributedType::Distributed(const DistributedType & other) : input(other.input), parameter(other.parameter)
{
    initialize();
}

} // namespace interface1
} // namespace svd
} // namespace algorithms
} // namespace daal
