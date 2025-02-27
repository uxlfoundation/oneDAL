/* file: implicit_als_train_init_csr_default_distr_step2_fpt_dispatcher.cpp */
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
//  Implementation of implicit ALS initialization algorithm container
//  for distributed computing mode.
//--
*/

#include "src/algorithms/implicit_als/implicit_als_train_init_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(implicit_als::training::init::DistributedContainer, distributed, step2Local, DAAL_FPTYPE,
                                      implicit_als::training::init::fastCSR)
namespace implicit_als
{
namespace training
{
namespace init
{
namespace interface1
{
using DistributedType = Distributed<step2Local, DAAL_FPTYPE, implicit_als::training::init::fastCSR>;

template <>
DAAL_EXPORT DistributedType::Distributed()
{
    initialize();
}

template <>
DAAL_EXPORT DistributedType::Distributed(const DistributedType & other)
{
    initialize();
    input.set(inputOfStep2FromStep1, other.input.get(inputOfStep2FromStep1));
}

} // namespace interface1
} // namespace init
} // namespace training
} // namespace implicit_als
} // namespace algorithms
} // namespace daal
