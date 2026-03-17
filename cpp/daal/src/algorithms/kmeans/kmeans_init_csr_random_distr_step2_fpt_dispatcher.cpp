/* file: kmeans_init_csr_random_distr_step2_fpt_dispatcher.cpp */
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
//  Implementation of K-means initialization random algorithm container for CSR
//--
*/

#include "src/algorithms/kmeans/kmeans_init_container.h"
#include "src/algorithms/algorithm_dispatch_container_common.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(kmeans::init::internal::DistributedContainer, distributed, step2Master, DAAL_FPTYPE, kmeans::init::randomCSR)

namespace kmeans
{
namespace init
{
namespace interface2
{
using DistributedType = Distributed<step2Master, DAAL_FPTYPE, kmeans::init::randomCSR>;

template <>
void DistributedType::initialize()
{
    Analysis<distributed>::_ac =
        new __DAAL_ALGORITHM_CONTAINER(distributed, internal::DistributedContainer, step2Master, DAAL_FPTYPE, kmeans::init::randomCSR)(&_env);
    _in = &input;
}
template <>
DAAL_EXPORT DistributedType::Distributed(size_t nClusters, size_t offset)
    : DistributedBase(new ParameterType(nClusters, offset)), parameter(*static_cast<ParameterType *>(_par))
{
    initialize();
}

} // namespace interface2
} // namespace init
} // namespace kmeans

} // namespace algorithms
} // namespace daal
