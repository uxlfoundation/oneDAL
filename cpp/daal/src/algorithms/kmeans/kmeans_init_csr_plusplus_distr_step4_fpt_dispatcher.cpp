/* file: kmeans_init_csr_plusplus_distr_step4_fpt_dispatcher.cpp */
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
//  Implementation of K-means plus plus initialization method for K-means algorithm
//--
*/

#include "src/algorithms/kmeans/kmeans_init_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(kmeans::init::DistributedContainer, distributed, step4Local, DAAL_FPTYPE, kmeans::init::plusPlusCSR)

namespace kmeans
{
namespace init
{
namespace interface2
{
using DistributedType = Distributed<step4Local, DAAL_FPTYPE, kmeans::init::plusPlusCSR>;

template <>
DAAL_EXPORT DistributedType::Distributed(size_t nClusters)
    : DistributedBase(new ParameterType(nClusters)), parameter(*static_cast<ParameterType *>(_par))
{
    initialize();
}

template <>
DAAL_EXPORT DistributedType::Distributed(const DistributedType & other)
    : DistributedBase(new ParameterType(other.parameter)), parameter(*static_cast<ParameterType *>(_par)), input(other.input)
{
    initialize();
}

} // namespace interface2
} // namespace init
} // namespace kmeans

} // namespace algorithms
} // namespace daal
