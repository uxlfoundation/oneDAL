/* file: kmeans_init_csr_parallelplus_distr_step1_fpt_dispatcher.cpp */
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
//  Implementation of parallel K-means++ initialization method for K-means algorithm
//--
*/

#include "src/algorithms/kmeans/kmeans_init_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(kmeans::init::DistributedContainer, distributed, step1Local, DAAL_FPTYPE, kmeans::init::parallelPlusCSR)

namespace kmeans
{
namespace init
{
namespace interface2
{
using DistributedType = Distributed<step1Local, DAAL_FPTYPE, kmeans::init::parallelPlusCSR>;

template <>
DAAL_EXPORT DistributedType::Distributed(size_t nClusters, size_t nRowsTotal, size_t offset)
    : DistributedBase(new ParameterType(nClusters, offset)), parameter(*static_cast<ParameterType *>(_par))
{
    initialize();
    parameter.nRowsTotal = nRowsTotal;
}

template <>
DAAL_EXPORT DistributedType::Distributed(const DistributedType & other)
    : DistributedBase(new ParameterType(other.parameter)), parameter(*static_cast<ParameterType *>(_par))
{
    initialize();
    input.set(data, other.input.get(data));
}

} // namespace interface2
} // namespace init
} // namespace kmeans

} // namespace algorithms
} // namespace daal
