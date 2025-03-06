/* file: kmeans_init_dense_random_distr_step2_fpt_dispatcher.cpp */
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
//  Implementation of K-means initialization random algorithm container
//--
*/

#include "src/algorithms/kmeans/kmeans_init_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(kmeans::init::DistributedContainer, distributed, step2Master, DAAL_FPTYPE, kmeans::init::randomDense)

namespace kmeans
{
namespace init
{
namespace interface2
{
using DistributedType = Distributed<step2Master, DAAL_FPTYPE, kmeans::init::randomDense>;

template <>
DAAL_EXPORT DistributedType::Distributed(size_t nClusters, size_t offset)
    : DistributedBase(new ParameterType(nClusters, offset)), parameter(*static_cast<ParameterType *>(_par))
{
    Analysis<distributed>::_ac =
        new __DAAL_ALGORITHM_CONTAINER(distributed, DistributedContainer, step2Master, DAAL_FPTYPE, kmeans::init::randomDense)(&_env);
    _in = &input;
}

} // namespace interface2
} // namespace init
} // namespace kmeans

} // namespace algorithms
} // namespace daal
