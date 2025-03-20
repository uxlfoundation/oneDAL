/* file: kmeans_dense_lloyd_distr_step2_fpt_dispatcher.cpp */
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
//  Implementation of K-means algorithm container -- a class that contains
//  Lloyd K-means kernels for supported architectures.
//--
*/

#include "src/algorithms/kmeans/kmeans_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(kmeans::interface2::DistributedContainer, distributed, step2Master, DAAL_FPTYPE, kmeans::lloydDense);

namespace kmeans
{
namespace interface2
{
using DistributedType = Distributed<step2Master, DAAL_FPTYPE, kmeans::lloydDense>;

template <>
DAAL_EXPORT DistributedType::Distributed(size_t nClusters, size_t nIterations)
{
    _par = new ParameterType(nClusters, nIterations);
    initialize();
    parameter().resultsToEvaluate &= ~computeAssignments;
}

template <>
DAAL_EXPORT DistributedType::Distributed(const DistributedType & other)
{
    _par = new ParameterType(other.parameter());
    initialize();
    input.set(partialResults, other.input.get(partialResults));
}

} // namespace interface2
} // namespace kmeans

} // namespace algorithms
} // namespace daal
