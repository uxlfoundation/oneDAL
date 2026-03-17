/* file: kmeans_init_csr_deterministic_batch_fpt_dispatcher.cpp */
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
//  Implementation of K-means initialization algorithm container for CSR
//--
*/

#include "src/algorithms/kmeans/kmeans_init_container.h"
#include "src/algorithms/algorithm_dispatch_container_batch.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(kmeans::init::internal::BatchContainer, batch, DAAL_FPTYPE, kmeans::init::deterministicCSR)

namespace kmeans
{
namespace init
{
namespace interface2
{
template <>
void Batch<DAAL_FPTYPE, kmeans::init::deterministicCSR>::initialize()
{
    Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, internal::BatchContainer, DAAL_FPTYPE, kmeans::init::deterministicCSR)(&_env);
    _in                  = &input;
}
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, kmeans::init::deterministicCSR>::Batch(size_t nClasses)
    : BatchBase(new ParameterType(nClasses)), parameter(*static_cast<ParameterType *>(_par))
{
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, kmeans::init::deterministicCSR>;
template <>
DAAL_EXPORT BatchType::Batch(const BatchType & other)
    : BatchBase(new ParameterType(other.parameter)), parameter(*static_cast<ParameterType *>(_par)), input(other.input)
{
    initialize();
}

} // namespace interface2
} // namespace init
} // namespace kmeans

} // namespace algorithms
} // namespace daal
