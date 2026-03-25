/* file: dbscan_dense_default_distr_step8_fpt_dispatcher.cpp */
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
//  Implementation of DBSCAN algorithm container for distributed
//  computing mode.
//--
*/

#include "src/algorithms/dbscan/dbscan_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(dbscan::internal::DistributedContainer, distributed, step8Local, DAAL_FPTYPE, dbscan::defaultDense)

namespace dbscan
{
namespace interface1
{
using DistributedType = Distributed<step8Local, DAAL_FPTYPE, defaultDense>;

template <>
void Distributed<step8Local, DAAL_FPTYPE, dbscan::defaultDense>::initialize()
{
    Analysis<distributed>::_ac =
        new __DAAL_ALGORITHM_CONTAINER(distributed, internal::DistributedContainer, step8Local, DAAL_FPTYPE, dbscan::defaultDense)(&_env);
    _in = &input;
    _partialResult.reset(new PartialResultType());
}
template <>
DAAL_EXPORT DistributedType::Distributed(size_t blockIndex, size_t nBlocks)
{
    ParameterType * par = new ParameterType();
    par->blockIndex     = blockIndex;
    par->nBlocks        = nBlocks;

    _par = par;
    initialize();
}

template <>
DAAL_EXPORT DistributedType::Distributed(const DistributedType & other) : input(other.input)
{
    _par = new ParameterType(other.parameter());
    initialize();
}

} // namespace interface1
} // namespace dbscan
} // namespace algorithms
} // namespace daal
