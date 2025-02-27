/* file: dbscan_dense_default_distr_step4_fpt_dispatcher.cpp */
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
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(dbscan::DistributedContainer, distributed, step4Local, DAAL_FPTYPE, dbscan::defaultDense)

namespace dbscan
{
namespace interface1
{
using DistributedType = Distributed<step4Local, DAAL_FPTYPE, defaultDense>;

template <>
DAAL_EXPORT DistributedType::Distributed(size_t leftBlocks, size_t rightBlocks)
{
    ParameterType * par = new ParameterType();
    par->leftBlocks     = leftBlocks;
    par->rightBlocks    = rightBlocks;

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
