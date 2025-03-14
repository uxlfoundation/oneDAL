/* file: dbscan_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of DBSCAN container.
//--
*/

#include "src/algorithms/dbscan/dbscan_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(dbscan::BatchContainer, batch, DAAL_FPTYPE, dbscan::defaultDense)

namespace dbscan
{
namespace interface1
{
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, dbscan::defaultDense>::Batch(DAAL_FPTYPE epsilon, size_t minObservations)
{
    _par = new ParameterType(epsilon, minObservations);
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, dbscan::defaultDense>;
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, dbscan::defaultDense>::Batch(const BatchType & other) : input(other.input)
{
    _par = new ParameterType(other.parameter());
    initialize();
}

} // namespace interface1
} // namespace dbscan
} // namespace algorithms
} // namespace daal
