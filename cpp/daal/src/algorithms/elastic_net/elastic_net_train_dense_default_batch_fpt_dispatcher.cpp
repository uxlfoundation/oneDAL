/* file: elastic_net_train_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of elastic net container.
//--
*/

#include "src/algorithms/elastic_net/elastic_net_train_container.h"

#include "src/services/daal_strings.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(elastic_net::training::BatchContainer, batch, DAAL_FPTYPE, elastic_net::training::defaultDense)

namespace elastic_net
{
namespace training
{
namespace interface1
{
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, elastic_net::training::defaultDense>::Batch(const optimization_solver::iterative_solver::BatchPtr & solver)
{
    _par = new ParameterType(solver);
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, elastic_net::training::defaultDense>;
template <>
DAAL_EXPORT BatchType::Batch(const BatchType & other) : input(other.input)
{
    _par = new ParameterType(other.parameter());
    initialize();
}

} // namespace interface1
} // namespace training
} // namespace elastic_net
} // namespace algorithms
} // namespace daal
