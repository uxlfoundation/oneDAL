/* file: df_regression_train_hist_batch_fpt_dispatcher.cpp */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  Implementation of decision forest container for the hist method.
//--
*/

#include "src/algorithms/dtrees/forest/regression/df_regression_train_container.h"
#include "src/algorithms/algorithm_dispatch_container_batch.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(decision_forest::regression::training::internal::BatchContainer, batch, DAAL_FPTYPE,
                                      decision_forest::regression::training::hist)
namespace decision_forest
{
namespace regression
{
namespace training
{
namespace interface2
{
using BatchType = Batch<DAAL_FPTYPE, decision_forest::regression::training::hist>;

template <>
void Batch<DAAL_FPTYPE, decision_forest::regression::training::hist>::initialize()
{
    _ac = new __DAAL_ALGORITHM_CONTAINER(batch, internal::BatchContainer, DAAL_FPTYPE, decision_forest::regression::training::hist)(&_env);
    _in = &input;
    _result.reset(new ResultType());
}
template <>
DAAL_EXPORT BatchType::Batch()
{
    _par = new ParameterType();
    initialize();
    parameter().minObservationsInLeafNode = 5;
}

template <>
DAAL_EXPORT BatchType::Batch(const BatchType & other) : input(other.input)
{
    _par = new ParameterType(other.parameter());
    initialize();
}
} // namespace interface2
} // namespace training
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
