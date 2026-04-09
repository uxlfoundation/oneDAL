/* file: ridge_regression_train_dense_normeq_online_fpt_dispatcher.cpp */
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
//  Implementation of ridge regression container.
//--
*/

#include "src/algorithms/ridge_regression/ridge_regression_train_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(ridge_regression::training::internal::OnlineContainer, online, DAAL_FPTYPE,
                                      ridge_regression::training::normEqDense)
namespace ridge_regression
{
namespace training
{
namespace interface1
{
template <>
void Online<DAAL_FPTYPE, ridge_regression::training::normEqDense>::initialize()
{
    _ac  = new __DAAL_ALGORITHM_CONTAINER(online, internal::OnlineContainer, DAAL_FPTYPE, ridge_regression::training::normEqDense)(&_env);
    _in  = &input;
    _par = &parameter;
    _partialResult.reset(new PartialResultType());
    _result.reset(new ResultType());
}
template <>
DAAL_EXPORT Online<DAAL_FPTYPE, ridge_regression::training::normEqDense>::Online()
{
    initialize();
}

using OnlineType = Online<DAAL_FPTYPE, ridge_regression::training::normEqDense>;

template <>
DAAL_EXPORT OnlineType::Online(const OnlineType & other) : linear_model::training::Online(other), input(other.input), parameter(other.parameter)
{
    initialize();
}

} // namespace interface1
} // namespace training
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal
