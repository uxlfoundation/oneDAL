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
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(ridge_regression::training::OnlineContainer, online, DAAL_FPTYPE, ridge_regression::training::normEqDense)
namespace ridge_regression
{
namespace training
{
namespace interface1
{
template <typename algorithmFPType, Method method>
DAAL_EXPORT Online<algorithmFPType, method>::Online()
{
    initialize();
}

template <typename algorithmFPType, Method method>
DAAL_EXPORT Online<algorithmFPType, method>::Online(const Online<algorithmFPType, method> & other)
    : linear_model::training::Online(other), input(other.input), parameter(other.parameter)
{
    initialize();
}
template class Online<DAAL_FPTYPE, ridge_regression::training::normEqDense>;
} // namespace interface1
} // namespace training
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal
