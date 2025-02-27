/* file: linear_regression_train_dense_normeq_online_fpt_dispatcher.cpp */
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
//  Implementation of linear regression container.
//--
*/

#include "src/algorithms/linear_regression/linear_regression_train_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(linear_regression::training::OnlineContainer, online, DAAL_FPTYPE, linear_regression::training::normEqDense)
namespace linear_regression
{
namespace training
{
namespace interface1
{
template <>
DAAL_EXPORT Online<DAAL_FPTYPE, normEqDense>::Online()
{
    initialize();
}

using OnlineType = Online<DAAL_FPTYPE, normEqDense>;

template <>
DAAL_EXPORT OnlineType::Online(const OnlineType & other) : linear_model::training::Online(other), input(other.input), parameter(other.parameter)
{
    initialize();
}

} // namespace interface1
} // namespace training
} // namespace linear_regression
} // namespace algorithms
} // namespace daal
