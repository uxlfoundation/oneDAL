/* file: stump_regression_predict_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of Decision Stump prediction algorithm container --
//  a class that contains Fast Decision Stump kernels for supported architectures.
//--
*/

#include "src/algorithms/stump/stump_regression_predict_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(stump::regression::prediction::BatchContainer, batch, DAAL_FPTYPE, stump::regression::prediction::defaultDense)

namespace stump
{
namespace regression
{
namespace prediction
{
namespace interface1
{
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, stump::regression::prediction::defaultDense>::Batch()
{
    _par = new ParameterType();
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, stump::regression::prediction::defaultDense>;

template <>
DAAL_EXPORT BatchType::Batch(const Batch & other) : algorithms::regression::prediction::Batch(other), input(other.input)
{
    _par = new ParameterType(other.parameter());
    initialize();
}

} // namespace interface1
} // namespace prediction
} // namespace regression
} // namespace stump
} // namespace algorithms
} // namespace daal
