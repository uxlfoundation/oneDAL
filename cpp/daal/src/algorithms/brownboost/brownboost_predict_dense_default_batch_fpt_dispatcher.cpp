/* file: brownboost_predict_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of Brown Boost prediction algorithm container --
//  a class that contains Fast Brown Boost kernels for supported architectures.
//--
*/

#include "algorithms/boosting/brownboost_predict.h"
#include "src/algorithms/brownboost/brownboost_predict_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(brownboost::prediction::BatchContainer, batch, DAAL_FPTYPE, brownboost::prediction::defaultDense)

namespace brownboost
{
namespace prediction
{
namespace interface2
{
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, brownboost::prediction::defaultDense>::Batch()
{
    _par = new ParameterType();
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, brownboost::prediction::defaultDense>;
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, brownboost::prediction::defaultDense>::Batch(const BatchType & other)
    : classifier::prediction::Batch(other), input(other.input)
{
    _par = new ParameterType(other.parameter());
    initialize();
}
} // namespace interface2
} // namespace prediction
} // namespace brownboost

} // namespace algorithms
} // namespace daal
