/* file: adaboost_predict_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of Ada Boost prediction algorithm container --
//  a class that contains Fast Ada Boost kernels for supported architectures.
//--
*/

#include "algorithms/boosting/adaboost_predict.h"
#include "src/algorithms/adaboost/adaboost_predict_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(adaboost::prediction::BatchContainer, batch, DAAL_FPTYPE, adaboost::prediction::defaultDense)
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(adaboost::prediction::BatchContainer, batch, DAAL_FPTYPE, adaboost::prediction::sammeR)

namespace adaboost
{
namespace prediction
{
namespace interface2
{
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, adaboost::prediction::defaultDense>::Batch(size_t nClasses)
{
    _par                 = new ParameterType(nClasses);
    parameter().nClasses = nClasses;
    initialize();
}

using BatchTypeDefault = Batch<DAAL_FPTYPE, adaboost::prediction::defaultDense>;
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, adaboost::prediction::defaultDense>::Batch(const BatchTypeDefault & other)
    : classifier::prediction::Batch(other), input(other.input)
{
    _par = new ParameterType(other.parameter());
    initialize();
}
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, adaboost::prediction::sammeR>::Batch(size_t nClasses)
{
    _par                 = new ParameterType(nClasses);
    parameter().nClasses = nClasses;
    initialize();
}

using BatchTypeSammeR = Batch<DAAL_FPTYPE, adaboost::prediction::sammeR>;
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, adaboost::prediction::sammeR>::Batch(const BatchTypeSammeR & other)
    : classifier::prediction::Batch(other), input(other.input)
{
    _par = new ParameterType(other.parameter());
    initialize();
}

} // namespace interface2
} // namespace prediction
} // namespace adaboost

} // namespace algorithms
} // namespace daal
