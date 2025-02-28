/* file: gbt_classification_predict_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of gradient boosted trees algorithm container -- a class
//  that contains fast gradient boosted trees prediction kernels
//  for supported architectures.
//--
*/

#include "src/algorithms/dtrees/gbt/classification/gbt_classification_predict_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(gbt::classification::prediction::BatchContainer, batch, DAAL_FPTYPE,
                                      gbt::classification::prediction::defaultDense)

namespace gbt
{
namespace classification
{
namespace prediction
{
namespace interface2
{
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, gbt::classification::prediction::defaultDense>::Batch(size_t nClasses)
{
    _par = new ParameterType(nClasses);
    initialize();
};

using BatchType = Batch<DAAL_FPTYPE, gbt::classification::prediction::defaultDense>;
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, gbt::classification::prediction::defaultDense>::Batch(const BatchType & other)
    : classifier::prediction::Batch(other), input(other.input)
{
    _par = new ParameterType(other.parameter());
    initialize();
}
} // namespace interface2
} // namespace prediction
} // namespace classification
} // namespace gbt

} // namespace algorithms
} // namespace daal
