/* file: linear_regression_single_beta_dense_default_batch_fpt_dispatcher.cpp */
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
//  Instantiation of the container for single beta quality metrics.
//--
*/

#include "src/algorithms/linear_regression/linear_regression_single_beta_dense_default_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(linear_regression::quality_metric::single_beta::BatchContainer, batch, DAAL_FPTYPE,
                                      linear_regression::quality_metric::single_beta::defaultDense)
namespace linear_regression
{
namespace quality_metric
{
namespace single_beta
{
namespace interface1
{
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, defaultDense>::Batch()
{
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, defaultDense>;

template <>
DAAL_EXPORT BatchType::Batch(const BatchType & other) : parameter(other.parameter)
{
    initialize();
    input.set(expectedResponses, other.input.get(expectedResponses));
    input.set(predictedResponses, other.input.get(predictedResponses));
    input.set(model, other.input.get(model));
}

} // namespace interface1
} // namespace single_beta
} // namespace quality_metric
} // namespace linear_regression
} // namespace algorithms
} // namespace daal
