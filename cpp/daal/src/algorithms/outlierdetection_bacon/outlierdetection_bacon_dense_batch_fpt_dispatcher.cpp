/* file: outlierdetection_bacon_dense_batch_fpt_dispatcher.cpp */
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
//  Implementation of container for multivariate Bacon outlier detection.
//--
*/

#include "src/algorithms/outlierdetection_bacon/outlierdetection_bacon_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(bacon_outlier_detection::BatchContainer, batch, DAAL_FPTYPE, bacon_outlier_detection::defaultDense)
namespace bacon_outlier_detection
{
namespace interface1
{
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, bacon_outlier_detection::defaultDense>::Batch()
{
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, bacon_outlier_detection::defaultDense>;

template <>
DAAL_EXPORT BatchType::Batch(const BatchType & other) : input(other.input), parameter(other.parameter)
{
    initialize();
}

} // namespace interface1
} // namespace bacon_outlier_detection
} // namespace algorithms
} // namespace daal
