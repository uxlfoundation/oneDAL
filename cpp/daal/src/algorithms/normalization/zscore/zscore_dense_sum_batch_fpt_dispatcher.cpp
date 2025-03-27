/* file: zscore_dense_sum_batch_fpt_dispatcher.cpp */
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

//++
//  Implementation of zscore normalization algorithm container.
//
//--

#include "src/algorithms/normalization/zscore/zscore_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(normalization::zscore::interface3::BatchContainer, batch, DAAL_FPTYPE, normalization::zscore::sumDense)
} // namespace algorithms
} // namespace daal

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace zscore
{
namespace interface3
{
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, normalization::zscore::sumDense>::Batch()
{
    _par = new ParameterType();
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, normalization::zscore::sumDense>;

template <>
DAAL_EXPORT BatchType::Batch(const Batch & other) : BatchImpl(other)
{
    _par = new ParameterType(other.parameter());
    initialize();
}

} // namespace interface3
} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal
