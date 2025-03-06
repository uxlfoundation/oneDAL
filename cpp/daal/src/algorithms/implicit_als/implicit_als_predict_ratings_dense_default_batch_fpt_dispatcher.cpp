/* file: implicit_als_predict_ratings_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of implicit ALS prediction algorithm container.
//--
*/

#include "src/algorithms/kernel.h"
#include "algorithms/implicit_als/implicit_als_predict_ratings_batch.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(implicit_als::prediction::ratings::BatchContainer, batch, DAAL_FPTYPE,
                                      implicit_als::prediction::ratings::defaultDense)

namespace implicit_als
{
namespace prediction
{
namespace ratings
{
namespace interface1
{
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, implicit_als::prediction::ratings::defaultDense>::Batch()
{
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, implicit_als::prediction::ratings::defaultDense>;

template <>
DAAL_EXPORT BatchType::Batch(const BatchType & other) : input(other.input), parameter(other.parameter)
{
    initialize();
}

} // namespace interface1
} // namespace ratings
} // namespace prediction
} // namespace implicit_als
} // namespace algorithms
} // namespace daal
