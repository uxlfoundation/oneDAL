/* file: low_order_moments_csr_sum_batch_fpt_dispatcher.cpp */
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
//  Instantiation of batch low order moments algorithm container.
//--
*/

#include "src/algorithms/low_order_moments/low_order_moments_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(low_order_moments::BatchContainer, batch, DAAL_FPTYPE, low_order_moments::sumCSR)
namespace low_order_moments
{
namespace interface1
{
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, low_order_moments::sumCSR>::Batch()
{
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, low_order_moments::sumCSR>;

template <>
DAAL_EXPORT BatchType::Batch(const BatchType & other) : BatchImpl(other)
{
    initialize();
}

} // namespace interface1
} // namespace low_order_moments
} // namespace algorithms
} // namespace daal
