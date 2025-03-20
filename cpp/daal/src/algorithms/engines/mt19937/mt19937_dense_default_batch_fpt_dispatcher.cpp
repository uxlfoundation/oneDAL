/* file: mt19937_dense_default_batch_fpt_dispatcher.cpp */
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
//  Implementation of mt19937 calculation algorithm dispatcher.
//--

#include "src/algorithms/engines/mt19937/mt19937_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(engines::mt19937::BatchContainer, batch, DAAL_FPTYPE, engines::mt19937::defaultDense)
namespace engines
{
namespace mt19937
{
namespace interface1
{
template <>
DAAL_EXPORT Batch<DAAL_FPTYPE, engines::mt19937::defaultDense>::Batch(size_t seed)
{
    initialize();
}

using BatchType = Batch<DAAL_FPTYPE, engines::mt19937::defaultDense>;

template <>
DAAL_EXPORT BatchType::Batch(const BatchType & other) : super(other)
{
    initialize();
}

} // namespace interface1
} // namespace mt19937
} // namespace engines
} // namespace algorithms
} // namespace daal
