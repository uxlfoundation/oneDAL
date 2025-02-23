/* file: philox4x32x10_dense_default_batch_fpt_dispatcher.cpp */
/*******************************************************************************
* Copyright contributors to the oneDAL project
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
//  Implementation of philox4x32x10 calculation algorithm dispatcher.
//--

#include "src/algorithms/engines/philox4x32x10/philox4x32x10_batch_container.h"

namespace daal
{
namespace algorithms
{
__DAAL_INSTANTIATE_DISPATCH_CONTAINER(engines::philox4x32x10::BatchContainer, batch, DAAL_FPTYPE, engines::philox4x32x10::defaultDense)
namespace engines
{
namespace philox4x32x10
{
namespace interface1
{
template <typename algorithmFPType, Method method>
DAAL_EXPORT Batch<algorithmFPType, method>::Batch(size_t seed)
{
    initialize();
}

template <typename algorithmFPType, Method method>
DAAL_EXPORT Batch<algorithmFPType, method>::Batch(const Batch<algorithmFPType, method> & other) : super(other)
{
    initialize();
}

template class Batch<DAAL_FPTYPE, engines::philox4x32x10::defaultDense>;
} // namespace interface1
} // namespace philox4x32x10
} // namespace engines
} // namespace algorithms
} // namespace daal
