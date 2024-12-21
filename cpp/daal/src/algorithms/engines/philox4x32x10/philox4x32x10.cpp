/* file: philox4x32x10.cpp */
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
//  Implementation of the Philox4x32-10 engine: a counter-based pseudorandom number generator (PRNG)
//  that uses 4x32-bit keys and performs 10 rounds of mixing to produce high-quality randomness.
//--

#include "algorithms/engines/philox4x32x10/philox4x32x10.h"
#include "src/externals/service_dispatch.h"
#include "src/algorithms/engines/philox4x32x10/philox4x32x10_batch_impl.h"

namespace daal
{
namespace algorithms
{
namespace engines
{
namespace philox4x32x10
{
namespace interface1
{
using namespace daal::services;
using namespace philox4x32x10::internal;

template <typename algorithmFPType, Method method>
SharedPtr<Batch<algorithmFPType, method> > Batch<algorithmFPType, method>::create(size_t seed)
{
    SharedPtr<Batch<algorithmFPType, method> > engPtr;
#define DAAL_CREATE_ENGINE_CPU(cpuId, ...) engPtr.reset(new BatchImpl<cpuId, algorithmFPType, method>(__VA_ARGS__));

    DAAL_DISPATCH_FUNCTION_BY_CPU(DAAL_CREATE_ENGINE_CPU, seed);

#undef DAAL_CREATE_ENGINE_CPU
    return engPtr;
}

template class Batch<double, defaultDense>;
template class Batch<float, defaultDense>;

} // namespace interface1
} // namespace philox4x32x10
} // namespace engines
} // namespace algorithms
} // namespace daal
