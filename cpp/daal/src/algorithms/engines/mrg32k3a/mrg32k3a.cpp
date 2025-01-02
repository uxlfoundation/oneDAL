/* file: mrg32k3a.cpp */
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
//  Implementation of the MRG32k3a engine: a 32-bit combined multiple recursive generator
//  with two components of order 3, optimized for batch processing.
//--

#include "algorithms/engines/mrg32k3a/mrg32k3a.h"
#include "src/algorithms/engines/mrg32k3a/mrg32k3a_batch_impl.h"
#include "src/externals/service_dispatch.h"

namespace daal
{
namespace algorithms
{
namespace engines
{
namespace mrg32k3a
{
namespace interface1
{
using namespace daal::services;
using namespace mrg32k3a::internal;

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
} // namespace mrg32k3a
} // namespace engines
} // namespace algorithms
} // namespace daal
