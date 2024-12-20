/* file: philox4x32x10_dense_default_batch_fpt_cpu.cpp */
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
//  Implementation of philox4x32x10 calculation functions.
//--

#include "src/algorithms/engines/philox4x32x10/philox4x32x10_batch_container.h"
#include "src/algorithms/engines/philox4x32x10/philox4x32x10_kernel.h"
#include "src/algorithms/engines/philox4x32x10/philox4x32x10_impl.i"

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
template class BatchContainer<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
} // namespace interface1

namespace internal
{
template class philox4x32x10Kernel<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
} // namespace internal

} // namespace philox4x32x10
} // namespace engines
} // namespace algorithms
} // namespace daal
