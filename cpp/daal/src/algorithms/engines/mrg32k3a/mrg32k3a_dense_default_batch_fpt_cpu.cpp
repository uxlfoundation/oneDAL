/* file: mrg32k3a_dense_default_batch_fpt_cpu.cpp */
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
//  Implementation of mrg32k3a calculation functions.
//--

#include "src/algorithms/engines/mrg32k3a/mrg32k3a_batch_container.h"
#include "src/algorithms/engines/mrg32k3a/mrg32k3a_kernel.h"
#include "src/algorithms/engines/mrg32k3a/mrg32k3a_impl.i"

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
template class BatchContainer<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
} // namespace interface1

namespace internal
{
template class mrg32k3aKernel<DAAL_FPTYPE, defaultDense, DAAL_CPU>;
} // namespace internal

} // namespace mrg32k3a
} // namespace engines
} // namespace algorithms
} // namespace daal
