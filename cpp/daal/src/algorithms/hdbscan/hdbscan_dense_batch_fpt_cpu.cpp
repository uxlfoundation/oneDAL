/* file: hdbscan_dense_batch_fpt_cpu.cpp */
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

#include "src/algorithms/hdbscan/hdbscan_dense_batch_impl.i"
#include "services/daal_defines.h"

using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace hdbscan
{
namespace internal
{

template class DAAL_EXPORT HDBSCANBatchKernel<DAAL_FPTYPE, defaultDense, DAAL_CPU>;

} // namespace internal
} // namespace hdbscan
} // namespace algorithms
} // namespace daal
