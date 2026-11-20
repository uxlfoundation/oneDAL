/* file: quantile_binner_fpt_cpu.cpp */
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

/*
//++
//  Instantiation of the quantile binner for each CpuType and floating-point type.
//--
*/

#include "src/algorithms/dtrees/engine/quantile_binner.i"

namespace daal
{
namespace algorithms
{
namespace dtrees
{
namespace engine
{
namespace internal
{
using namespace daal::internal; // DAAL_CPU names a CpuType value

template class QuantileBinner<DAAL_FPTYPE, DAAL_CPU>;
} // namespace internal
} // namespace engine
} // namespace dtrees
} // namespace algorithms
} // namespace daal
