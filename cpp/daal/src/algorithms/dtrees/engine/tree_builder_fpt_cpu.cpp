/* file: tree_builder_fpt_cpu.cpp */
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
//  Instantiation of the tree builder for each CpuType, floating-point type and record layout.
//--
*/

#include "src/algorithms/dtrees/engine/tree_builder.i"

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

#define DTREES_ENGINE_INSTANTIATE_TRAIN_TREES(Record)                                                                                        \
    template services::Status trainTrees<DAAL_FPTYPE, Record, DAAL_CPU>(const TreeData<DAAL_FPTYPE, DAAL_CPU> &, const TreeParams &, size_t, \
                                                                        TreeInputs<Record, DAAL_CPU> &);

DTREES_ENGINE_INSTANTIATE_TRAIN_TREES(ClsCountRecord)
DTREES_ENGINE_INSTANTIATE_TRAIN_TREES(ClsWeightRecord<DAAL_FPTYPE>)
DTREES_ENGINE_INSTANTIATE_TRAIN_TREES(RegCountRecord<DAAL_FPTYPE>)
DTREES_ENGINE_INSTANTIATE_TRAIN_TREES(RegWeightRecord<DAAL_FPTYPE>)

#undef DTREES_ENGINE_INSTANTIATE_TRAIN_TREES
} // namespace internal
} // namespace engine
} // namespace dtrees
} // namespace algorithms
} // namespace daal
