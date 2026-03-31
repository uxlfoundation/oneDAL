/* file: engine_factory.cpp */
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
//  Implementation of engine factory function.
//--
*/

#include "algorithms/engines/engine.h"
#include "src/algorithms/engines/mt19937/mt19937.h"
#include "src/algorithms/engines/mt2203/mt2203.h"
#include "src/algorithms/engines/mcg59/mcg59.h"
#include "src/algorithms/engines/mrg32k3a/mrg32k3a.h"
#include "src/algorithms/engines/philox4x32x10/philox4x32x10.h"

namespace daal
{
namespace algorithms
{
namespace engines
{
namespace interface1
{
template <typename algorithmFPType>
DAAL_EXPORT EngineIfacePtr createEngine(EngineType type, size_t seed)
{
    switch (type)
    {
    case mt19937Engine: return mt19937::internal::Batch<algorithmFPType>::create(seed);
    case mt2203Engine: return mt2203::internal::Batch<algorithmFPType>::create(seed);
    case mcg59Engine: return mcg59::internal::Batch<algorithmFPType>::create(seed);
    case mrg32k3aEngine: return mrg32k3a::internal::Batch<algorithmFPType>::create(seed);
    case philox4x32x10Engine: return philox4x32x10::internal::Batch<algorithmFPType>::create(seed);
    default: return EngineIfacePtr();
    }
}

// Explicit instantiations
template DAAL_EXPORT EngineIfacePtr createEngine<float>(EngineType type, size_t seed);
template DAAL_EXPORT EngineIfacePtr createEngine<double>(EngineType type, size_t seed);

} // namespace interface1
} // namespace engines
} // namespace algorithms
} // namespace daal
