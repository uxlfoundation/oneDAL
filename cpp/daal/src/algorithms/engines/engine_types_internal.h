/* file: engine_types_internal.h */
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
//  Implementation of initializer types.
//--

#ifndef __ENGINE_TYPES_INTERNAL_H__
#define __ENGINE_TYPES_INTERNAL_H__

#include "src/algorithms/engines/engine_batch_impl.h"
#include "algorithms/engines/engine_family.h"
#include "src/services/service_arrays.h"

namespace daal
{
namespace algorithms
{
namespace engines
{
namespace internal
{
using namespace daal::internal;
template <CpuType cpu>
struct Params
{
    size_t numberOfStreams;
    services::internal::TArray<size_t, cpu> nSkip;

    Params(size_t numStreams) : numberOfStreams(numStreams), nSkip(numStreams) {}
};

template <CpuType cpu>
class EnginesCollection
{
private:
    engines::FamilyEnginePtr _clonedEngine;
    ParallelizationTechnique _technique;
    size_t _numberOfStreams;

public:
    EnginesCollection(engines::EnginePtr engine, ParallelizationTechnique technique, Params<cpu> & params,
                      services::internal::TArray<engines::EnginePtr, cpu> & engines, services::Status * st)
    {
        *st = initEngines(engine, technique, params, engines);
    }

    engines::EnginePtr getUpdatedEngine(engines::EnginePtr engine, services::internal::TArray<engines::EnginePtr, cpu> & engines,
                                        services::internal::TArray<size_t, cpu> & numElems)
    {
        switch (_technique)
        {
        case skipahead:
        {
            return engines[_numberOfStreams - 1]->clone();
        }
        case leapfrog:
        {
            size_t nSkip = 0;
            for (size_t i = 0; i < _numberOfStreams; i++)
            {
                if (nSkip < numElems[i])
                {
                    nSkip = numElems[i];
                }
            }
            auto updatedEngine = engine->clone();
            updatedEngine->skipAhead(nSkip);
            return updatedEngine;
        }
        case family: return _clonedEngine;
        }
        return engines::EnginePtr();
    }

    services::Status initEngines(engines::EnginePtr engine, ParallelizationTechnique technique, Params<cpu> & params,
                                 services::internal::TArray<engines::EnginePtr, cpu> & engines)
    {
        _technique       = technique;
        _numberOfStreams = params.numberOfStreams;

        auto engineImpl = dynamic_cast<engines::internal::BatchBaseImpl *>(engine.get());
        DAAL_CHECK(engineImpl, ErrorEngineNotSupported);
        DAAL_CHECK(engineImpl->hasSupport(_technique), ErrorEngineNotSupported);
        switch (_technique)
        {
        case skipahead:
        {
            for (size_t i = 0; i < _numberOfStreams; i++)
            {
                auto engineLocal = engine->clone();
                DAAL_CHECK_STATUS_VAR(engineLocal->skipAhead(params.nSkip[i]));
                engines[i] = engineLocal;
            }
            break;
        }
        case leapfrog:
        {
            for (size_t i = 0; i < _numberOfStreams; i++)
            {
                auto engineLocal = engine->clone();
                DAAL_CHECK_STATUS_VAR(engineLocal->leapfrog(i, params.numberOfStreams));
                engines[i] = engineLocal;
            }
            break;
        }
        case family:
        {
            _clonedEngine = services::dynamicPointerCast<engines::FamilyBatchBase>(engine->clone());
            DAAL_CHECK(_clonedEngine, ErrorEngineNotSupported);
            DAAL_CHECK(_clonedEngine->getMaxNumberOfStreams() >= _numberOfStreams, ErrorEngineNotSupported);
            size_t numStreams = _clonedEngine->getNumberOfStreams();
            if (numStreams < _numberOfStreams)
            {
                DAAL_CHECK_STATUS_VAR(_clonedEngine->add(_numberOfStreams - numStreams)); // silently initialize more independent streams
            }
            for (size_t i = 0; i < _numberOfStreams; i++)
            {
                engines[i] = _clonedEngine->get(i);
            }
        }
        }
        return services::Status();
    }
};

} // namespace internal
} // namespace engines
} // namespace algorithms
} // namespace daal

#endif // __ENGINE_TYPES_INTERNAL_H__
