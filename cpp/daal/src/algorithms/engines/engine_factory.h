#ifndef __ENGINE_FACTORY_H__
#define __ENGINE_FACTORY_H__

#include "algorithms/engines/engine_backend.h"
#include "src/algorithms/engines/engine.h"
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

inline EnginePtr createEngine(engine_type type, size_t seed = 777)
{
    switch (type)
    {
    case engine_type::mt2203:   return mt2203::Batch<>::create(seed);
    case engine_type::mcg59:    return mcg59::Batch<>::create(seed);
    case engine_type::mrg32k3a: return mrg32k3a::Batch<>::create(seed);
    case engine_type::philox4x32x10: return philox4x32x10::Batch<>::create(seed);
    case engine_type::mt19937:
    default:                    return mt19937::Batch<>::create(seed);
    }
}

} // namespace engines
} // namespace algorithms
} // namespace daal
#endif
