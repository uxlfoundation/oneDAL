/* file: env_detect_precision.cpp */
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
//  Float32 matmul precision control.
//
//  Inspired by torch.set_float32_matmul_precision / jax_default_matmul_precision.
//
//  The *effective* precision folds in hardware availability:
//  requesting Float32MatmulPrecision::high on hardware without AMX-BF16
//  silently returns Float32MatmulPrecision::highest so call-sites need no
//  hardware checks of their own.
//--
*/

#include "services/cpu_type.h"
#include "src/services/service_defines.h"

#include <atomic>
#include <cstdlib>  /* std::getenv */
#include <cstring>  /* std::strcmp */

namespace
{

/// Read ONEDAL_FLOAT32_MATMUL_PRECISION env var once at startup.
/// Returns Float32MatmulPrecision::highest if unset or unrecognised.
static daal::Float32MatmulPrecision parse_env_precision()
{
    const char * val = std::getenv("ONEDAL_FLOAT32_MATMUL_PRECISION");
    if (val && (std::strcmp(val, "HIGH") == 0 || std::strcmp(val, "high") == 0))
    {
        return daal::Float32MatmulPrecision::high;
    }
    return daal::Float32MatmulPrecision::highest;
}

/// Fold hardware availability into the requested precision level.
/// If the user requested 'high' but HW doesn't support AMX-BF16, downgrade
/// to 'highest' so call-sites never need to query hardware themselves.
static daal::Float32MatmulPrecision compute_effective_precision(daal::Float32MatmulPrecision requested)
{
    if (requested == daal::Float32MatmulPrecision::highest)
    {
        return daal::Float32MatmulPrecision::highest;
    }
    // 'high' requested: check AMX-BF16 availability once.
    const bool has_amx_bf16 = (daal_serv_cpu_feature_detect() & daal::CpuFeature::amx_bf16) != 0;
    return has_amx_bf16 ? daal::Float32MatmulPrecision::high : daal::Float32MatmulPrecision::highest;
}

/// Global effective precision. Initialised once; can be overridden at runtime
/// via daal_set_float32_matmul_precision().
static std::atomic<int> g_float32_matmul_precision{ static_cast<int>(
    compute_effective_precision(parse_env_precision())) };

} // anonymous namespace

DAAL_EXPORT daal::Float32MatmulPrecision daal_get_float32_matmul_precision()
{
    return static_cast<daal::Float32MatmulPrecision>(g_float32_matmul_precision.load(std::memory_order_relaxed));
}

DAAL_EXPORT void daal_set_float32_matmul_precision(daal::Float32MatmulPrecision p)
{
    const daal::Float32MatmulPrecision effective = compute_effective_precision(p);
    g_float32_matmul_precision.store(static_cast<int>(effective), std::memory_order_relaxed);
}
