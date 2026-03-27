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
//  Precision levels:
//    highest (default) -- always compute in float32. Full IEEE-754, bit-exact.
//    high              -- allow reduced-precision kernels (e.g. AMX BF16) where
//                         oneDAL determines the accuracy impact is acceptable.
//                         The library, not the user, decides which operations
//                         are eligible. Currently: float32 Euclidean GEMM,
//                         dims >= 64, hardware with AMX-BF16.
//
//  Hardware availability is checked ONCE at process initialisation (g_hw_bf16).
//  The setter stores the user's requested level as-is; BF16GemmDispatcher
//  gates on both g_hw_bf16 and the stored precision level.
//  This way the getter is a plain atomic load with no HW queries.
//--
*/

#include "services/cpu_type.h"
#include "src/services/service_defines.h"

#include <atomic>
#include <cstdlib>  /* std::getenv */
#include <cstring>  /* std::strcmp */

namespace
{

/// Hardware capability flag — evaluated ONCE at process start.
/// True iff AMX-BF16 is present and OS-enabled (CPUID + XCR0 check via
/// daal_serv_cpu_feature_detect(), which is also cached internally).
static const bool g_hw_amx_bf16 =
    (daal_serv_cpu_feature_detect() & daal::CpuFeature::amx_bf16) != 0;

/// Parse ONEDAL_FLOAT32_MATMUL_PRECISION env var.
/// Recognised values: "HIGH" / "high".  Everything else → highest.
static daal::Float32MatmulPrecision parse_env_precision()
{
    const char * val = std::getenv("ONEDAL_FLOAT32_MATMUL_PRECISION");
    if (val && (std::strcmp(val, "HIGH") == 0 || std::strcmp(val, "high") == 0))
    {
        return daal::Float32MatmulPrecision::high;
    }
    return daal::Float32MatmulPrecision::highest;
}

/// Stored precision level (what the user requested).
/// Default: read from env at startup; can be overridden at runtime.
/// BF16GemmDispatcher combines this with g_hw_amx_bf16 to make the
/// final dispatch decision.
static std::atomic<int> g_float32_matmul_precision{
    static_cast<int>(parse_env_precision())
};

} // anonymous namespace

/// Return whether AMX-BF16 hardware is available in this process.
/// Result is constant for the lifetime of the process.
DAAL_EXPORT bool daal_has_amx_bf16()
{
    return g_hw_amx_bf16;
}

/// Return the currently requested float32 matmul precision.
/// This is a plain atomic load — no hardware queries.
DAAL_EXPORT daal::Float32MatmulPrecision daal_get_float32_matmul_precision()
{
    return static_cast<daal::Float32MatmulPrecision>(
        g_float32_matmul_precision.load(std::memory_order_relaxed));
}

/// Set the float32 matmul precision hint.
/// Stores the requested level directly; does NOT re-check hardware.
/// If 'high' is requested but hardware lacks AMX-BF16, BF16GemmDispatcher
/// will fall through to sgemm transparently via its own g_use_bf16_gemm gate.
DAAL_EXPORT void daal_set_float32_matmul_precision(daal::Float32MatmulPrecision p)
{
    g_float32_matmul_precision.store(static_cast<int>(p), std::memory_order_relaxed);
}
