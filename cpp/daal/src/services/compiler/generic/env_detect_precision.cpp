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
//    strict (default) -- always compute in float32. Full IEEE-754, bit-exact.
//    allow_bf16       -- allow BF16 operand conversion for internal GEMM paths
//                        where oneDAL determines the accuracy impact is acceptable.
//                        Accumulation and output remain float32. The library,
//                        not the user, decides which operations are eligible.
//                        Currently: float32 Euclidean GEMM, dims >= 64, on
//                        hardware with AMX-BF16.
//    require_bf16     -- use BF16 operands for operations that have an implemented
//                        BF16 path and can execute it on the current hardware.
//                        Unsupported algorithms/hardware still use the regular path.
//
//  Hardware availability is queried independently from the stored precision hint.
//  The getter/setter operate only on the requested precision value. Dispatchers
//  combine the hint with runtime hardware capability checks.
//--
*/

#include "src/services/cpu_type.h"
#include "src/services/service_defines.h"

#include <atomic>
#include <cstdlib> /* std::getenv */
#include <cstring> /* std::strcmp */

namespace
{

/// Parse ONEDAL_FLOAT32_MATMUL_PRECISION env var.
/// Recognised values: "ALLOW_BF16" / "allow_bf16" and "REQUIRE_BF16" / "require_bf16".
/// Everything else → strict.
static daal::internal::Float32MatmulPrecision parse_env_precision()
{
    const char * val = std::getenv("ONEDAL_FLOAT32_MATMUL_PRECISION");
    if (val && (std::strcmp(val, "ALLOW_BF16") == 0 || std::strcmp(val, "allow_bf16") == 0))
    {
        return daal::internal::Float32MatmulPrecision::allow_bf16;
    }
    if (val && (std::strcmp(val, "REQUIRE_BF16") == 0 || std::strcmp(val, "require_bf16") == 0))
    {
        return daal::internal::Float32MatmulPrecision::require_bf16;
    }
    return daal::internal::Float32MatmulPrecision::strict;
}

/// Stored precision level (what the user requested).
/// Default: read from env at startup; can be overridden at runtime.
static std::atomic<int> g_float32_matmul_precision { static_cast<int>(parse_env_precision()) };

} // anonymous namespace

/// Return the currently requested float32 matmul precision.
/// This is a plain atomic load — no hardware queries.
DAAL_EXPORT daal::internal::Float32MatmulPrecision daal_get_float32_matmul_precision()
{
    return static_cast<daal::internal::Float32MatmulPrecision>(g_float32_matmul_precision.load(std::memory_order_relaxed));
}

/// Set the float32 matmul precision hint.
/// Stores the requested level directly; dispatchers combine it with runtime
/// hardware capability checks.
DAAL_EXPORT void daal_set_float32_matmul_precision(daal::internal::Float32MatmulPrecision p)
{
    g_float32_matmul_precision.store(static_cast<int>(p), std::memory_order_relaxed);
}
