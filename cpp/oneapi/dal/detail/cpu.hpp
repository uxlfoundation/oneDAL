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

#pragma once

#include <cstdint>
#include "oneapi/dal/common.hpp"

#include <map>
#include <string>

// TODO: Clean up this redefinition and import the defines globally.
#if defined(__x86_64__) || defined(__x86_64) || defined(__amd64) || defined(_M_AMD64)
#define TARGET_X86_64
#endif

#if defined(__ARM_ARCH) || defined(__aarch64__)
#define TARGET_ARM
#endif

#if defined(__riscv) && (__riscv_xlen == 64)
#define TARGET_RISCV64
#endif

namespace oneapi::dal::detail {
namespace v1 {

/// CPU vendor enumeration.
enum class cpu_vendor { unknown = 0, intel = 1, amd = 2, arm = 3, riscv64 = 4 };

/// CPU extension enumeration.
/// This enum is used to represent the highest supported CPU extension.
enum class cpu_extension : uint64_t {
    none = 0U
#if defined(TARGET_X86_64)
    ,
    sse2 = 1U << 0, /// Intel(R) Streaming SIMD Extensions 2 (Intel(R) SSE2)
    sse42 = 1U << 2, /// Intel(R) Streaming SIMD Extensions 4.2 (Intel(R) SSE4.2)
    avx2 = 1U << 4, /// Intel(R) Advanced Vector Extensions 2 (Intel(R) AVX2)
    avx512 =
        1U
        << 5 /// Intel(R) Xeon(R) processors based on Intel(R) Advanced Vector Extensions 512 (Intel(R) AVX-512)
#elif defined(TARGET_ARM)
    ,
    sve = 1U << 0 /// Arm(R) processors based on Arm's Scalable Vector Extension (SVE)
#elif defined(TARGET_RISCV64)
    ,
    rv64 = 1U << 0 /// RISC-V 64-bit architecture
#endif
};

enum class cpu_feature : uint64_t {
    unknown = 0ULL
#if defined(TARGET_X86_64)
    ,
    sstep = 1ULL << 0, /// Intel(R) SpeedStep
    tb = 1ULL << 1, /// Intel(R) Turbo Boost
    avx512_bf16 = 1ULL << 2, /// AVX512 bfloat16
    avx512_vnni = 1ULL << 3, /// AVX512 VNNI
    tb3 = 1ULL << 4 /// Intel(R) Turbo Boost Max 3.0
#endif
};

/// A map of CPU features to their string representations.
/// This map is used to convert CPU feature bitmasks to human-readable strings.
/// Keys are bitflags representing CPU features. They are defined in daal::CpuFeature enumeration.
inline const std::map<uint64_t, const std::string> cpu_feature_map = {
    { uint64_t(cpu_feature::unknown), "Unknown" }
#if defined(TARGET_X86_64)
    ,
    { uint64_t(cpu_feature::sstep), "Intel(R) SpeedStep" },
    { uint64_t(cpu_feature::tb), "Intel(R) Turbo Boost" },
    { uint64_t(cpu_feature::avx512_bf16), "AVX-512 bfloat16" },
    { uint64_t(cpu_feature::avx512_vnni), "AVX-512 VNNI" },
    { uint64_t(cpu_feature::tb3), "Intel(R) Turbo Boost Max 3.0" }
#endif
};

/// Converts a DAAL CPU extension value to oneDAL enumeration.
/// @param ext The DAAL CPU extension value.
/// @return The corresponding oneDAL CPU extension value.
ONEDAL_EXPORT cpu_extension from_daal_cpu_type(int);

/// Detects the highest supported CPU extension.
/// @return The corresponding oneDAL CPU extension value.
ONEDAL_EXPORT cpu_extension detect_top_cpu_extension();

/// Detects the highest CPU extension used by oneDAL.
/// If REQCPU was used, it might be different from the one returned by detect_top_cpu_extension.
/// @return The corresponding oneDAL CPU extension value.
ONEDAL_EXPORT cpu_extension detect_onedal_cpu_extension();

/// Detects the CPU features.
/// @return Bitmask representing the supported CPU features.
/// @note The bitmask is a combination of the CPU feature bitflags defined in daal::CpuFeature enumeration.
uint64_t detect_cpu_features();

} // namespace v1
using v1::cpu_vendor;
using v1::cpu_extension;
using v1::cpu_feature_map;
using v1::detect_top_cpu_extension;
using v1::detect_onedal_cpu_extension;
using v1::detect_cpu_features;
} // namespace oneapi::dal::detail
