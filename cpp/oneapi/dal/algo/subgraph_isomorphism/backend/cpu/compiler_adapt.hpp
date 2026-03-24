/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include <daal/include/services/daal_defines.h>

#if defined(TARGET_X86_64)
#include <immintrin.h>
#endif

#include <daal/src/services/service_defines.h>
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::backend {

#if defined(__INTEL_COMPILER)
#define ONEDAL_IVDEP         _Pragma("ivdep")
#define ONEDAL_VECTOR_ALWAYS _Pragma("vector always")
#else
#define ONEDAL_IVDEP
#define ONEDAL_VECTOR_ALWAYS
#endif

template <typename Cpu>
ONEDAL_FORCEINLINE std::int32_t ONEDAL_lzcnt_u32(std::uint32_t a) {
#if defined(__AVX2__) && defined(__INTEL_COMPILER)
    return _lzcnt_u32(a);
#else
    if (a == 0)
        return 32;
    std::uint32_t one_bit = 0x80000000; // binary: 1000 0000 0000 0000 0000 0000 0000 0000
    std::int32_t bit_pos = 0;
    while ((a & one_bit) == 0) {
        bit_pos++;
        one_bit >>= 1;
    }
    return bit_pos;
#endif
}

template <typename Cpu>
ONEDAL_FORCEINLINE std::int32_t ONEDAL_lzcnt_u64(std::uint64_t a) {
#if defined(__AVX2__) && defined(__INTEL_COMPILER)
    return _lzcnt_u64(a);
#else
    if (a == 0)
        return 64;
    std::uint64_t one_bit = 0x8000000000000000; // binary: 1000 ... 0000
    std::int32_t bit_pos = 0;
    while ((a & one_bit) == 0) {
        bit_pos++;
        one_bit >>= 1;
    }
    return bit_pos;
#endif
}

template <typename Cpu>
ONEDAL_FORCEINLINE std::int32_t ONEDAL_popcnt64(std::uint64_t a) {
#if defined(__AVX2__) && defined(__INTEL_COMPILER)
    return _popcnt64(a);
#else
    if (a == 0)
        return 0;
    std::uint64_t last_bit = 1;
    std::int32_t bit_cnt = 0;
    for (std::int32_t i = 0; i < 64; i++) {
        if (a & last_bit) {
            bit_cnt++;
        }
        a = a >> 1;
    }
    return bit_cnt;
#endif
}

#if defined(TARGET_X86_64)
template <>
ONEDAL_FORCEINLINE std::int32_t ONEDAL_lzcnt_u32<dal::backend::cpu_dispatch_sse2>(std::uint32_t a) {
    if (a == 0)
        return 32;
    std::uint32_t one_bit = 0x80000000; // binary: 1000 0000 0000 0000 0000 0000 0000 0000
    std::int32_t bit_pos = 0;
    while ((a & one_bit) == 0) {
        bit_pos++;
        one_bit >>= 1;
    }
    return bit_pos;
}

template <>
ONEDAL_FORCEINLINE std::int32_t ONEDAL_lzcnt_u64<dal::backend::cpu_dispatch_sse2>(std::uint64_t a) {
    if (a == 0)
        return 64;
    std::uint64_t one_bit = 0x8000000000000000; // binary: 1000 ... 0000
    std::int32_t bit_pos = 0;
    while ((a & one_bit) == 0) {
        bit_pos++;
        one_bit >>= 1;
    }
    return bit_pos;
}

template <>
ONEDAL_FORCEINLINE std::int32_t ONEDAL_popcnt64<dal::backend::cpu_dispatch_sse2>(std::uint64_t a) {
    if (a == 0)
        return 0;
    std::uint64_t last_bit = 1;
    std::int32_t bit_cnt = 0;
    for (std::int32_t i = 0; i < 64; i++) {
        if (a & last_bit) {
            bit_cnt++;
        }
        a = a >> 1;
    }
    return bit_cnt;
}

template <>
ONEDAL_FORCEINLINE std::int32_t ONEDAL_lzcnt_u32<dal::backend::cpu_dispatch_avx2>(std::uint32_t a) {
    if (a == 0)
        return 32;
    std::uint32_t one_bit = 0x80000000; // binary: 1000 0000 0000 0000 0000 0000 0000 0000
    std::int32_t bit_pos = 0;
    while ((a & one_bit) == 0) {
        bit_pos++;
        one_bit >>= 1;
    }
    return bit_pos;
}

template <>
ONEDAL_FORCEINLINE std::int32_t ONEDAL_lzcnt_u64<dal::backend::cpu_dispatch_avx2>(std::uint64_t a) {
    if (a == 0)
        return 64;
    std::uint64_t one_bit = 0x8000000000000000; // binary: 1000 ... 0000
    std::int32_t bit_pos = 0;
    while ((a & one_bit) == 0) {
        bit_pos++;
        one_bit >>= 1;
    }
    return bit_pos;
}

template <>
ONEDAL_FORCEINLINE std::int32_t ONEDAL_popcnt64<dal::backend::cpu_dispatch_avx2>(std::uint64_t a) {
    if (a == 0)
        return 0;
    std::uint64_t last_bit = 1;
    std::int32_t bit_cnt = 0;
    for (std::int32_t i = 0; i < 64; i++) {
        if (a & last_bit) {
            bit_cnt++;
        }
        a = a >> 1;
    }
    return bit_cnt;
}
#elif defined(TARGET_ARM)
template <>
ONEDAL_FORCEINLINE std::int32_t ONEDAL_lzcnt_u32<dal::backend::cpu_dispatch_sve>(std::uint32_t a) {
    return __builtin_clz(a);
}

template <>
ONEDAL_FORCEINLINE std::int32_t ONEDAL_lzcnt_u64<dal::backend::cpu_dispatch_sve>(std::uint64_t a) {
    return __builtin_clzl(a);
}

template <>
ONEDAL_FORCEINLINE std::int32_t ONEDAL_popcnt64<dal::backend::cpu_dispatch_sve>(std::uint64_t a) {
    return __builtin_popcountl(a);
}
#elif defined(TARGET_RISCV64)
template <>
ONEDAL_FORCEINLINE std::int32_t ONEDAL_lzcnt_u32<dal::backend::cpu_dispatch_rv64>(std::uint32_t a) {
    return __builtin_clz(a);
}

template <>
ONEDAL_FORCEINLINE std::int32_t ONEDAL_lzcnt_u64<dal::backend::cpu_dispatch_rv64>(std::uint64_t a) {
    return __builtin_clzl(a);
}

template <>
ONEDAL_FORCEINLINE std::int32_t ONEDAL_popcnt64<dal::backend::cpu_dispatch_rv64>(std::uint64_t a) {
    return __builtin_popcountl(a);
}
#endif
} // namespace oneapi::dal::preview::subgraph_isomorphism::backend
