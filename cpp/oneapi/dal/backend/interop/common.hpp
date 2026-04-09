/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <daal/include/services/env_detect.h>
#include <daal/src/services/cpu_type.h>

#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::backend::interop {

template <typename DispatchId>
struct to_daal_cpu_type;

template <daal::internal::CpuType cpu>
struct daal_cpu_value {
    constexpr static daal::internal::CpuType value = cpu;
};

#if defined(TARGET_X86_64)
template <>
struct to_daal_cpu_type<cpu_dispatch_default> : daal_cpu_value<daal::internal::sse2> {};
template <>
struct to_daal_cpu_type<cpu_dispatch_avx2> : daal_cpu_value<daal::internal::avx2> {};
template <>
struct to_daal_cpu_type<cpu_dispatch_avx512> : daal_cpu_value<daal::internal::avx512> {};

#elif defined(TARGET_ARM)
template <>
struct to_daal_cpu_type<cpu_dispatch_sve> : daal_cpu_value<daal::internal::sve> {};

#elif defined(TARGET_RISCV64)
template <>
struct to_daal_cpu_type<cpu_dispatch_rv64> : daal_cpu_value<daal::internal::rv64> {};

#endif

template <typename Float,
          template <typename, daal::internal::CpuType>
          typename CpuKernel,
          typename... Args>
inline auto call_daal_kernel(const context_cpu& ctx, Args&&... args) {
    return dal::backend::dispatch_by_cpu(ctx, [&](auto cpu) {
        return CpuKernel<Float, to_daal_cpu_type<decltype(cpu)>::value>().compute(
            std::forward<Args>(args)...);
    });
}

template <typename Float,
          template <typename, daal::internal::CpuType>
          typename CpuKernel,
          typename... Args>
inline auto call_daal_kernel_finalize_merge(const context_cpu& ctx, Args&&... args) {
    return dal::backend::dispatch_by_cpu(ctx, [&](auto cpu) {
        return CpuKernel<Float, to_daal_cpu_type<decltype(cpu)>::value>().finalizeMerge(
            std::forward<Args>(args)...);
    });
}

template <typename Float,
          template <typename, daal::internal::CpuType>
          typename CpuKernel,
          typename... Args>
inline auto call_daal_kernel_finalize_compute(const context_cpu& ctx, Args&&... args) {
    return dal::backend::dispatch_by_cpu(ctx, [&](auto cpu) {
        return CpuKernel<Float, to_daal_cpu_type<decltype(cpu)>::value>().finalizeCompute(
            std::forward<Args>(args)...);
    });
}

} // namespace oneapi::dal::backend::interop
