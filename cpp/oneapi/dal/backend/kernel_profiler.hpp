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

#include <cstring>

#include "oneapi/dal/detail/profiler.hpp"
#include "src/services/service_defines.h"

namespace oneapi::dal::backend {

/// The default step of a kernel, which adds nothing to its task name.
struct kernel_step_default {
    static constexpr const char* suffix = "";
};

/// The compiler's spelling of this function, which contains the type `Kernel`.
template <typename Kernel>
inline const char* kernel_signature() {
    return PRETTY_FUNCTION;
}

/// The profiler task name of the kernel type `Kernel`: the same address for the same type and step.
/// The signature spells the type up to a top-level `,`, `;` or `]`, or the closing `>` on MSVC.
template <typename Kernel, typename Step = kernel_step_default>
inline const char* kernel_task_name() {
    const char* const signature = kernel_signature<Kernel>();
#ifdef _MSC_VER
    const char* const marker = "kernel_signature<";
#else
    const char* const marker = "Kernel = ";
#endif
    const char* const found = std::strstr(signature, marker);
    const char* const begin = found ? found + std::strlen(marker) : signature;
    const char* end = begin;
    for (int depth = 0; *end; ++end) {
        if (depth == 0 && (*end == ',' || *end == ';' || *end == ']' || *end == '>')) {
            break;
        }
        depth += (*end == '<') - (*end == '>');
    }
    return daal::internal::profiler_kernel_task_name(begin, std::size_t(end - begin), Step::suffix);
}

using kernel_task_name_t = const char* (*)();

/// Starts the task named by `name`; out of line, so a disabled profiler leaves only a load
/// and a branch at the call site.
DAAL_FORCENOINLINE inline daal::internal::profiler_task start_named_kernel_task(
    kernel_task_name_t name) {
    return daal::internal::profiler::start_task(name());
}

/// Starts a profiler task named after the kernel type `Kernel` and its step `Step`.
template <typename Kernel, typename Step = kernel_step_default>
inline daal::internal::profiler_task start_kernel_task() {
    if (DAAL_PROFILER_UNLIKELY(daal::internal::is_profiler_enabled())) {
        return start_named_kernel_task(&kernel_task_name<Kernel, Step>);
    }
    return daal::internal::profiler_task();
}

#ifdef ONEDAL_DATA_PARALLEL
/// Starts the task named by `name` on `queue`, out of line for the same reason.
DAAL_FORCENOINLINE inline detail::queue_sync_guard start_named_kernel_task(
    sycl::queue& queue,
    kernel_task_name_t name) {
    return detail::queue_sync_guard::make(queue, name(), true);
}

/// Starts a profiler task named after the kernel type `Kernel` that covers the device work
/// it submits to `queue`.
template <typename Kernel>
inline detail::queue_sync_guard start_kernel_task(sycl::queue& queue) {
    if (DAAL_PROFILER_UNLIKELY(daal::internal::is_profiler_enabled())) {
        return start_named_kernel_task(queue, &kernel_task_name<Kernel>);
    }
    return detail::queue_sync_guard::make(queue, nullptr, false);
}
#endif

} // namespace oneapi::dal::backend
