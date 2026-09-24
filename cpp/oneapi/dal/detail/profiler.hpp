/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "src/services/service_profiler.h"

#ifdef ONEDAL_DATA_PARALLEL
#include <sycl/sycl.hpp>
#endif

namespace oneapi::dal::detail {

#ifdef ONEDAL_DATA_PARALLEL
// RAII wrapper that synchronizes a SYCL queue around a profiled region so the recorded duration
// reflects device-side execution rather than just host-side enqueue.
//
// Ordering matters: the pre-region wait_and_throw() MUST happen before start_task() captures its
// timestamp, otherwise any work already in flight on the queue when the region begins would be
// charged to the region. The pre-region wait therefore lives in a static factory that runs before
// the profiler_task is constructed and passed to the guard's ctor.
//
// When `enabled` is false the guard is a no-op: no waits are issued and the passed profiler_task
// is a null one, so the profiler-disabled fast path pays only for a nullptr check on each side.
class queue_sync_guard {
public:
    static queue_sync_guard make(sycl::queue& q, const char* task_name, bool enabled) {
        if (enabled) {
            q.wait_and_throw();
            return queue_sync_guard(&q, daal::internal::profiler::start_task(task_name));
        }
        return queue_sync_guard(nullptr, daal::internal::profiler::start_task(nullptr));
    }

    ~queue_sync_guard() {
        if (queue_) {
#if (!defined(DAAL_NOTHROW_EXCEPTIONS))
            try {
#endif
                queue_->wait_and_throw();
#if (!defined(DAAL_NOTHROW_EXCEPTIONS))
            } catch (std::exception& e) {
                // Match existing profiler policy: swallow so a stray SYCL error during shutdown
                // does not terminate the process.
                std::cerr << e.what() << std::endl;
            } catch (...) {
            }
#endif
        }
    }

    queue_sync_guard(const queue_sync_guard&)            = delete;
    queue_sync_guard& operator=(const queue_sync_guard&) = delete;

    queue_sync_guard(queue_sync_guard&& other) noexcept
        : queue_(other.queue_), task_(std::move(other.task_)) {
        other.queue_ = nullptr;
    }

    queue_sync_guard& operator=(queue_sync_guard&&) = delete;

private:
    queue_sync_guard(sycl::queue* q, daal::internal::profiler_task&& task)
        : queue_(q), task_(std::move(task)) {}

    sycl::queue* queue_ = nullptr;
    daal::internal::profiler_task task_;
};
#endif

} // namespace oneapi::dal::detail

#define ONEDAL_PROFILER_CONCAT2(x, y) x##y
#define ONEDAL_PROFILER_CONCAT(x, y)  ONEDAL_PROFILER_CONCAT2(x, y)
#define ONEDAL_PROFILER_UNIQUE_ID     __LINE__

// UTILS
#define ONEDAL_PROFILER_MACRO_1(name)                       ONEDAL_PROFILER_START_TASK(name)
#define ONEDAL_PROFILER_MACRO_2(name, queue)                ONEDAL_PROFILER_START_TASK_WITH_QUEUE(name, queue)
#define ONEDAL_PROFILER_GET_MACRO(arg_1, arg_2, MACRO, ...) MACRO

// START_TASKS
// Each START_TASK* macro self-gates on `is_profiler_enabled()` so a single top-level TASK macro
// can dispatch by arg count without a wrapping ternary. Both branches of every ternary here
// return the same type, so `auto` at the call site picks up whichever type the arg count
// selected. When profiler is off, `start_task(nullptr)` returns a null profiler_task, and the
// queue variant leaves `queue_` null so no SYCL wait is ever issued.
#define ONEDAL_PROFILER_START_TASK(name)                                                            \
    (daal::internal::is_profiler_enabled() ? daal::internal::profiler::start_task(#name)            \
                                           : daal::internal::profiler::start_task(nullptr))

#ifdef ONEDAL_DATA_PARALLEL
    #define ONEDAL_PROFILER_START_TASK_WITH_QUEUE(name, queue)                                      \
        ::oneapi::dal::detail::queue_sync_guard::make((queue), #name,                               \
                                                      daal::internal::is_profiler_enabled())
#else
    // Non-SYCL translation unit that happens to include this header — fall back to plain
    // host-side timing. Semantically the same as the no-queue variant.
    #define ONEDAL_PROFILER_START_TASK_WITH_QUEUE(name, queue) ONEDAL_PROFILER_START_TASK(name)
#endif

#define ONEDAL_PROFILER_START_NULL_TASK() daal::internal::profiler::start_task(nullptr)

// PROFILER TASKS
// The variable name is uniquified via __LINE__ so multiple profiled scopes on adjacent lines in
// the same block coexist. `auto` accepts either a profiler_task or a queue_sync_guard depending
// on which START_TASK* macro the arg count selects. The logger side-effect body is guarded by
// `is_logger_enabled()`; both is_logger_enabled and is_profiler_enabled are cached-bool functions
// initialized once per process so calling them per macro is cheap.
#define ONEDAL_PROFILER_TASK_WITH_ARGS(task_name, ...)                                              \
    auto ONEDAL_PROFILER_CONCAT(__profiler_task__, ONEDAL_PROFILER_UNIQUE_ID) = [&]() {             \
        if (daal::internal::is_logger_enabled()) {                                                  \
            DAAL_PROFILER_LOG_ARGS(task_name, __VA_ARGS__);                                         \
        }                                                                                           \
        return ONEDAL_PROFILER_START_TASK(task_name);                                               \
    }()

#define ONEDAL_PROFILER_TASK_WITH_ARGS_QUEUE(task_name, queue, ...)                                 \
    auto ONEDAL_PROFILER_CONCAT(__profiler_task__, ONEDAL_PROFILER_UNIQUE_ID) = [&]() {             \
        if (daal::internal::is_logger_enabled()) {                                                  \
            DAAL_PROFILER_LOG_ARGS(task_name, __VA_ARGS__);                                         \
        }                                                                                           \
        return ONEDAL_PROFILER_START_TASK_WITH_QUEUE(task_name, queue);                             \
    }()

#define ONEDAL_PROFILER_TASK(...)                                                                   \
    auto ONEDAL_PROFILER_CONCAT(__profiler_task__, ONEDAL_PROFILER_UNIQUE_ID) = [&]() {             \
        if (daal::internal::is_logger_enabled()) {                                                  \
            DAAL_PROFILER_PRINT_HEADER();                                                           \
            std::cerr << "Profiler task_name: " << #__VA_ARGS__ << '\n';                            \
        }                                                                                           \
        return ONEDAL_PROFILER_GET_MACRO(__VA_ARGS__,                                               \
                                         ONEDAL_PROFILER_MACRO_2,                                   \
                                         ONEDAL_PROFILER_MACRO_1,                                   \
                                         FICTIVE)(__VA_ARGS__);                                     \
    }()

// SERVICE TASK variants gate on `is_service_debug_enabled()`. Structurally identical to the
// non-service TASK macros; the only difference is the enable predicate applied inside each
// START_TASK*. To keep behavior consistent we define parallel service variants of the START_TASK
// macros too.
#define ONEDAL_PROFILER_SERVICE_START_TASK(name)                                                    \
    (daal::internal::is_service_debug_enabled() ? daal::internal::profiler::start_task(#name)       \
                                                : daal::internal::profiler::start_task(nullptr))

#ifdef ONEDAL_DATA_PARALLEL
    #define ONEDAL_PROFILER_SERVICE_START_TASK_WITH_QUEUE(name, queue)                              \
        ::oneapi::dal::detail::queue_sync_guard::make((queue), #name,                               \
                                                      daal::internal::is_service_debug_enabled())
#else
    #define ONEDAL_PROFILER_SERVICE_START_TASK_WITH_QUEUE(name, queue) ONEDAL_PROFILER_SERVICE_START_TASK(name)
#endif

#define ONEDAL_PROFILER_SERVICE_MACRO_1(name)        ONEDAL_PROFILER_SERVICE_START_TASK(name)
#define ONEDAL_PROFILER_SERVICE_MACRO_2(name, queue) ONEDAL_PROFILER_SERVICE_START_TASK_WITH_QUEUE(name, queue)

// SERVICE variants gate the logger side-effect on `is_service_debug_enabled()`, not the plain
// `is_logger_enabled()` used by the non-service TASK macros. Service tasks fire in hot paths
// (e.g. every table2ndarray conversion), so allowing them to print under ONEDAL_VERBOSE=1/4 would
// swamp the output; only DEBUG mode (5) prints service task args, matching the pre-refactor gate.
#define ONEDAL_PROFILER_SERVICE_TASK_WITH_ARGS(task_name, ...)                                      \
    auto ONEDAL_PROFILER_CONCAT(__profiler_task__, ONEDAL_PROFILER_UNIQUE_ID) = [&]() {             \
        if (daal::internal::is_service_debug_enabled()) {                                           \
            DAAL_PROFILER_LOG_ARGS(task_name, __VA_ARGS__);                                         \
        }                                                                                           \
        return ONEDAL_PROFILER_SERVICE_START_TASK(task_name);                                       \
    }()

#define ONEDAL_PROFILER_SERVICE_TASK_WITH_ARGS_QUEUE(task_name, queue, ...)                         \
    auto ONEDAL_PROFILER_CONCAT(__profiler_task__, ONEDAL_PROFILER_UNIQUE_ID) = [&]() {             \
        if (daal::internal::is_service_debug_enabled()) {                                           \
            DAAL_PROFILER_LOG_ARGS(task_name, __VA_ARGS__);                                         \
        }                                                                                           \
        return ONEDAL_PROFILER_SERVICE_START_TASK_WITH_QUEUE(task_name, queue);                     \
    }()

#define ONEDAL_PROFILER_SERVICE_TASK(...)                                                           \
    auto ONEDAL_PROFILER_CONCAT(__profiler_task__, ONEDAL_PROFILER_UNIQUE_ID) = [&]() {             \
        if (daal::internal::is_service_debug_enabled()) {                                           \
            DAAL_PROFILER_PRINT_HEADER();                                                           \
            std::cerr << "Profiler task_name: " << #__VA_ARGS__ << '\n';                            \
        }                                                                                           \
        return ONEDAL_PROFILER_GET_MACRO(__VA_ARGS__,                                               \
                                         ONEDAL_PROFILER_SERVICE_MACRO_2,                           \
                                         ONEDAL_PROFILER_SERVICE_MACRO_1,                           \
                                         FICTIVE)(__VA_ARGS__);                                     \
    }()
