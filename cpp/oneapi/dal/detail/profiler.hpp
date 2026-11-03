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
// Waits on the queue at both ends of a profiled task, so its time covers the device work it
// submitted and none of the work submitted before it.
class queue_sync_guard {
public:
    static queue_sync_guard make(sycl::queue& q, const char* task_name, bool enabled) {
        if (!enabled) {
            return queue_sync_guard(nullptr, daal::internal::profiler::start_task(nullptr));
        }
        q.wait_and_throw();
        return queue_sync_guard(&q, daal::internal::profiler::start_queue_task(task_name));
    }

    ~queue_sync_guard() {
        if (queue_) {
            try {
                queue_->wait_and_throw();
            }
            catch (std::exception& e) {
                std::cerr << "ONEDAL KERNEL_PROFILER: unexpected error while waiting on the queue: "
                          << e.what() << std::endl;
            }
            catch (...) {
                std::cerr << "ONEDAL KERNEL_PROFILER: unexpected error while waiting on the queue"
                          << std::endl;
            }
        }
    }

    queue_sync_guard(const queue_sync_guard&) = delete;
    queue_sync_guard& operator=(const queue_sync_guard&) = delete;

    queue_sync_guard(queue_sync_guard&& other) noexcept
            : queue_(other.queue_),
              task_(std::move(other.task_)) {
        other.queue_ = nullptr;
    }

    queue_sync_guard& operator=(queue_sync_guard&&) = delete;

private:
    queue_sync_guard(sycl::queue* q, daal::internal::profiler_task&& task)
            : queue_(q),
              task_(std::move(task)) {}

    sycl::queue* queue_ = nullptr;
    daal::internal::profiler_task task_;
};
#endif

} // namespace oneapi::dal::detail

#define ONEDAL_PROFILER_CONCAT2(x, y) x##y
#define ONEDAL_PROFILER_CONCAT(x, y)  ONEDAL_PROFILER_CONCAT2(x, y)
#define ONEDAL_PROFILER_UNIQUE_ID     __LINE__

// START_TASKS
// `enabled` is the one mode check the TASK macro made; a task that is off costs only that check.
#define ONEDAL_PROFILER_START_TASK(name, enabled) \
    daal::internal::profiler::start_task((enabled) ? #name : nullptr)

#ifdef ONEDAL_DATA_PARALLEL
#define ONEDAL_PROFILER_START_TASK_WITH_QUEUE(name, queue, enabled) \
    ::oneapi::dal::detail::queue_sync_guard::make((queue), #name, (enabled))
#else
#define ONEDAL_PROFILER_START_TASK_WITH_QUEUE(name, queue, enabled) \
    ONEDAL_PROFILER_START_TASK(name, enabled)
#endif

#define ONEDAL_PROFILER_START_NULL_TASK() daal::internal::profiler::start_task(nullptr)

// UTILS
#define ONEDAL_PROFILER_MACRO_1(enabled, name) ONEDAL_PROFILER_START_TASK(name, enabled)
#define ONEDAL_PROFILER_MACRO_2(enabled, name, queue) \
    ONEDAL_PROFILER_START_TASK_WITH_QUEUE(name, queue, enabled)
#define ONEDAL_PROFILER_GET_MACRO(arg_1, arg_2, MACRO, ...) MACRO

// PROFILER TASKS
// `auto` holds a profiler_task or a queue_sync_guard, depending on the argument count.
#define ONEDAL_PROFILER_TASK_WITH_ARGS(task_name, ...)                                     \
    auto ONEDAL_PROFILER_CONCAT(__profiler_task__, ONEDAL_PROFILER_UNIQUE_ID) = [&]() {    \
        const bool profiler_on_ = daal::internal::is_profiler_enabled();                   \
        if (DAAL_PROFILER_UNLIKELY(profiler_on_) && daal::internal::is_logger_enabled()) { \
            DAAL_PROFILER_LOG_ARGS(task_name, __VA_ARGS__);                                \
        }                                                                                  \
        return ONEDAL_PROFILER_START_TASK(task_name, profiler_on_);                        \
    }()

#define ONEDAL_PROFILER_TASK_WITH_ARGS_QUEUE(task_name, queue, ...)                        \
    auto ONEDAL_PROFILER_CONCAT(__profiler_task__, ONEDAL_PROFILER_UNIQUE_ID) = [&]() {    \
        const bool profiler_on_ = daal::internal::is_profiler_enabled();                   \
        if (DAAL_PROFILER_UNLIKELY(profiler_on_) && daal::internal::is_logger_enabled()) { \
            DAAL_PROFILER_LOG_ARGS(task_name, __VA_ARGS__);                                \
        }                                                                                  \
        return ONEDAL_PROFILER_START_TASK_WITH_QUEUE(task_name, queue, profiler_on_);      \
    }()

#define ONEDAL_PROFILER_TASK(...)                                                          \
    auto ONEDAL_PROFILER_CONCAT(__profiler_task__, ONEDAL_PROFILER_UNIQUE_ID) = [&]() {    \
        const bool profiler_on_ = daal::internal::is_profiler_enabled();                   \
        if (DAAL_PROFILER_UNLIKELY(profiler_on_) && daal::internal::is_logger_enabled()) { \
            DAAL_PROFILER_LOG("Profiler task_name: " << #__VA_ARGS__ << '\n');             \
        }                                                                                  \
        return ONEDAL_PROFILER_GET_MACRO(__VA_ARGS__,                                      \
                                         ONEDAL_PROFILER_MACRO_2,                          \
                                         ONEDAL_PROFILER_MACRO_1,                          \
                                         FICTIVE)(profiler_on_, __VA_ARGS__);              \
    }()

// SERVICE tasks run in hot paths such as table2ndarray, so they record and log only at level 5.
#define ONEDAL_PROFILER_SERVICE_TASK_WITH_ARGS(task_name, ...)                          \
    auto ONEDAL_PROFILER_CONCAT(__profiler_task__, ONEDAL_PROFILER_UNIQUE_ID) = [&]() { \
        const bool profiler_on_ = daal::internal::is_service_debug_enabled();           \
        if (DAAL_PROFILER_UNLIKELY(profiler_on_)) {                                     \
            DAAL_PROFILER_LOG_ARGS(task_name, __VA_ARGS__);                             \
        }                                                                               \
        return ONEDAL_PROFILER_START_TASK(task_name, profiler_on_);                     \
    }()

#define ONEDAL_PROFILER_SERVICE_TASK_WITH_ARGS_QUEUE(task_name, queue, ...)             \
    auto ONEDAL_PROFILER_CONCAT(__profiler_task__, ONEDAL_PROFILER_UNIQUE_ID) = [&]() { \
        const bool profiler_on_ = daal::internal::is_service_debug_enabled();           \
        if (DAAL_PROFILER_UNLIKELY(profiler_on_)) {                                     \
            DAAL_PROFILER_LOG_ARGS(task_name, __VA_ARGS__);                             \
        }                                                                               \
        return ONEDAL_PROFILER_START_TASK_WITH_QUEUE(task_name, queue, profiler_on_);   \
    }()

#define ONEDAL_PROFILER_SERVICE_TASK(...)                                               \
    auto ONEDAL_PROFILER_CONCAT(__profiler_task__, ONEDAL_PROFILER_UNIQUE_ID) = [&]() { \
        const bool profiler_on_ = daal::internal::is_service_debug_enabled();           \
        if (DAAL_PROFILER_UNLIKELY(profiler_on_)) {                                     \
            DAAL_PROFILER_LOG("Profiler task_name: " << #__VA_ARGS__ << '\n');          \
        }                                                                               \
        return ONEDAL_PROFILER_GET_MACRO(__VA_ARGS__,                                   \
                                         ONEDAL_PROFILER_MACRO_2,                       \
                                         ONEDAL_PROFILER_MACRO_1,                       \
                                         FICTIVE)(profiler_on_, __VA_ARGS__);           \
    }()
