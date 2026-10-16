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

#include <cstdlib>
#include <cstring>
#include <string>

#include "src/services/service_profiler.h"

#ifdef ONEDAL_DATA_PARALLEL
#include <sycl/sycl.hpp>
#endif

namespace oneapi::dal::detail {

#ifdef ONEDAL_DATA_PARALLEL
/// Whether tasks given a queue wait for it; ONEDAL_PROFILER_GPU_SYNC=0 times only the host side,
/// so profiling does not change how device work overlaps.
inline bool is_queue_sync_enabled() {
    static const bool value = [] {
        const char* sync = std::getenv("ONEDAL_PROFILER_GPU_SYNC");
        return !(sync && std::strcmp(sync, "0") == 0);
    }();
    return value;
}

// Waits on the queue at both ends of a profiled region so the duration covers device execution.
// make() waits before start_task() takes its timestamp, so earlier work is not charged to the region.
// When `enabled` is false, or queue waits are turned off, no waits are issued.
class queue_sync_guard {
public:
    /// @param[in] args  The task's arguments for the analyzer, or empty
    static queue_sync_guard make(sycl::queue& q,
                                 const char* task_name,
                                 bool enabled,
                                 const std::string& args = std::string()) {
        if (enabled) {
            const bool sync = is_queue_sync_enabled();
            if (sync) {
                q.wait_and_throw();
            }
            return queue_sync_guard(
                sync ? &q : nullptr,
                args.empty() ? daal::internal::profiler::start_task(task_name)
                             : daal::internal::profiler::start_task_with_args(task_name, args));
        }
        return queue_sync_guard(nullptr, daal::internal::profiler::start_task(nullptr));
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

/// Shortens a type name taken from a function signature to the part a reader needs, for example
/// `kmeans::train_kernel_cpu<float, lloyd_dense, clustering>`.
inline std::string profiler_short_type_name(const char* signature) {
    std::string text = signature;
#ifdef _WIN32
    const auto open = text.find("profiler_type_name<");
    const auto close = text.rfind(">(");
    if (open != std::string::npos && close != std::string::npos && close > open) {
        text = text.substr(open + 19, close - open - 19);
    }
#else
    const auto open = text.find("T = ");
    if (open != std::string::npos) {
        text = text.substr(open + 4);
        const auto close = text.find_first_of(";]");
        text = text.substr(0, close);
    }
#endif
    for (const char* noise : { "oneapi::dal::",
                               "backend::",
                               "detail::",
                               "method::",
                               "task::",
                               "v1::",
                               "v2::",
                               "struct ",
                               "class " }) {
        const std::string what = noise;
        for (auto at = text.find(what); at != std::string::npos; at = text.find(what, at)) {
            text.erase(at, what.size());
        }
    }
    // The algorithm's namespace once, at the front: `kmeans::train_kernel_cpu<float, lloyd_dense>`.
    const auto scope = text.find("::");
    if (scope != std::string::npos) {
        const std::string algorithm = text.substr(0, scope + 2);
        for (auto at = text.find(algorithm, scope + 2); at != std::string::npos;
             at = text.find(algorithm, at)) {
            text.erase(at, algorithm.size());
        }
    }
    for (auto at = text.find(",", 0); at != std::string::npos; at = text.find(",", at + 1)) {
        if (at + 1 < text.size() && text[at + 1] != ' ') {
            text.insert(at + 1, " ");
        }
    }
    return text;
}

/// The name of `T` for a profiler task, computed once per type.
template <typename T>
inline const char* profiler_type_name() {
    static const std::string name = profiler_short_type_name(PRETTY_FUNCTION);
    return name.c_str();
}

} // namespace oneapi::dal::detail

#define ONEDAL_PROFILER_CONCAT2(x, y) x##y
#define ONEDAL_PROFILER_CONCAT(x, y)  ONEDAL_PROFILER_CONCAT2(x, y)
#define ONEDAL_PROFILER_UNIQUE_ID     __LINE__

// UTILS
#define ONEDAL_PROFILER_MACRO_1(name)                       ONEDAL_PROFILER_START_TASK(name)
#define ONEDAL_PROFILER_MACRO_2(name, queue)                ONEDAL_PROFILER_START_TASK_WITH_QUEUE(name, queue)
#define ONEDAL_PROFILER_GET_MACRO(arg_1, arg_2, MACRO, ...) MACRO

// START_TASKS
// Each START_TASK* macro checks is_profiler_enabled() itself, so the TASK macros can dispatch by
// argument count alone.
#define ONEDAL_PROFILER_START_TASK(name)                                                 \
    (daal::internal::is_profiler_enabled() ? daal::internal::profiler::start_task(#name) \
                                           : daal::internal::profiler::start_task(nullptr))

#ifdef ONEDAL_DATA_PARALLEL
#define ONEDAL_PROFILER_START_TASK_WITH_QUEUE(name, queue) \
    ::oneapi::dal::detail::queue_sync_guard::make((queue), \
                                                  #name,   \
                                                  daal::internal::is_profiler_enabled())
#else
// Host-only build: time on the host, as the no-queue variant does.
#define ONEDAL_PROFILER_START_TASK_WITH_QUEUE(name, queue) ONEDAL_PROFILER_START_TASK(name)
#endif

#define ONEDAL_PROFILER_START_NULL_TASK() daal::internal::profiler::start_task(nullptr)

// The arguments of a task, formatted only when the analyzer reports them.
#define ONEDAL_PROFILER_ARGS(...)                                         \
    (daal::internal::is_analyzer_enabled()                                \
         ? daal::internal::profiler_named_args(#__VA_ARGS__, __VA_ARGS__) \
         : std::string())

#define ONEDAL_PROFILER_START_TASK_WITH_ARGS(name, ...)                                      \
    (!daal::internal::is_profiler_enabled() ? daal::internal::profiler::start_task(nullptr)  \
     : daal::internal::is_analyzer_enabled()                                                 \
         ? daal::internal::profiler::start_task_with_args(#name,                             \
                                                          ONEDAL_PROFILER_ARGS(__VA_ARGS__)) \
         : daal::internal::profiler::start_task(#name))

#ifdef ONEDAL_DATA_PARALLEL
#define ONEDAL_PROFILER_START_TASK_WITH_QUEUE_ARGS(name, queue, ...)                     \
    ::oneapi::dal::detail::queue_sync_guard::make((queue),                               \
                                                  #name,                                 \
                                                  daal::internal::is_profiler_enabled(), \
                                                  ONEDAL_PROFILER_ARGS(__VA_ARGS__))
#else
#define ONEDAL_PROFILER_START_TASK_WITH_QUEUE_ARGS(name, queue, ...) \
    ONEDAL_PROFILER_START_TASK_WITH_ARGS(name, __VA_ARGS__)
#endif

// PROFILER TASKS
// `auto` holds a profiler_task or a queue_sync_guard, depending on the argument count.
#define ONEDAL_PROFILER_TASK_WITH_ARGS(task_name, ...)                                  \
    auto ONEDAL_PROFILER_CONCAT(__profiler_task__, ONEDAL_PROFILER_UNIQUE_ID) = [&]() { \
        if (daal::internal::is_logger_enabled()) {                                      \
            DAAL_PROFILER_LOG_ARGS(task_name, __VA_ARGS__);                             \
        }                                                                               \
        return ONEDAL_PROFILER_START_TASK_WITH_ARGS(task_name, __VA_ARGS__);            \
    }()

#define ONEDAL_PROFILER_TASK_WITH_ARGS_QUEUE(task_name, queue, ...)                       \
    auto ONEDAL_PROFILER_CONCAT(__profiler_task__, ONEDAL_PROFILER_UNIQUE_ID) = [&]() {   \
        if (daal::internal::is_logger_enabled()) {                                        \
            DAAL_PROFILER_LOG_ARGS(task_name, __VA_ARGS__);                               \
        }                                                                                 \
        return ONEDAL_PROFILER_START_TASK_WITH_QUEUE_ARGS(task_name, queue, __VA_ARGS__); \
    }()

#define ONEDAL_PROFILER_TASK(...)                                                       \
    auto ONEDAL_PROFILER_CONCAT(__profiler_task__, ONEDAL_PROFILER_UNIQUE_ID) = [&]() { \
        if (daal::internal::is_logger_enabled()) {                                      \
            DAAL_PROFILER_LOG("Profiler task_name: " << #__VA_ARGS__ << '\n');          \
        }                                                                               \
        return ONEDAL_PROFILER_GET_MACRO(__VA_ARGS__,                                   \
                                         ONEDAL_PROFILER_MACRO_2,                       \
                                         ONEDAL_PROFILER_MACRO_1,                       \
                                         FICTIVE)(__VA_ARGS__);                         \
    }()

// SERVICE variants are enabled by is_service_debug_enabled() instead of is_profiler_enabled().
#define ONEDAL_PROFILER_SERVICE_START_TASK(name)                                              \
    (daal::internal::is_service_debug_enabled() ? daal::internal::profiler::start_task(#name) \
                                                : daal::internal::profiler::start_task(nullptr))

#ifdef ONEDAL_DATA_PARALLEL
#define ONEDAL_PROFILER_SERVICE_START_TASK_WITH_QUEUE(name, queue) \
    ::oneapi::dal::detail::queue_sync_guard::make((queue),         \
                                                  #name,           \
                                                  daal::internal::is_service_debug_enabled())
#else
#define ONEDAL_PROFILER_SERVICE_START_TASK_WITH_QUEUE(name, queue) \
    ONEDAL_PROFILER_SERVICE_START_TASK(name)
#endif

#define ONEDAL_PROFILER_SERVICE_MACRO_1(name) ONEDAL_PROFILER_SERVICE_START_TASK(name)
#define ONEDAL_PROFILER_SERVICE_MACRO_2(name, queue) \
    ONEDAL_PROFILER_SERVICE_START_TASK_WITH_QUEUE(name, queue)

// Service tasks run in hot paths such as table2ndarray, so they log only in debug mode (5).
#define ONEDAL_PROFILER_SERVICE_TASK_WITH_ARGS(task_name, ...)                          \
    auto ONEDAL_PROFILER_CONCAT(__profiler_task__, ONEDAL_PROFILER_UNIQUE_ID) = [&]() { \
        if (daal::internal::is_service_debug_enabled()) {                               \
            DAAL_PROFILER_LOG_ARGS(task_name, __VA_ARGS__);                             \
        }                                                                               \
        return ONEDAL_PROFILER_SERVICE_START_TASK(task_name);                           \
    }()

#define ONEDAL_PROFILER_SERVICE_TASK_WITH_ARGS_QUEUE(task_name, queue, ...)             \
    auto ONEDAL_PROFILER_CONCAT(__profiler_task__, ONEDAL_PROFILER_UNIQUE_ID) = [&]() { \
        if (daal::internal::is_service_debug_enabled()) {                               \
            DAAL_PROFILER_LOG_ARGS(task_name, __VA_ARGS__);                             \
        }                                                                               \
        return ONEDAL_PROFILER_SERVICE_START_TASK_WITH_QUEUE(task_name, queue);         \
    }()

#define ONEDAL_PROFILER_SERVICE_TASK(...)                                               \
    auto ONEDAL_PROFILER_CONCAT(__profiler_task__, ONEDAL_PROFILER_UNIQUE_ID) = [&]() { \
        if (daal::internal::is_service_debug_enabled()) {                               \
            DAAL_PROFILER_LOG("Profiler task_name: " << #__VA_ARGS__ << '\n');          \
        }                                                                               \
        return ONEDAL_PROFILER_GET_MACRO(__VA_ARGS__,                                   \
                                         ONEDAL_PROFILER_SERVICE_MACRO_2,               \
                                         ONEDAL_PROFILER_SERVICE_MACRO_1,               \
                                         FICTIVE)(__VA_ARGS__);                         \
    }()
