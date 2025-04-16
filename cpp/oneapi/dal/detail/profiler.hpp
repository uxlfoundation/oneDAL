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

#ifdef ONEDAL_DATA_PARALLEL
#include <sycl/sycl.hpp>
#endif

#include <chrono>
#include <cstdint>
#include <cstring>
#include <map>
#include <vector>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <algorithm>

#ifdef _WIN32
#define PRETTY_FUNCTION __FUNCSIG__
#else
#define PRETTY_FUNCTION __PRETTY_FUNCTION__
#endif

// UTILS
#define ONEDAL_PROFILER_MACRO_1(name)                       ONEDAL_PROFILER_START_TASK(name)
#define ONEDAL_PROFILER_MACRO_2(name, queue)                ONEDAL_PROFILER_START_TASK_WITH_QUEUE(name, queue)
#define ONEDAL_PROFILER_GET_MACRO(arg_1, arg_2, MACRO, ...) MACRO

// HEADER OUTPUT
#define ONEDAL_PROFILER_PRINT_HEADER()                                                         \
    do {                                                                                       \
        std::cerr                                                                              \
            << "-----------------------------------------------------------------------------" \
            << std::endl;                                                                      \
        std::cerr << __PRETTY_FUNCTION__ << std::endl;                                         \
    } while (0)

// ARGS LOGGING
#define ONEDAL_PROFILER_LOG_ARGS(task_name, ...)                                 \
    do {                                                                         \
        ONEDAL_PROFILER_PRINT_HEADER();                                          \
        std::cerr << "Profiler task_name: " << #task_name << " Printed args: ";  \
        oneapi::dal::detail::profiler_log_named_args(#__VA_ARGS__, __VA_ARGS__); \
        std::cerr << std::endl;                                                  \
    } while (0)

// START_TASKS
#define ONEDAL_PROFILER_START_TASK(name) oneapi::dal::detail::profiler::start_task(#name)
#define ONEDAL_PROFILER_START_TASK_WITH_QUEUE(name, queue) \
    oneapi::dal::detail::profiler::start_task(#name, queue)
#define ONEDAL_PROFILER_START_NULL_TASK() oneapi::dal::detail::profiler::start_task(nullptr)

// PROFILER TASKS
#define ONEDAL_PROFILER_TASK_WITH_ARGS(task_name, ...)            \
    oneapi::dal::detail::profiler_task __profiler_task =          \
        (oneapi::dal::detail::profiler::is_verbose_enabled())     \
        ? [&]() -> oneapi::dal::detail::profiler_task {           \
        if (oneapi::dal::detail::profiler::is_logger_enabled()) { \
            ONEDAL_PROFILER_LOG_ARGS(task_name, __VA_ARGS__);     \
        }                                                         \
        return ONEDAL_PROFILER_START_TASK(task_name);             \
    }()                                                           \
        : ONEDAL_PROFILER_START_NULL_TASK()

#define ONEDAL_PROFILER_TASK_WITH_ARGS_QUEUE(task_name, queue, ...)     \
    oneapi::dal::detail::profiler_task __profiler_task =                \
        (oneapi::dal::detail::profiler::is_verbose_enabled())           \
        ? [&]() -> oneapi::dal::detail::profiler_task {                 \
        if (oneapi::dal::detail::profiler::is_logger_enabled()) {       \
            ONEDAL_PROFILER_LOG_ARGS(task_name, __VA_ARGS__);           \
        }                                                               \
        return ONEDAL_PROFILER_START_TASK_WITH_QUEUE(task_name, queue); \
    }()                                                                 \
        : ONEDAL_PROFILER_START_NULL_TASK()

#define ONEDAL_PROFILER_TASK(...)                                             \
    oneapi::dal::detail::profiler_task __profiler_task =                      \
        (oneapi::dal::detail::profiler::is_verbose_enabled())                 \
        ? [&]() -> oneapi::dal::detail::profiler_task {                       \
        if (oneapi::dal::detail::profiler::is_logger_enabled()) {             \
            ONEDAL_PROFILER_PRINT_HEADER();                                   \
            std::cerr << "Profiler task_name: " << #__VA_ARGS__ << std::endl; \
        }                                                                     \
        return ONEDAL_PROFILER_GET_MACRO(__VA_ARGS__,                         \
                                         ONEDAL_PROFILER_MACRO_2,             \
                                         ONEDAL_PROFILER_MACRO_1,             \
                                         FICTIVE)(__VA_ARGS__);               \
    }()                                                                       \
        : ONEDAL_PROFILER_START_NULL_TASK()

#define ONEDAL_PROFILER_SERVICE_TASK_WITH_ARGS(task_name, ...)      \
    oneapi::dal::detail::profiler_task __profiler_task =            \
        (oneapi::dal::detail::profiler::is_service_debug_enabled()) \
        ? [&]() -> oneapi::dal::detail::profiler_task {             \
        if (oneapi::dal::detail::profiler::is_logger_enabled()) {   \
            ONEDAL_PROFILER_LOG_ARGS(task_name, __VA_ARGS__);       \
        }                                                           \
        return ONEDAL_PROFILER_START_TASK(task_name);               \
    }()                                                             \
        : ONEDAL_PROFILER_START_NULL_TASK()

#define ONEDAL_PROFILER_SERVICE_TASK_WITH_ARGS_QUEUE(task_name, queue, ...) \
    oneapi::dal::detail::profiler_task __profiler_task =                    \
        (oneapi::dal::detail::profiler::is_service_debug_enabled())         \
        ? [&]() -> oneapi::dal::detail::profiler_task {                     \
        if (oneapi::dal::detail::profiler::is_logger_enabled()) {           \
            ONEDAL_PROFILER_LOG_ARGS(task_name, __VA_ARGS__);               \
        }                                                                   \
        return ONEDAL_PROFILER_START_TASK_WITH_QUEUE(task_name, queue);     \
    }()                                                                     \
        : ONEDAL_PROFILER_START_NULL_TASK()

#define ONEDAL_PROFILER_SERVICE_TASK(...)                                     \
    oneapi::dal::detail::profiler_task __profiler_task =                      \
        (oneapi::dal::detail::profiler::is_service_debug_enabled())           \
        ? [&]() -> oneapi::dal::detail::profiler_task {                       \
        if (oneapi::dal::detail::profiler::is_logger_enabled()) {             \
            ONEDAL_PROFILER_PRINT_HEADER();                                   \
            std::cerr << "Profiler task_name: " << #__VA_ARGS__ << std::endl; \
        }                                                                     \
        return ONEDAL_PROFILER_GET_MACRO(__VA_ARGS__,                         \
                                         ONEDAL_PROFILER_MACRO_2,             \
                                         ONEDAL_PROFILER_MACRO_1,             \
                                         FICTIVE)(__VA_ARGS__);               \
    }()                                                                       \
        : ONEDAL_PROFILER_START_NULL_TASK()

namespace oneapi::dal::detail {
inline void profiler_log_named_args(const char* /*names*/) {
    // base case — no args
}

template <typename T, typename... Rest>
void profiler_log_named_args(const char* names, const T& value, Rest&&... rest) {
    const char* comma = strchr(names, ',');
    std::string name = comma ? std::string(names, comma) : std::string(names);

    name.erase(0, name.find_first_not_of(" \t\n\r"));

    std::cerr << name << ": " << value << "; ";

    if (comma) {
        profiler_log_named_args(comma + 1, std::forward<Rest>(rest)...);
    }
}

struct task_entry {
    std::int64_t idx;
    std::string name;
    std::uint64_t duration;
    std::int64_t level;
};

struct task {
    std::vector<task_entry> kernels;
};

class profiler_task {
public:
    profiler_task(const char* task_name, int idx);
    ~profiler_task();

    profiler_task(const profiler_task&) = delete;
    profiler_task& operator=(const profiler_task&) = delete;

private:
    const char* task_name_;
    int idx;
};

class profiler {
public:
    profiler();
    ~profiler();
    static profiler_task start_task(const char* task_name);
    static std::uint64_t get_time();
    static profiler* get_instance();
    task& get_task();
    std::int64_t& get_current_level();
    std::int64_t& get_kernel_count();
    static bool is_verbose_enabled();
    static bool is_profiler_enabled();
    static bool is_logger_enabled();
    static bool is_analyzer_enabled();
    static bool is_service_debug_enabled();
#ifdef ONEDAL_DATA_PARALLEL
    sycl::queue& get_queue();
    void set_queue(const sycl::queue& q);
    bool is_queue_exists();
    static profiler_task start_task(const char* task_name, sycl::queue& task_queue);
#endif
    static void end_task(const char* task_name, int idx);

private:
    std::int64_t current_level = 0;
    std::int64_t kernel_count = 0;
    task task_;
#ifdef ONEDAL_DATA_PARALLEL
    sycl::queue queue_;
    bool queue_exists_;
#endif
};

} // namespace oneapi::dal::detail
