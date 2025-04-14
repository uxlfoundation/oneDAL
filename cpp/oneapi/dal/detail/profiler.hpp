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

#define ONEDAL_PROFILER_CONCAT2(x, y) x##y
#define ONEDAL_PROFILER_CONCAT(x, y)  ONEDAL_PROFILER_CONCAT2(x, y)

#define ONEDAL_PROFILER_UNIQUE_ID __LINE__

#define ONEDAL_PROFILER_MACRO_1(name)                       oneapi::dal::detail::profiler::start_task(#name)
#define ONEDAL_PROFILER_MACRO_2(name, queue)                oneapi::dal::detail::profiler::start_task(#name, queue)
#define ONEDAL_PROFILER_GET_MACRO(arg_1, arg_2, MACRO, ...) MACRO

#define ONEDAL_PROFILER_TASK_WITH_ARGS(task_name, ...)                                      \
    oneapi::dal::detail::profiler_task ONEDAL_PROFILER_CONCAT(                              \
        __profiler_task__,                                                                  \
        ONEDAL_PROFILER_UNIQUE_ID) = [&]() -> oneapi::dal::detail::profiler_task {          \
        if (oneapi::dal::detail::profiler::is_profiling_enabled()) {                        \
            std::cerr << "--------------------------------------------------" << std::endl; \
            std::cerr << __PRETTY_FUNCTION__ << std::endl;                                  \
            std::cerr << "task_name: " << #task_name << " args: ";                          \
            oneapi::dal::detail::profiler_log_named_args(#__VA_ARGS__, __VA_ARGS__);        \
            std::cerr << std::endl;                                                         \
            return oneapi::dal::detail::profiler::start_task(#task_name);                   \
        }                                                                                   \
        return oneapi::dal::detail::profiler::start_task(nullptr);                          \
    }()

#define ONEDAL_PROFILER_TASK_WITH_ARGS_QUEUE(task_name, queue, ...)                         \
    oneapi::dal::detail::profiler_task ONEDAL_PROFILER_CONCAT(                              \
        __profiler_task__,                                                                  \
        ONEDAL_PROFILER_UNIQUE_ID) = [&]() -> oneapi::dal::detail::profiler_task {          \
        if (oneapi::dal::detail::profiler::is_profiling_enabled()) {                        \
            std::cerr << "--------------------------------------------------" << std::endl; \
            std::cerr << __PRETTY_FUNCTION__ << std::endl;                                  \
            std::cerr << "task_name: " << #task_name << " args: ";                          \
            oneapi::dal::detail::profiler_log_named_args(#__VA_ARGS__, __VA_ARGS__);        \
            std::cerr << std::endl;                                                         \
            return oneapi::dal::detail::profiler::start_task(#task_name, queue);            \
        }                                                                                   \
        return oneapi::dal::detail::profiler::start_task(nullptr);                          \
    }()

#define ONEDAL_PROFILER_TASK(...)                                                           \
    oneapi::dal::detail::profiler_task ONEDAL_PROFILER_CONCAT(                              \
        __profiler_task__,                                                                  \
        ONEDAL_PROFILER_UNIQUE_ID) = [&]() -> oneapi::dal::detail::profiler_task {          \
        if (oneapi::dal::detail::profiler::is_profiling_enabled()) {                        \
            std::cerr << "--------------------------------------------------" << std::endl; \
            std::cerr << __PRETTY_FUNCTION__ << std::endl;                                  \
            std::cerr << "task_name: " << #__VA_ARGS__ << std::endl;                        \
            return ONEDAL_PROFILER_GET_MACRO(__VA_ARGS__,                                   \
                                             ONEDAL_PROFILER_MACRO_2,                       \
                                             ONEDAL_PROFILER_MACRO_1,                       \
                                             FICTIVE)(__VA_ARGS__);                         \
        }                                                                                   \
        return oneapi::dal::detail::profiler::start_task(nullptr);                          \
    }()

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
#ifdef ONEDAL_DATA_PARALLEL
    profiler_task(const char* task_name, const sycl::queue& task_queue, int idx);
#endif
    ~profiler_task();

    profiler_task(const profiler_task&) = delete;
    profiler_task& operator=(const profiler_task&) = delete;

private:
    const char* task_name_;
    int idx;
#ifdef ONEDAL_DATA_PARALLEL
    sycl::queue task_queue_;
    bool has_queue_ = false;
#endif
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
    static bool is_profiling_enabled();
#ifdef ONEDAL_DATA_PARALLEL
    sycl::queue& get_queue();
    void set_queue(const sycl::queue& q);
    static profiler_task start_task(const char* task_name, sycl::queue& task_queue);
#endif
    static void end_task(const char* task_name, int idx);

private:
    std::int64_t current_level = 0;
    std::int64_t kernel_count = 0;
    task task_;
    static std::mutex mutex_;
#ifdef ONEDAL_DATA_PARALLEL
    sycl::queue queue_;
#endif
};

} // namespace oneapi::dal::detail
