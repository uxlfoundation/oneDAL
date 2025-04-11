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

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif
#include <time.h>
#include <cstdint>
#include <cstring>
#include <map>
#include <vector>
#include <mutex>
#include <stdexcept>
#include <algorithm>

#define ONEDAL_PROFILER_CONCAT2(x, y) x##y
#define ONEDAL_PROFILER_CONCAT(x, y)  ONEDAL_PROFILER_CONCAT2(x, y)

#define ONEDAL_PROFILER_UNIQUE_ID __LINE__

#define ONEDAL_PROFILER_MACRO_1(name)        oneapi::dal::detail::profiler::start_task(#name)
#define ONEDAL_PROFILER_MACRO_2(name, queue) oneapi::dal::detail::profiler::start_task(#name, queue)
#define ONEDAL_PROFILER_SERVICE_MACRO_1(name) \
    oneapi::dal::detail::profiler::start_service_task(#name)
#define ONEDAL_PROFILER_SERVICE_MACRO_2(name, queue) \
    oneapi::dal::detail::profiler::start_service_task(#name, queue)
#define ONEDAL_PROFILER_GET_MACRO(arg_1, arg_2, MACRO, ...) MACRO

#define ONEDAL_PROFILER_TASK(...)                                                          \
    oneapi::dal::detail::profiler_task ONEDAL_PROFILER_CONCAT(__profiler_task__,           \
                                                              ONEDAL_PROFILER_UNIQUE_ID) = \
        (oneapi::dal::detail::profiler::is_profiling_enabled()                             \
             ? ONEDAL_PROFILER_GET_MACRO(__VA_ARGS__,                                      \
                                         ONEDAL_PROFILER_MACRO_2,                          \
                                         ONEDAL_PROFILER_MACRO_1,                          \
                                         FICTIVE)(__VA_ARGS__)                             \
             : oneapi::dal::detail::profiler::start_task(nullptr))

#define ONEDAL_PROFILER_SERVICE(...)                                                       \
    oneapi::dal::detail::profiler_task ONEDAL_PROFILER_CONCAT(__profiler_task__,           \
                                                              ONEDAL_PROFILER_UNIQUE_ID) = \
        (oneapi::dal::detail::profiler::is_profiling_enabled()                             \
             ? ONEDAL_PROFILER_GET_MACRO(__VA_ARGS__,                                      \
                                         ONEDAL_PROFILER_MACRO_2,                          \
                                         ONEDAL_PROFILER_MACRO_1,                          \
                                         FICTIVE)(__VA_ARGS__)                             \
             : oneapi::dal::detail::profiler::start_service_task(nullptr))

namespace oneapi::dal::detail {

struct task_entry {
    std::int64_t idx;
    std::string name;
    std::uint64_t duration;
    std::uint64_t level;
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
    static profiler_task start_service_task(const char* task_name);
    static std::uint64_t get_time();
    static profiler* get_instance();
    task& get_task();
    std::uint64_t& get_current_level();
    std::int64_t& get_current_kernel_count();
    static bool is_profiling_enabled();
#ifdef ONEDAL_DATA_PARALLEL
    sycl::queue& get_queue();
    void set_queue(const sycl::queue& q);
    static profiler_task start_task(const char* task_name, sycl::queue& task_queue);
    static profiler_task start_service_task(const char* task_name, sycl::queue& task_queue);
#endif
    static void end_task(const char* task_name, int idx);

private:
    std::uint64_t start_time = 0;
    std::uint64_t current_kernel = 0;
    std::int64_t total_kernel_count = 0;
    task task_;
    static std::mutex mutex_;
#ifdef ONEDAL_DATA_PARALLEL
    sycl::queue queue_;
#endif
};

} // namespace oneapi::dal::detail
