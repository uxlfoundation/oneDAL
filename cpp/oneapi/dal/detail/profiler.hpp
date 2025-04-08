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
#include <stdexcept>

#define ONEDAL_PROFILER_CONCAT2(x, y) x##y
#define ONEDAL_PROFILER_CONCAT(x, y)  ONEDAL_PROFILER_CONCAT2(x, y)

#define ONEDAL_PROFILER_UNIQUE_ID __LINE__

#define ONEDAL_PROFILER_MACRO_1(name)                       oneapi::dal::detail::profiler::start_task(#name)
#define ONEDAL_PROFILER_MACRO_2(name, queue)                oneapi::dal::detail::profiler::start_task(#name, queue)
#define ONEDAL_PROFILER_GET_MACRO(arg_1, arg_2, MACRO, ...) MACRO

#define ONEDAL_PROFILER_TASK(...)                                                          \
    oneapi::dal::detail::profiler_task ONEDAL_PROFILER_CONCAT(__profiler_task__,           \
                                                              ONEDAL_PROFILER_UNIQUE_ID) = \
        ONEDAL_PROFILER_GET_MACRO(__VA_ARGS__,                                             \
                                  ONEDAL_PROFILER_MACRO_2,                                 \
                                  ONEDAL_PROFILER_MACRO_1,                                 \
                                  FICTIVE)(__VA_ARGS__)

namespace oneapi::dal::detail {

struct task {
    std::map<std::string, std::uint64_t> kernels;
    std::uint64_t current_kernel = 0;
    std::vector<std::uint64_t> time_kernels;
    void clear();
};

class profiler_task {
public:
    profiler_task(const char* task_name);
#ifdef ONEDAL_DATA_PARALLEL
    profiler_task(const char* task_name, const sycl::queue& task_queue);
#endif
    ~profiler_task();

    profiler_task(const profiler_task&) = delete;
    profiler_task& operator=(const profiler_task&) = delete;

private:
    const char* task_name_;
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

#ifdef ONEDAL_DATA_PARALLEL
    sycl::queue& get_queue();
    void set_queue(const sycl::queue& q);
    static profiler_task start_task(const char* task_name, sycl::queue& task_queue);
#endif
    static void end_task(const char* task_name);

private:
    std::uint64_t start_time;
    std::uint64_t total_time;
    task task_;
#ifdef ONEDAL_DATA_PARALLEL
    sycl::queue queue_;
#endif
};

} // namespace oneapi::dal::detail
