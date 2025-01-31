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

#include "oneapi/dal/detail/profiler.hpp"
#include <iostream>

namespace oneapi::dal::detail {

profiler::profiler() {
    start_time = get_time();
}

profiler::~profiler() {
    auto end_time = get_time();
    auto total_time = end_time - start_time;
    std::cerr << "KERNEL_PROFILER: total time " << total_time / 1e6 << std::endl;
}

std::uint64_t profiler::get_time() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000000000 + t.tv_nsec;
}

profiler* profiler::get_instance() {
    static profiler instance;
    return &instance;
}

task& profiler::get_task() {
    return task_;
}

#ifdef ONEDAL_DATA_PARALLEL
sycl::queue& profiler::get_queue() {
    return queue_;
}

void profiler::set_queue(const sycl::queue& q) {
    queue_ = q;
}
#endif

profiler_task profiler::start_task(const char* task_name) {
    auto ns_start = get_time();
    auto& tasks_info = get_instance()->get_task();
    tasks_info.time_kernels[tasks_info.current_kernel] = ns_start;
    tasks_info.current_kernel++;
    return profiler_task(task_name);
}

void profiler::end_task(const char* task_name) {
    const std::uint64_t ns_end = get_time();
    auto& tasks_info = get_instance()->get_task();
#ifdef ONEDAL_DATA_PARALLEL
    auto& queue = get_instance()->get_queue();
    queue.wait_and_throw();
#endif
    tasks_info.current_kernel--;
    const std::uint64_t times = ns_end - tasks_info.time_kernels[tasks_info.current_kernel];

    auto it = tasks_info.kernels.find(task_name);
    if (it == tasks_info.kernels.end()) {
        tasks_info.kernels.insert({ task_name, times });
    }
    else {
        it->second += times;
    }
    std::cerr << "KERNEL_PROFILER: " << std::string(task_name) << " " << times / 1e6 << std::endl;
}

#ifdef ONEDAL_DATA_PARALLEL
profiler_task profiler::start_task(const char* task_name, sycl::queue& task_queue) {
    task_queue.wait_and_throw();
    get_instance()->set_queue(task_queue);
    auto ns_start = get_time();
    auto& tasks_info = get_instance()->get_task();
    tasks_info.time_kernels[tasks_info.current_kernel] = ns_start;
    tasks_info.current_kernel++;
    return profiler_task(task_name, task_queue);
}



profiler_task::profiler_task(const char* task_name, const sycl::queue& task_queue)
        : task_name_(task_name),
          task_queue_(task_queue),
          has_queue_(true) {}
          
#endif

profiler_task::profiler_task(const char* task_name) 
    : task_name_(task_name) {}

profiler_task::~profiler_task() {
    #ifdef ONEDAL_DATA_PARALLEL
    if (has_queue_)
        task_queue_.wait_and_throw();
    #endif // ONEDAL_DATA_PARALLEL
    profiler::end_task(task_name_);
}

} // namespace oneapi::dal::detail
