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

namespace oneapi::dal::detail {
profiler_task profiler::start_task(const char* task_name) {
    return profiler_task(task_name);
}

void profiler::end_task(const char* task_name) {}

profiler_task::profiler_task(const char* task_name) : task_name_(task_name) {}

#ifdef ONEDAL_DATA_PARALLEL
profiler_task profiler::start_task(const char* task_name, const sycl::queue& task_queue) {
    return profiler_task(task_name, task_queue);
}

profiler_task::profiler_task(const char* task_name, const sycl::queue& task_queue)
        : task_name_(task_name),
          task_queue_(task_queue) {}
#endif

profiler_task::~profiler_task() {
    profiler::end_task(task_name_);
}

} // namespace oneapi::dal::detail
