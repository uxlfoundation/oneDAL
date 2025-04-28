/* file: service_profiler.cpp */
/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "services/internal/service_profiler.h"
#include <sstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include "services/library_version_info.h"
#include <mutex>
#include <unordered_set>

namespace daal
{
namespace internal
{

profiler_task profiler::start_threading_task(const char * task_name)
{
    if (!task_name) return profiler_task(nullptr, -1);
    static std::mutex mutex;

    std::lock_guard<std::mutex> lock(mutex);
    if (is_logger_enabled())
    {
        if (!is_service_debug_enabled())
        {
            static std::unordered_set<std::string> unique_task_names;
            if (unique_task_names.insert(task_name).second)
            {
                std::cerr << "-----------------------------------------------------------------------------" << '\n';
                std::cerr << "THREADING Profiler task started on the main rank: " << task_name << '\n';
            }
        }
        else
        {
            std::cerr << "-----------------------------------------------------------------------------" << '\n';
            std::cerr << "THREADING Profiler task started " << task_name << '\n';
        }
    }
    auto ns_start                = get_time();
    auto & tasks_info            = get_instance()->get_task();
    auto & current_level_        = get_instance()->get_current_level();
    auto & current_kernel_count_ = get_instance()->get_kernel_count();
    std::int64_t tmp             = current_kernel_count_;
    tasks_info.kernels.push_back({ tmp, task_name, ns_start, current_level_, 1, true });
    current_kernel_count_++;
    return profiler_task(task_name, tmp, true);
}

void profiler::end_threading_task(const char * task_name, int idx_)
{
    if (!task_name) return;

    static std::mutex mutex;

    std::lock_guard<std::mutex> lock(mutex);
    const std::uint64_t ns_end = get_time();
    auto & tasks_info          = get_instance()->get_task();

    if (idx_ < 0 || static_cast<std::size_t>(idx_) >= tasks_info.kernels.size()) return;

    auto & entry   = tasks_info.kernels[idx_];
    auto duration  = ns_end - entry.duration;
    entry.duration = duration;

    if (is_tracer_enabled())
    {
        static std::unordered_set<std::string> unique_task_names;
        if (unique_task_names.emplace(task_name).second)
        {
            std::cerr << "THREADING " << task_name
                      << " finished on the main rank(time could be different for other ranks): " << format_time_for_output(duration) << '\n';
        }
    }
}

} // namespace internal
} // namespace daal
