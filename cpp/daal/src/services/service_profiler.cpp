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

#include "services/service_profiler.h"
#include <sstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include "services/library_version_info.h"
#include <mutex>

namespace daal
{
namespace internal
{

static volatile int daal_verbose_val = -1;

static void set_verbose_from_env(void)
{
    const char * verbose_str = std::getenv("ONEDAL_VERBOSE");
    int newval               = 0;

    if (verbose_str)
    {
        char * endptr = nullptr;
        errno         = 0;
        long val      = std::strtol(verbose_str, &endptr, 10);

        bool parsed_ok = (errno == 0 && endptr != verbose_str && *endptr == '\0');

        if (parsed_ok && val >= 0 && val <= 5)
        {
            newval = static_cast<int>(val);
        }
    }

    daal_verbose_val = newval;
}

std::string format_time_for_output(std::uint64_t time_ns)
{
    std::ostringstream out;
    double time = static_cast<double>(time_ns);

    if (time <= 0)
    {
        out << "0.00s";
    }
    else
    {
        if (time > 1e9)
        {
            out << std::fixed << std::setprecision(2) << time / 1e9 << "s";
        }
        else if (time > 1e6)
        {
            out << std::fixed << std::setprecision(2) << time / 1e6 << "ms";
        }
        else if (time > 1e3)
        {
            out << std::fixed << std::setprecision(2) << time / 1e3 << "us";
        }
        else
        {
            out << static_cast<std::uint64_t>(time) << "ns";
        }
    }

    return out.str();
}

/**
* Returns the pointer to variable that holds oneDAL verbose mode information (enabled/disabled)
*
*  @returns pointer to mode
*                      0/empty disabled
*                      1 enabled only with logger
*                      2 enabled only with tracer
*                      3 enabled only with analyzer
*                      4 enabled with logger tracer and analyzer
*                      5 enabled with logger tracer and analyzer with service functions
*/
int daal_verbose_mode()
{
#if defined(__GNUC__) || defined(__clang__)
    if (__builtin_expect((daal_verbose_val == -1), 0))
#else
    if (daal_verbose_val == -1)
#endif
        set_verbose_from_env();
    return daal_verbose_val;
}

// Check for the service functions profiling
bool profiler::is_service_debug_enabled()
{
    static const bool service_debug_value = [] {
        int value = daal_verbose_mode();
        return value == 5;
    }();
    return service_debug_value;
}

// Check for logger enabling(PRETTY_FUNCTION + Params)
bool profiler::is_logger_enabled()
{
    static const bool logger_value = [] {
        int value = daal_verbose_mode();
        return value == 1 || value == 4 || value == 5;
    }();
    return logger_value;
}

// Check for tracer enabling(kernel runtimes in runtime)
bool profiler::is_tracer_enabled()
{
    static const bool verbose_value = [] {
        std::ios::sync_with_stdio(false);
        int value = daal_verbose_mode();
        return value == 2 || value == 4 || value == 5;
    }();
    return verbose_value;
}

// General check for profiler
bool profiler::is_profiler_enabled()
{
    static const bool profiler_value = [] {
        int value = daal_verbose_mode();
        return value == 1 || value == 2 || value == 3 || value == 4 || value == 5;
    }();
    return profiler_value;
}

// Check for analyzer enabling(algorithm trees and kernel times)
bool profiler::is_analyzer_enabled()
{
    static const bool profiler_value = [] {
        int value = daal_verbose_mode();
        return value == 3 || value == 4 || value == 5;
    }();
    return profiler_value;
}

profiler::profiler()
{
    int verbose = daal_verbose_mode();
    if (verbose == 1 || verbose == 4 || verbose == 5)
    {
        daal::internal::print_header();
    }
}

profiler::~profiler()
{
    merge_tasks();
    if (is_analyzer_enabled())
    {
        const auto & tasks_info  = get_instance()->get_task();
        std::uint64_t total_time = 0;
        std::cerr << "Algorithm tree analyzer" << '\n';

        for (size_t i = 0; i < tasks_info.kernels.size(); ++i)
        {
            const auto & entry = tasks_info.kernels[i];
            if (entry.level == 0)
            {
                total_time += entry.duration;
            }
        }

        for (size_t i = 0; i < tasks_info.kernels.size(); ++i)
        {
            const auto & entry = tasks_info.kernels[i];
            std::string prefix;

            for (std::int64_t lvl = 0; lvl < entry.level; ++lvl)
            {
                prefix += "|   ";
            }

            bool is_last = true;
            if (i + 1 < tasks_info.kernels.size())
            {
                const auto & next = tasks_info.kernels[i + 1];
                if (next.level >= entry.level)
                {
                    is_last = false;
                }
            }

            prefix += is_last ? "|-- " : "|-- ";

            std::cerr << prefix << entry.name << " time: " << format_time_for_output(entry.duration) << " " << std::fixed << std::setprecision(2)
                      << (total_time > 0 ? (double(entry.duration) / total_time) * 100 : 0.0) << "% " << entry.count << " times"
                      << " in a " << entry.threading_task << " region" << '\n';
        }

        std::cerr << "|---(end)" << '\n';

        std::cerr << "DAAL KERNEL_PROFILER: kernels total time " << format_time_for_output(total_time) << '\n';
    }
}

std::uint64_t profiler::get_time()
{
    auto now = std::chrono::steady_clock::now();
    auto ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    return static_cast<std::uint64_t>(ns);
}

profiler * profiler::get_instance()
{
    static profiler instance;
    return &instance;
}

std::int64_t & profiler::get_current_level()
{
    return current_level;
}

std::int64_t & profiler::get_kernel_count()
{
    return kernel_count;
}

task & profiler::get_task()
{
    return task_;
}

void profiler::merge_tasks()
{
    auto & tasks_info = get_instance()->get_task();
    auto & kernels    = tasks_info.kernels;

    size_t i = 0;
    while (i < kernels.size())
    {
        // finding the start and the end of the current level
        size_t start      = i;
        int current_level = kernels[i].level;
        size_t end        = start;
        while (end < kernels.size() && kernels[end].level == current_level) ++end;

        // A flag to check is it merged region or no
        bool merged = false;

        for (size_t j = start; j < end; ++j)
        {
            for (size_t k = j + 1; k < end; ++k)
            {
                // The same kernels on the same level
                if (kernels[j].name == kernels[k].name)
                {
                    // For thread tasks takes the max, for all other sums
                    if (kernels[j].threading_task)
                    {
                        kernels[j].duration = std::max(kernels[j].duration, kernels[k].duration);
                    }
                    else
                    {
                        kernels[j].duration += kernels[k].duration;
                    }

                    kernels.erase(kernels.begin() + k);
                    --k;
                    --end;
                    kernels[j].count++;
                    merged = true;
                }
            }
        }

        if (merged)
        {
            std::cout << "Loop/Parallel section. The output is squashed. "
                      << "To get unsquashed version use ONEDAL_VERBOSE=5" << std::endl;
        }

        i = end;
    }
}

profiler_task profiler::start_task(const char * task_name)
{
    if (task_name == nullptr) return profiler_task(nullptr, -1);

    auto ns_start     = get_time();
    auto & tasks_info = get_instance()->get_task();

    auto & current_level_        = get_instance()->get_current_level();
    auto & current_kernel_count_ = get_instance()->get_kernel_count();

    std::int64_t tmp = current_kernel_count_;
    tasks_info.kernels.push_back({ tmp, task_name, ns_start, current_level_, 1, false });

    current_level_++;
    current_kernel_count_++;
    return profiler_task(task_name, tmp);
}

profiler_task profiler::start_threading_task(const char * task_name)
{
    if (task_name == nullptr) return profiler_task(nullptr, -1);

    auto ns_start     = get_time();
    auto & tasks_info = get_instance()->get_task();

    auto & current_level_        = get_instance()->get_current_level();
    auto & current_kernel_count_ = get_instance()->get_kernel_count();

    std::int64_t tmp = current_kernel_count_;
    tasks_info.kernels.push_back({ tmp, task_name, ns_start, current_level_, 1, true });

    current_kernel_count_++;
    return profiler_task(task_name, tmp, true);
}

void profiler::end_task(const char * task_name, int idx_)
{
    if (task_name == nullptr) return;

    const std::uint64_t ns_end = get_time();
    auto & tasks_info          = get_instance()->get_task();

    auto it = std::find_if(tasks_info.kernels.begin(), tasks_info.kernels.end(), [&](const task_entry & entry) { return entry.idx == idx_; });

    auto duration         = ns_end - it->duration;
    it->duration          = duration;
    auto & current_level_ = profiler::get_instance()->get_current_level();
    current_level_--;
    if (is_tracer_enabled()) std::cerr << task_name << " " << format_time_for_output(duration) << '\n';
}

void profiler::end_threading_task(const char * task_name, int idx_)
{
    if (task_name == nullptr) return;

    const std::uint64_t ns_end = get_time();
    auto & tasks_info          = get_instance()->get_task();

    auto it = std::find_if(tasks_info.kernels.begin(), tasks_info.kernels.end(), [&](const task_entry & entry) { return entry.idx == idx_; });

    auto duration = ns_end - it->duration;
    it->duration  = duration;

    if (is_tracer_enabled()) std::cerr << task_name << " " << format_time_for_output(duration) << '\n';
}

profiler_task::profiler_task(const char * task_name, int idx_) : task_name_(task_name), idx(idx_) {}

profiler_task::profiler_task(const char * task_name, int idx_, bool thread_) : task_name_(task_name), idx(idx_), is_thread(thread_) {}

profiler_task::~profiler_task()
{
    if (task_name_)
    {
        if (is_thread)
        {
            profiler::end_threading_task(task_name_, idx);
        }
        else
        {
            profiler::end_task(task_name_, idx);
        }
    }
}

} // namespace internal
} // namespace daal
