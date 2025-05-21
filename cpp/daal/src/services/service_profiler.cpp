/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#include "src/services/service_profiler.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <sstream>
#include <iomanip>
#include <cstring>

namespace daal
{
namespace internal
{

void set_verbose_from_env()
{
    const char * verbose_str = std::getenv("ONEDAL_VERBOSE");
    int newval               = PROFILER_MODE_OFF;
    if (verbose_str)
    {
        char * endptr  = nullptr;
        errno          = 0;
        long val       = std::strtol(verbose_str, &endptr, 10);
        bool parsed_ok = (errno == 0 && endptr != verbose_str && *endptr == '\0');
        if (parsed_ok && val >= 0 && val <= 5) newval = static_cast<int>(val);
    }
    daal_verbose_val = newval;
}

int daal_verbose_mode()
{
    if (daal_verbose_val == -1) set_verbose_from_env();
    return daal_verbose_val;
}

std::string format_time_for_output(std::uint64_t time_ns)
{
    std::ostringstream out;
    double time = static_cast<double>(time_ns);
    if (time <= 0)
        out << "0.00s";
    else if (time > 1e9)
        out << std::fixed << std::setprecision(2) << time / 1e9 << "s";
    else if (time > 1e6)
        out << std::fixed << std::setprecision(2) << time / 1e6 << "ms";
    else if (time > 1e3)
        out << std::fixed << std::setprecision(2) << time / 1e3 << "us";
    else
        out << static_cast<std::uint64_t>(time) << "ns";
    return out.str();
}

bool is_service_debug_enabled()
{
    static const bool service_debug_value = [] {
        int value = daal_verbose_mode();
        return value == PROFILER_MODE_DEBUG;
    }();
    return service_debug_value;
}

bool is_logger_enabled()
{
    static const bool logger_value = [] {
        int value = daal_verbose_mode();
        return value == PROFILER_MODE_LOGGER || value == PROFILER_MODE_ALL_TOOLS || value == PROFILER_MODE_DEBUG;
    }();
    return logger_value;
}

bool is_tracer_enabled()
{
    static const bool verbose_value = [] {
        std::ios::sync_with_stdio(false);
        int value = daal_verbose_mode();
        return value == PROFILER_MODE_TRACER || value == PROFILER_MODE_ALL_TOOLS || value == PROFILER_MODE_DEBUG;
    }();
    return verbose_value;
}

bool is_profiler_enabled()
{
    static const bool profiler_value = [] {
        int value = daal_verbose_mode();
        return value == PROFILER_MODE_LOGGER || value == PROFILER_MODE_TRACER || value == PROFILER_MODE_ANALYZER || value == PROFILER_MODE_ALL_TOOLS
               || value == PROFILER_MODE_DEBUG;
    }();
    return profiler_value;
}

bool is_analyzer_enabled()
{
    static const bool profiler_value = [] {
        int value = daal_verbose_mode();
        return value == PROFILER_MODE_ANALYZER || value == PROFILER_MODE_ALL_TOOLS || value == PROFILER_MODE_DEBUG;
    }();
    return profiler_value;
}

void print_header()
{
    if (is_profiler_enabled())
    {
        daal::services::LibraryVersionInfo ver;
        std::cerr << "Major version:          " << ver.majorVersion << '\n';
        std::cerr << "Minor version:          " << ver.minorVersion << '\n';
        std::cerr << "Update version:         " << ver.updateVersion << '\n';
        std::cerr << "Product status:         " << ver.productStatus << '\n';
        std::cerr << "Build:                  " << ver.build << '\n';
        std::cerr << "Build revision:         " << ver.build_rev << '\n';
        std::cerr << "Name:                   " << ver.name << '\n';
        std::cerr << "Processor optimization: " << ver.processor << '\n';
        std::cerr << '\n';
    }
}

profiler::profiler()
{
    daal_verbose_mode();
}

profiler::~profiler()
{
    if (is_analyzer_enabled())
    {
        merge_tasks();
        const auto & tasks_info  = get_task();
        std::uint64_t total_time = 0;
        std::cerr << "Algorithm tree analyzer" << '\n';

        for (size_t i = 0; i < tasks_info.kernels.size(); ++i)
        {
            const auto & entry = tasks_info.kernels[i];
            if (entry.level == 0) total_time += entry.duration;
        }

        for (size_t i = 0; i < tasks_info.kernels.size(); ++i)
        {
            const auto & entry = tasks_info.kernels[i];
            std::string prefix;
            for (std::int64_t lvl = 0; lvl < entry.level; ++lvl) prefix += "|   ";
            bool is_last = (i + 1 < tasks_info.kernels.size()) && (tasks_info.kernels[i + 1].level >= entry.level) ? false : true;
            prefix += is_last ? "|-- " : "|-- ";
            std::cerr << prefix << entry.name << " time: " << format_time_for_output(entry.duration) << " " << std::fixed << std::setprecision(2)
                      << (total_time > 0 ? (double(entry.duration) / total_time) * 100 : 0.0) << "% " << entry.count << " times"
                      << " in a " << entry.threading_task << " region" << '\n';
        }
        std::cerr << "|---(end)" << '\n';
        std::cerr << "DAAL KERNEL_PROFILER: kernels total time " << format_time_for_output(total_time) << '\n';
    }
}

profiler_task profiler::start_task(const char * task_name)
{
    if (!task_name) return profiler_task(nullptr, -1);
    auto ns_start                = get_time();
    auto & tasks_info            = get_instance()->get_task();
    auto & current_level_        = get_instance()->get_current_level();
    auto & current_kernel_count_ = get_instance()->get_kernel_count();
    std::int64_t tmp             = current_kernel_count_;
    tasks_info.kernels.push_back({ tmp, task_name, ns_start, current_level_, 1, false });
    current_level_++;
    current_kernel_count_++;
    return profiler_task(task_name, tmp);
}

profiler_task profiler::start_threading_task(const char * task_name)
{
    if (!task_name) return profiler_task(nullptr, -1);
    static std::mutex mutex;

    std::lock_guard<std::mutex> lock(mutex);
    if (is_logger_enabled())
    {
        if (!is_service_debug_enabled())
        {
            static std::vector<std::string> unique_task_names;
            bool is_new_task = std::find(unique_task_names.begin(), unique_task_names.end(), task_name) == unique_task_names.end();
            if (is_new_task)
            {
                unique_task_names.push_back(task_name);
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

void profiler::end_task(const char * task_name, int idx_)
{
    if (!task_name) return;
    const std::uint64_t ns_end = get_time();
    auto & tasks_info          = get_instance()->get_task();
    auto & entry               = tasks_info.kernels[idx_];
    auto duration              = ns_end - entry.duration;
    entry.duration             = duration;
    auto & current_level_      = get_instance()->get_current_level();
    current_level_--;
    if (is_tracer_enabled()) std::cerr << task_name << " " << format_time_for_output(duration) << '\n';
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
        static std::vector<std::string> unique_task_names;
        bool is_new_task = std::find(unique_task_names.begin(), unique_task_names.end(), task_name) == unique_task_names.end();
        if (is_new_task)
        {
            unique_task_names.push_back(task_name);
            std::cerr << "THREADING " << task_name
                      << " finished on the main rank(time could be different for other ranks): " << format_time_for_output(duration) << '\n';
        }
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

void profiler::merge_tasks()
{
    if (is_service_debug_enabled())
    {
        return;
    }
    auto & tasks_info = get_instance()->get_task();
    auto & kernels    = tasks_info.kernels;
    size_t i          = 0;
    while (i < kernels.size())
    {
        size_t start      = i;
        int current_level = kernels[i].level;
        size_t end        = start;
        while (end < kernels.size() && kernels[end].level == current_level) ++end;
        for (size_t j = start; j < end; ++j)
        {
            for (size_t k = j + 1; k < end; ++k)
            {
                if (kernels[j].name == kernels[k].name)
                {
                    if (kernels[j].threading_task)
                        kernels[j].duration = std::max(kernels[j].duration, kernels[k].duration);
                    else
                        kernels[j].duration += kernels[k].duration;
                    kernels.erase(kernels.begin() + k);
                    --k;
                    --end;
                    kernels[j].count++;
                }
            }
        }
        i = end;
    }
}

task & profiler::get_task()
{
    return task_;
}

std::int64_t & profiler::get_current_level()
{
    return current_level_;
}

std::int64_t & profiler::get_kernel_count()
{
    return kernel_count_;
}

profiler_task::~profiler_task()
{
    if (task_name_)
    {
        if (is_thread_)
            profiler::end_threading_task(task_name_, idx_);
        else
            profiler::end_task(task_name_, idx_);
    }
}

} // namespace internal
} // namespace daal
