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

#include "src/externals/service_profiler.h"
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

void print_header()
{
    daal::services::LibraryVersionInfo ver;

    std::cout << "Major version:          " << ver.majorVersion << std::endl;
    std::cout << "Minor version:          " << ver.minorVersion << std::endl;
    std::cout << "Update version:         " << ver.updateVersion << std::endl;
    std::cout << "Product status:         " << ver.productStatus << std::endl;
    std::cout << "Build:                  " << ver.build << std::endl;
    std::cout << "Build revision:         " << ver.build_rev << std::endl;
    std::cout << "Name:                   " << ver.name << std::endl;
    std::cout << "Processor optimization: " << ver.processor << std::endl;
    std::cout << std::endl;
}

static void set_verbose_from_env(void)
{
    // Every time check that the env is not changed
    // static volatile int read_done = 0;
    // if (read_done) return;

    const char * verbose_str = std::getenv("ONEDAL_VERBOSE");
    int newval               = 0;
    if (verbose_str)
    {
        newval = std::atoi(verbose_str);
        if (newval < 0 || newval > 5)
        {
            newval = 0;
        }
        else
        {
            print_header();
        }
    }

    daal_verbose_val = newval;
    // read_done        = 1;
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
int * daal_verbose_mode()
{
#ifdef _MSC_VER
    if (daal_verbose_val == -1)
#else
    if (__builtin_expect((daal_verbose_val == -1), 0))
#endif
    {
        if (daal_verbose_val == -1) set_verbose_from_env();
    }
    return (int *)&daal_verbose_val;
}

// Check for the service functions profiling
bool profiler::is_service_debug_enabled()
{
    static const bool service_debug_value = [] {
        int value = *daal_verbose_mode();
        return value == 5;
    }();
    return service_debug_value;
}

// Check for logger enabling(PRETTY_FUNCTION + Params)
bool profiler::is_logger_enabled()
{
    static const bool logger_value = [] {
        int value = *daal_verbose_mode();
        return value == 1 || value == 4 || value == 5;
    }();
    return logger_value;
}

// Check for tracer enabling(kernel runtimes in runtime)
bool profiler::is_tracer_enabled()
{
    static const bool verbose_value = [] {
        std::ios::sync_with_stdio(false);
        int value = *daal_verbose_mode();
        return value == 2 || value == 4 || value == 5;
    }();
    return verbose_value;
}

// General check for profiler
bool profiler::is_profiler_enabled()
{
    static const bool profiler_value = [] {
        int value = *daal_verbose_mode();
        return value == 1 || value == 2 || value == 3 || value == 4 || value == 5;
    }();
    return profiler_value;
}

// Check for analyzer enabling(algorithm trees and kernel times)
bool profiler::is_analyzer_enabled()
{
    static const bool profiler_value = [] {
        int value = *daal_verbose_mode();
        return value == 3 || value == 4 || value == 5;
    }();
    return profiler_value;
}

int daal_verbose(int option)
{
    int * retVal = daal_verbose_mode();
    if (option != 0 && option != 1 && option != 2 && option != 3)
    {
        return -1;
    }
    {
        if (option != daal_verbose_val) daal_verbose_val = option;
    }
    return *retVal;
}

profiler::profiler()
{
    int verbose = *daal_verbose_mode();
    if (verbose == 1 || verbose == 4 || verbose == 5)
    {
        print_header();
    }
}

profiler::~profiler()
{
    if (is_analyzer_enabled())
    {
        const auto & tasks_info  = get_instance()->get_task();
        std::uint64_t total_time = 0;
        std::cerr << "Algorithm tree analyzer" << std::endl;

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
                prefix += "│   ";
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

            prefix += is_last ? "└── " : "├── ";

            std::cerr << prefix << entry.name << " time: " << format_time_for_output(entry.duration) << " " << std::fixed << std::setprecision(2)
                      << (total_time > 0 ? (double(entry.duration) / total_time) * 100 : 0.0) << "%" << std::endl;
        }

        std::cerr << "╰── (end)" << std::endl;

        std::cerr << "ONEDAL KERNEL_PROFILER: ALL KERNELS total time " << format_time_for_output(total_time) << std::endl;
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

profiler_task profiler::start_task(const char * task_name)
{
    if (task_name == nullptr) return profiler_task(nullptr, -1);

    auto ns_start     = get_time();
    auto & tasks_info = get_instance()->get_task();

    auto & current_level_        = get_instance()->get_current_level();
    auto & current_kernel_count_ = get_instance()->get_kernel_count();

    std::int64_t tmp = current_kernel_count_;
    tasks_info.kernels.push_back({ tmp, task_name, ns_start, current_level_ });

    current_level_++;
    current_kernel_count_++;
    return profiler_task(task_name, tmp);
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
    if (is_tracer_enabled()) std::cerr << task_name << " " << format_time_for_output(duration) << std::endl;
}

profiler_task::profiler_task(const char * task_name, int idx_) : task_name_(task_name), idx(idx_) {}

profiler_task::~profiler_task()
{
    if (task_name_) profiler::end_task(task_name_, idx);
}

} // namespace internal
} // namespace daal
