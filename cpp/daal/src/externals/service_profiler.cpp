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

std::mutex profiler::mutex_;

static volatile int daal_verbose_val = -1;

static bool device_info_needed   = false;
static bool kernel_info_needed   = false;
static bool function_info_needed = false;

void print_header()
{
    // daal::services::LibraryVersionInfo ver;

    // std::cout << "Major version:          " << ver.majorVersion << std::endl;
    // std::cout << "Minor version:          " << ver.minorVersion << std::endl;
    // std::cout << "Update version:         " << ver.updateVersion << std::endl;
    // std::cout << "Product status:         " << ver.productStatus << std::endl;
    // std::cout << "Build:                  " << ver.build << std::endl;
    // std::cout << "Build revision:         " << ver.build_rev << std::endl;
    // std::cout << "Name:                   " << ver.name << std::endl;
    // std::cout << "Processor optimization: " << ver.processor << std::endl;
    std::cout << std::endl;
}

/**
* Returns the pointer to variable that holds oneDAL verbose mode information (enabled/disabled)
*
*  @returns pointer to mode
*                      0 disabled
*                      1 enabled
*                      2 enabled with device and library information(will be added soon)
*/
static void set_verbose_from_env(void)
{
    static volatile int read_done = 0;
    if (read_done) return;

    const char * verbose_str = std::getenv("ONEDAL_VERBOSE");
    int newval               = 0;
    if (verbose_str)
    {
        newval = std::atoi(verbose_str);
        if (newval < 0 || newval > 4)
        {
            newval = 0;
        }
        else
        {
            print_header();
        }
    }

    daal_verbose_val = newval;
    read_done        = 1;
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

int * onedal_verbose_mode()
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

bool profiler::is_profiling_enabled()
{
    return *onedal_verbose_mode() > 0;
}

int onedal_verbose(int option)
{
    int * retVal = onedal_verbose_mode();
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
    int verbose = *onedal_verbose_mode();
    if (verbose == 1)
    {
        kernel_info_needed = true;
    }
    else if (verbose == 2)
    {
        kernel_info_needed = true;
    }
    else if (verbose == 3)
    {
        device_info_needed   = true;
        kernel_info_needed   = true;
        function_info_needed = true;
    }
}

profiler::~profiler()
{
    if (kernel_info_needed)
    {
        const auto & tasks_info  = get_instance()->get_task();
        std::uint64_t total_time = 0;
        std::cerr << "Algorithm tree profiler" << std::endl;
        for (size_t i = 0; i < tasks_info.kernels.size(); ++i)
        {
            const auto & entry = tasks_info.kernels[i];
            std::string prefix;

            for (std::int64_t lvl = 0; lvl < entry.level; ++lvl)
            {
                prefix += "│   ";
            }
            if (entry.level == 0)
            {
                total_time += entry.duration;
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

            std::cerr << prefix << entry.name << " time: " << format_time_for_output(entry.duration) << std::endl;
        }

        std::cerr << "╰── (end)" << std::endl;

        std::cerr << "DAAL KERNEL_PROFILER: ALL KERNELS total time " << format_time_for_output(total_time) << std::endl;
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
    if (!is_profiling_enabled()) return profiler_task(nullptr, -1);

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
    if (!is_profiling_enabled()) return;

    std::lock_guard<std::mutex> lock(mutex_);
    const std::uint64_t ns_end = get_time();
    auto & tasks_info          = get_instance()->get_task();

    auto it = std::find_if(tasks_info.kernels.begin(), tasks_info.kernels.end(), [&](const task_entry & entry) { return entry.idx == idx_; });

    auto duration         = ns_end - it->duration;
    it->duration          = duration;
    auto & current_level_ = get_instance()->get_current_level();
    current_level_--;
}

profiler_task::profiler_task(const char * task_name, int idx_) : task_name_(task_name), idx(idx_) {}

profiler_task::~profiler_task()
{
    if (task_name_) profiler::end_task(task_name_, idx);
}

} // namespace internal
} // namespace daal
