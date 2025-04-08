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
#include <iostream>
#include <sstream>
#include <iomanip>
#include "services/library_version_info.h"
namespace daal
{
namespace internal
{

#ifdef DAAL_REF
static volatile int daal_verbose_val __attribute__((aligned(64))) = -1;
#else
__declspec(align(64)) static volatile int daal_verbose_val = -1;
#endif

static bool device_info_needed = false;
static bool kernel_info_needed = false;

//__declspec(align(64)) static volatile char verbose_file_val[PATH_MAX] = {'\0'};

/**
* Returns the pointer to variable that holds oneDAL verbose mode information (enabled/disabled)
*
*  @returns pointer to mode
*                      0 disabled
*                      1 enabled
*                      2 enabled with device and library information(will be added soon)
*/

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
    static volatile int read_done = 0;
    if (read_done) return;

    const char * verbose_str = std::getenv("ONEDAL_VERBOSE");
    int newval               = 0;
    if (verbose_str)
    {
        newval = std::atoi(verbose_str);
        if (newval < 0 || newval > 2)
        {
            newval = 0;
        }
    }

    daal_verbose_val = newval;
    read_done        = 1;
}

int * onedal_verbose_mode()
{
    if (__builtin_expect((daal_verbose_val == -1), 0))
    {
        // ADD MUTEX
        if (daal_verbose_val == -1) set_verbose_from_env();
        // DISABLE MUTEX
    }
    return (int *)&daal_verbose_val;
}

int onedal_verbose(int option)
{
    int * retVal = onedal_verbose_mode();
    if (option != 0 && option != 1 && option != 2)
    {
        return -1;
    }
    if (option != daal_verbose_val)
    {
        // ADD MUTEX
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
        device_info_needed = true;
        kernel_info_needed = true;
    }

    if (device_info_needed)
    {
        print_header();
    }
    start_time = get_time();
}

profiler::~profiler()
{
    if (kernel_info_needed)
    {
        std::cerr << "DAAL KERNEL_PROFILER: ALL KERNELS total time "
                  << " " << format_time_for_output(total_time) << std::endl;
    }
}

std::uint64_t profiler::get_time()
{
#ifdef _WIN32
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (counter.QuadPart * 1000000000) / frequency.QuadPart; // Перевод в наносекунды
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000000000 + t.tv_nsec;
#endif
}

profiler * profiler::get_instance()
{
    static profiler instance;
    return &instance;
}

task & profiler::get_task()
{
    return task_;
}

profiler_task profiler::start_task(const char * task_name)
{
    auto ns_start     = get_time();
    auto & tasks_info = get_instance()->get_task();
    tasks_info.time_kernels.push_back(ns_start);
    return profiler_task(task_name);
}

void profiler::end_task(const char * task_name)
{
    const std::uint64_t ns_end = get_time();
    auto & tasks_info          = get_instance()->get_task();

    if (tasks_info.time_kernels.empty())
    {
        throw std::runtime_error("Attempting to end a task when no tasks are running");
    }
    const std::uint64_t ns_start = tasks_info.time_kernels.back();
    tasks_info.time_kernels.pop_back();
    const std::uint64_t times = ns_end - ns_start;

    auto it = tasks_info.kernels.find(task_name);
    if (it == tasks_info.kernels.end())
    {
        tasks_info.kernels.insert({ task_name, times });
    }
    else
    {
        it->second += times;
    }
    get_instance()->total_time += times;
    if (kernel_info_needed)
    {
        std::cerr << "DAAL KERNEL_PROFILER: total time " << std::string(task_name) << " " << format_time_for_output(times) << std::endl;
    }
}

profiler_task::profiler_task(const char * task_name) : task_name_(task_name) {}

profiler_task::~profiler_task()
{
    profiler::end_task(task_name_);
}
} // namespace internal
} // namespace daal
