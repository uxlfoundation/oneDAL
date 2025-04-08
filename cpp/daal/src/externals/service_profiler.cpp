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

namespace daal
{
namespace internal
{

__declspec(align(64)) static volatile int onedal_verbose_val = -1;

//__declspec(align(64)) static volatile char verbose_file_val[PATH_MAX] = {'\0'};

#define ONEDAL_VERBOSE_ENV      "ONEDAL_VERBOSE"
#define ONEDAL_VERBOSE_FILE_ENV "ONEDAL_VERBOSE_OUTPUT_FILE"

// static __forceinline int strtoint(const char *str, int def) {
//     int val;
//     char *tail;
//     if (str == NULL) return def;
//     val = strtol(str, &tail, 0);
//     return (*tail == '\0' && tail != str ? val : def);
// }

/**
* Returns the pointer to variable that holds oneDAL verbose mode information (enabled/disabled)
*
*  @returns pointer to mode
*                      0 disabled
*                      1 enabled
*                      2 enabled (on cpu) OR enabled with timing (on GPU)
*/

static void set_verbose_from_env(void)
{
    static volatile int read_done = 0;
    if (read_done) return;

    const char * verbose_str = std::getenv("ONEDAL_VERBOSE");
    int newval               = 0;

    if (verbose_str && *verbose_str != '\0')
    {
        bool valid = true;
        for (const char * p = verbose_str; *p != '\0'; ++p)
        {
            if (!std::isdigit(*p))
            {
                valid = false;
                break;
            }
        }

        if (valid)
        {
            newval = std::atoi(verbose_str);
            if (newval != 0 && newval != 1 && newval != 2)
            {
                newval = 0;
            }
        }
    }

    onedal_verbose_val = newval;
    read_done          = 1;
}

int * onedal_verbose_mode()
{
    if (__builtin_expect((onedal_verbose_val == -1), 0))
    {
        // ADD MUTEX
        if (onedal_verbose_val == -1) set_verbose_from_env();
        // DISABLE MUTEX
    }
    return (int *)&onedal_verbose_val;
}

int onedal_verbose(int option)
{
    int * retVal = onedal_verbose_mode();
    if (option != 0 && option != 1 && option != 2)
    {
        return -1;
    }
    if (option != onedal_verbose_val)
    {
        // ADD MUTEX
        if (option != onedal_verbose_val) onedal_verbose_val = option;
    }
    return *retVal;
}

profiler::profiler()
{
    start_time = get_time();
}

profiler::~profiler()
{
    auto end_time   = get_time();
    auto total_time = end_time - start_time;
    if (*onedal_verbose_mode() == 1)
    {
        std::cerr << "DAAL_KERNEL_PROFILER: total time " << total_time / 1e6 << std::endl;
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
    if (*onedal_verbose_mode() == 1)
    {
        std::cerr << "DAAL_KERNEL_PROFILER: " << std::string(task_name) << " " << times / 1e6 << std::endl;
    }
}

profiler_task::profiler_task(const char * task_name) : task_name_(task_name) {}

profiler_task::~profiler_task()
{
    profiler::end_task(task_name_);
}
} // namespace internal
} // namespace daal
