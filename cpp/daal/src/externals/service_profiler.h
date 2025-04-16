/* file: service_profiler.h */
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

/*
//++
//  Profiler for time measurement of kernels
//--
*/
#pragma once

#include <chrono>
#include <cstdint>
#include <cstring>
#include <map>
#include <vector>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <algorithm>

#ifdef _WIN32
    #define PRETTY_FUNCTION __FUNCSIG__
#else
    #define PRETTY_FUNCTION __PRETTY_FUNCTION__
#endif

#ifndef __SERVICE_PROFILER_H__
    #define __SERVICE_PROFILER_H__

    #define DAAL_ITTNOTIFY_CONCAT2(x, y) x##y
    #define DAAL_ITTNOTIFY_CONCAT(x, y)  DAAL_ITTNOTIFY_CONCAT2(x, y)
    #define DAAL_ITTNOTIFY_UNIQUE_ID     __LINE__

    #define DAAL_PROFILER_CONCAT2(x, y) x##y
    #define DAAL_PROFILER_CONCAT(x, y)  DAAL_PROFILER_CONCAT2(x, y)

    #define DAAL_PROFILER_UNIQUE_ID __LINE__

    #define DAAL_PROFILER_MACRO_1(name)                       daal::internal::profiler::start_task(#name)
    #define DAAL_PROFILER_MACRO_2(name, queue)                daal::internal::profiler::start_task(#name, queue)
    #define DAAL_PROFILER_GET_MACRO(arg_1, arg_2, MACRO, ...) MACRO

    #define DAAL_ITTNOTIFY_SCOPED_TASK_WITH_ARGS(task_name, ...)                                                                                  \
        daal::internal::profiler_task DAAL_PROFILER_CONCAT(__profiler_task__, DAAL_PROFILER_UNIQUE_ID) = [&]() -> daal::internal::profiler_task { \
            if (daal::internal::profiler::is_profiling_enabled())                                                                                 \
            {                                                                                                                                     \
                std::cerr << "-----------------------------------------------------------------------------" << std::endl;                        \
                std::cerr << PRETTY_FUNCTION << std::endl;                                                                                        \
                std::cerr << "Profiler task_name: " << #task_name << " Printed args: ";                                                           \
                daal::internal::profiler::_log_named_args(#__VA_ARGS__, __VA_ARGS__);                                                             \
                std::cerr << std::endl;                                                                                                           \
                return daal::internal::profiler::start_task(#task_name);                                                                          \
            }                                                                                                                                     \
            return daal::internal::profiler::start_task(nullptr);                                                                                 \
        }()

    #define DAAL_PROFILER_TASK(...)                                                                                                               \
        daal::internal::profiler_task DAAL_PROFILER_CONCAT(__profiler_task__, DAAL_PROFILER_UNIQUE_ID) = [&]() -> daal::internal::profiler_task { \
            if (daal::internal::profiler::is_profiling_enabled())                                                                                 \
            {                                                                                                                                     \
                std::cerr << "-----------------------------------------------------------------------------" << std::endl;                        \
                std::cerr << PRETTY_FUNCTION << std::endl;                                                                                        \
                std::cerr << "Profiler task_name: " << #__VA_ARGS__ << std::endl;                                                                 \
                return DAAL_PROFILER_GET_MACRO(__VA_ARGS__, DAAL_PROFILER_MACRO_2, DAAL_PROFILER_MACRO_1, FICTIVE)(__VA_ARGS__);                  \
            }                                                                                                                                     \
            return daal::internal::profiler::start_task(nullptr);                                                                                 \
        }()

    #define DAAL_ITTNOTIFY_SCOPED_TASK(...)                                                                                                       \
        daal::internal::profiler_task DAAL_PROFILER_CONCAT(__profiler_task__, DAAL_PROFILER_UNIQUE_ID) = [&]() -> daal::internal::profiler_task { \
            if (daal::internal::profiler::is_profiling_enabled())                                                                                 \
            {                                                                                                                                     \
                std::cerr << "-----------------------------------------------------------------------------" << std::endl;                        \
                std::cerr << PRETTY_FUNCTION << std::endl;                                                                                        \
                std::cerr << "Profiler task_name: " << #__VA_ARGS__ << std::endl;                                                                 \
                return DAAL_PROFILER_GET_MACRO(__VA_ARGS__, DAAL_PROFILER_MACRO_2, DAAL_PROFILER_MACRO_1, FICTIVE)(__VA_ARGS__);                  \
            }                                                                                                                                     \
            return daal::internal::profiler::start_task(nullptr);                                                                                 \
        }()

namespace daal
{
namespace internal
{

inline void profiler_log_named_args(const char * /*names*/)
{
    // base case — no args
}

template <typename T, typename... Rest>
void profiler_log_named_args(const char * names, const T & value, Rest &&... rest)
{
    const char * comma = strchr(names, ',');
    std::string name   = comma ? std::string(names, comma) : std::string(names);

    name.erase(0, name.find_first_not_of(" \t\n\r"));

    std::cerr << name << ": " << value << "; ";

    if (comma)
    {
        profiler_log_named_args(comma + 1, std::forward<Rest>(rest)...);
    }
}

struct task_entry
{
    std::int64_t idx;
    std::string name;
    std::uint64_t duration;
    std::int64_t level;
};

struct task
{
    std::vector<task_entry> kernels;
};

class profiler_task
{
public:
    profiler_task(const char * task_name, int idx);
    ~profiler_task();

private:
    const char * task_name_;
    int idx;
};

class profiler
{
public:
    profiler();
    ~profiler();
    static profiler_task start_task(const char * task_name);
    static std::uint64_t get_time();
    static profiler * get_instance();
    task & get_task();
    std::int64_t & get_current_level();
    std::int64_t & get_kernel_count();
    static bool is_profiling_enabled();
    static void end_task(const char * task_name, int idx);

private:
    std::int64_t current_level = 0;
    std::int64_t kernel_count  = 0;
    task task_;
    static std::mutex mutex_;
};

} // namespace internal
} // namespace daal

#endif // __SERVICE_PROFILER_H__
