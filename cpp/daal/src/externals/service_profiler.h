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

#include <sys/time.h>
#include <time.h>
#include <cstdint>
#include <cstring>
#include <map>
#include <vector>

#ifndef __SERVICE_PROFILER_H__
    #define __SERVICE_PROFILER_H__

    #define DAAL_ITTNOTIFY_CONCAT2(x, y) x##y
    #define DAAL_ITTNOTIFY_CONCAT(x, y)  DAAL_ITTNOTIFY_CONCAT2(x, y)

    #define DAAL_ITTNOTIFY_UNIQUE_ID __LINE__

    #define DAAL_ITTNOTIFY_SCOPED_TASK(name)                                                               \
        daal::internal::profiler_task DAAL_ITTNOTIFY_CONCAT(__profiler_taks__, DAAL_ITTNOTIFY_UNIQUE_ID) = \
            daal::internal::profiler::start_task(#name);

namespace daal
{
namespace internal
{

struct task
{
    static const std::uint64_t MAX_KERNELS = 256;
    std::map<const char *, std::uint64_t> kernels;
    std::uint64_t current_kernel = 0;
    std::uint64_t time_kernels[MAX_KERNELS];
    void clear();
};

class profiler_task
{
public:
    profiler_task(const char * task_name);
    ~profiler_task();

private:
    const char * task_name_;
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

    static void end_task(const char * task_name);

private:
    std::uint64_t start_time;
    task task_;
};

} // namespace internal
} // namespace daal

#endif // __SERVICE_PROFILER_H__
