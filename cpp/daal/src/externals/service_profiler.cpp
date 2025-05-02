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

namespace daal
{
namespace internal
{
#ifdef ONEDAL_KERNEL_PROFILER

ProfilerTask::ProfilerTask(const char * taskName) : _taskName(taskName)
{
    _handle = __itt_string_handle_create(taskName);

    __itt_task_begin(Profiler::getDomain(), __itt_null, __itt_null, _handle);
}

ProfilerTask::~ProfilerTask()
{
    Profiler::endTask(_taskName);
}

ProfilerTask Profiler::startTask(const char * taskName)
{
    return ProfilerTask(taskName);
}

void Profiler::endTask(const char * taskName)
{
    __itt_task_end(Profiler::getDomain());
}

#else
ProfilerTask Profiler::startTask(const char * taskName)
{
    return ProfilerTask(taskName);
}

void Profiler::endTask(const char * taskName) {}

ProfilerTask::ProfilerTask(const char * taskName) : _taskName(taskName) {}

ProfilerTask::~ProfilerTask()
{
    Profiler::endTask(_taskName);
}
#endif

} // namespace internal
} // namespace daal
