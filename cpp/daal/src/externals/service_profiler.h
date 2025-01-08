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

#ifndef __SERVICE_PROFILER_H__
#define __SERVICE_PROFILER_H__

#ifdef ONEDAL_KERNEL_PROFILER
    /* Here if oneDAL kernel profiling is enabled in the build */
    #include <ittnotify.h>
#endif

#define DAAL_ITTNOTIFY_CONCAT2(x, y) x##y
#define DAAL_ITTNOTIFY_CONCAT(x, y)  DAAL_ITTNOTIFY_CONCAT2(x, y)

#define DAAL_ITTNOTIFY_UNIQUE_ID __LINE__

#define DAAL_ITTNOTIFY_SCOPED_TASK(name) \
    daal::internal::ProfilerTask DAAL_ITTNOTIFY_CONCAT(__profiler_task__, DAAL_ITTNOTIFY_UNIQUE_ID) = daal::internal::Profiler::startTask(#name);

namespace daal
{
namespace internal
{
/**
 * Defines a logical unit of work to be tracked by performance profilier.
 */
class ProfilerTask
{
public:
    /**
     * Constructs a task with a given name.
     * \param[in] taskName   Name of the task.
     */
    ProfilerTask(const char * taskName);
    ~ProfilerTask();

private:
    const char * _taskName;
#ifdef ONEDAL_KERNEL_PROFILER
    /* Here if oneDAL kernel profiling is enabled */
    __itt_string_handle * _handle; /* The task string handle */
    __itt_domain * _domain;        /* Pointer to the domain of the task */
#endif
};

/**
 * Global performance profiler.
 *
 * By default this class is a stub in the library and its redefinition will be in C++ Bechmarks.
 * If oneDAL kernel profiling is enabled, the profiler uses Task API from <ittnotify.h>
 */
class Profiler
{
public:
    /**
     * Start the task to be profiled.
     * \param[in] taskName   Name of the task.
     */
    static ProfilerTask startTask(const char * taskName);

    /**
     * Start the task to profile.
     * \param[in] taskName   Name of the task.
     */
    static void endTask(const char * taskName);

#ifdef ONEDAL_KERNEL_PROFILER
    /* Here if oneDAL kernel profiling is enabled */

    /**
     * Get pointer to a global profiler state.
     * \return Pointer to a global profiler state.
     */
    static Profiler * getInstance()
    {
        static Profiler instance;
        return &instance;
    }

    /**
     * Get pointer to the ITT domain associated with the profiler.
     * \return Pointer to the ITT domain.
     */
    static __itt_domain * getDomain()
    {
        return (getInstance())->_domain;
    }

private:
    /**
     * Construct the profiler.
     */
    Profiler()
    {
        _domain = __itt_domain_create("oneDAL");
    }
    ~Profiler() {}
    __itt_domain * _domain; /* Pointer to the ITT domain */
#endif
};

} // namespace internal
} // namespace daal

#endif // __SERVICE_PROFILER_H__
