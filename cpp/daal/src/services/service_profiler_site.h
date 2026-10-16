/* file: service_profiler_site.h */
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

/*
//++
//  Profiler mode and the call sites of parallel loops, kept apart from service_profiler.h so the
//  threading layer can name its regions without the profiler's other dependencies.
//--
*/

#ifndef __SERVICE_PROFILER_SITE_H__
#define __SERVICE_PROFILER_SITE_H__

#include <cerrno>
#include <cstdlib>
#include "services/daal_defines.h"

namespace daal
{
namespace internal
{
inline static constexpr int PROFILER_MODE_OFF       = 0;
inline static constexpr int PROFILER_MODE_LOGGER    = 1;
inline static constexpr int PROFILER_MODE_TRACER    = 2;
inline static constexpr int PROFILER_MODE_ANALYZER  = 3;
inline static constexpr int PROFILER_MODE_ALL_TOOLS = 4;
inline static constexpr int PROFILER_MODE_DEBUG     = 5;

// Every shared object that includes this header reads the mode on its own, and they all get the
// same answer.
inline static int daal_verbose_mode()
{
    static const int cached = [] {
        const char * verbose_str = std::getenv("ONEDAL_VERBOSE");
        int result               = PROFILER_MODE_OFF;
        if (verbose_str)
        {
            char * endptr = nullptr;
            errno         = 0;
            long parsed   = std::strtol(verbose_str, &endptr, 10);
            if (errno == 0 && endptr != verbose_str && *endptr == '\0' && parsed >= 0 && parsed <= 5)
            {
                result = static_cast<int>(parsed);
            }
        }
        return result;
    }();
    return cached;
}

/// Path of the timeline file requested through ONEDAL_PROFILER_TRACE, or nullptr.
inline static const char * profiler_trace_path()
{
    static const char * path = [] {
        const char * value = std::getenv("ONEDAL_PROFILER_TRACE");
        return (value && *value) ? value : nullptr;
    }();
    return path;
}

inline static bool is_profiler_enabled()
{
    static const bool profiler_value = [] { return daal_verbose_mode() != PROFILER_MODE_OFF || profiler_trace_path() != nullptr; }();
    return profiler_value;
}

/// Where a parallel loop was called from; the analyzer names the loop's region after it.
struct profiler_site
{
    const char * file;
    int line;
    const char * function;
};

/// Sets the call site of the next parallel region the calling thread starts, or clears it.
///
/// @param[in] site     The call site, which must live until the region starts, or nullptr
DAAL_EXPORT void profiler_set_region_site(const profiler_site * site);

/// Names the parallel region started while it is alive after the given call site.
/// Does nothing while the profiler is off.
class profiler_region_site
{
public:
    profiler_region_site(const char * file, int line, const char * function) : site_ { file, line, function }
    {
        if (is_profiler_enabled()) profiler_set_region_site(&site_);
    }
    ~profiler_region_site()
    {
        if (is_profiler_enabled()) profiler_set_region_site(nullptr);
    }
    profiler_region_site(const profiler_region_site &)             = delete;
    profiler_region_site & operator=(const profiler_region_site &) = delete;

private:
    profiler_site site_;
};

} // namespace internal
} // namespace daal

#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
    #define DAAL_PROFILER_CALLER_FILE     __builtin_FILE()
    #define DAAL_PROFILER_CALLER_LINE     __builtin_LINE()
    #define DAAL_PROFILER_CALLER_FUNCTION __builtin_FUNCTION()
#else
    #define DAAL_PROFILER_CALLER_FILE     nullptr
    #define DAAL_PROFILER_CALLER_LINE     0
    #define DAAL_PROFILER_CALLER_FUNCTION nullptr
#endif

#endif
