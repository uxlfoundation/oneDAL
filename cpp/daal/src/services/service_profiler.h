/* file: service_profiler.h */
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
//  Profiler for time measurement of kernels
//--
*/
#pragma once

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "services/daal_defines.h"
#include "services/library_version_info.h"

#ifdef _WIN32
    #define PRETTY_FUNCTION __FUNCSIG__
#else
    #define PRETTY_FUNCTION __PRETTY_FUNCTION__
#endif

#define __SERVICE_PROFILER_H__

// Some Windows builds compile DAAL without /EHsc, where a try block does not compile.
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    #define DAAL_PROFILER_EXCEPTIONS_ENABLED
#endif

#define DAAL_PROFILER_CONCAT2(x, y) x##y
#define DAAL_PROFILER_CONCAT(x, y)  DAAL_PROFILER_CONCAT2(x, y)
#define DAAL_PROFILER_UNIQUE_ID     __LINE__

#define DAAL_PROFILER_STRINGIFY2(x) #x
#define DAAL_PROFILER_STRINGIFY(x)  DAAL_PROFILER_STRINGIFY2(x)

#define DAAL_PROFILER_MACRO_1(name)                       daal::internal::profiler::start_task(#name)
#define DAAL_PROFILER_MACRO_2(name, queue)                daal::internal::profiler::start_task(#name)
#define DAAL_PROFILER_GET_MACRO(arg_1, arg_2, MACRO, ...) MACRO

// LOGGER OUTPUT
// Each call site is logged the first time it runs (every time at level 5), as one write, so a
// stderr pipe nobody drains does not fill up and lines from different threads do not interleave.
#define DAAL_PROFILER_LOG(...)                                                                                      \
    do                                                                                                              \
    {                                                                                                               \
        if (daal::internal::profiler_claim_log_site(__FILE__ ":" DAAL_PROFILER_STRINGIFY(__LINE__)))                \
        {                                                                                                           \
            std::ostringstream profiler_log_line_;                                                                  \
            profiler_log_line_ << "-----------------------------------------------------------------------------\n" \
                               << "File: " << __FILE__ << ", Line: " << __LINE__ << '\n';                           \
            if (daal::internal::is_service_debug_enabled()) profiler_log_line_ << PRETTY_FUNCTION << '\n';          \
            profiler_log_line_ << __VA_ARGS__;                                                                      \
            const std::string profiler_log_text_ = profiler_log_line_.str();                                        \
            daal::internal::profiler_log(profiler_log_text_.data(), profiler_log_text_.size());                     \
        }                                                                                                           \
    } while (0)

// ARGS LOGGING
#define DAAL_PROFILER_LOG_ARGS(task_name, ...)                                                                                                    \
    DAAL_PROFILER_LOG("Profiler task_name: " << #task_name << " Printed args: " << daal::internal::profiler_named_args(#__VA_ARGS__, __VA_ARGS__) \
                                             << '\n')

#define DAAL_PROFILER_TASK_WITH_ARGS(task_name, ...)                                                                                          \
    daal::internal::profiler_task DAAL_PROFILER_CONCAT(__profiler_task__, DAAL_PROFILER_UNIQUE_ID) = [&]() -> daal::internal::profiler_task { \
        if (daal::internal::is_profiler_enabled())                                                                                            \
        {                                                                                                                                     \
            if (daal::internal::is_logger_enabled())                                                                                          \
            {                                                                                                                                 \
                DAAL_PROFILER_LOG_ARGS(task_name, __VA_ARGS__);                                                                               \
            }                                                                                                                                 \
            return daal::internal::profiler::start_task(#task_name);                                                                          \
        }                                                                                                                                     \
        return daal::internal::profiler::start_task(nullptr);                                                                                 \
    }()

#define DAAL_PROFILER_TASK(...)                                                                                                               \
    daal::internal::profiler_task DAAL_PROFILER_CONCAT(__profiler_task__, DAAL_PROFILER_UNIQUE_ID) = [&]() -> daal::internal::profiler_task { \
        if (daal::internal::is_profiler_enabled())                                                                                            \
        {                                                                                                                                     \
            if (daal::internal::is_logger_enabled())                                                                                          \
            {                                                                                                                                 \
                DAAL_PROFILER_LOG("Profiler task_name: " << #__VA_ARGS__ << '\n');                                                            \
            }                                                                                                                                 \
            return DAAL_PROFILER_GET_MACRO(__VA_ARGS__, DAAL_PROFILER_MACRO_2, DAAL_PROFILER_MACRO_1, FICTIVE)(__VA_ARGS__);                  \
        }                                                                                                                                     \
        return daal::internal::profiler::start_task(nullptr);                                                                                 \
    }()

#define DAAL_PROFILER_THREADING_TASK(task_name)                                                                                               \
    daal::internal::profiler_task DAAL_PROFILER_CONCAT(__profiler_task__, DAAL_PROFILER_UNIQUE_ID) = [&]() -> daal::internal::profiler_task { \
        if (daal::internal::is_profiler_enabled())                                                                                            \
        {                                                                                                                                     \
            return daal::internal::profiler::start_threading_task(#task_name);                                                                \
        }                                                                                                                                     \
        return daal::internal::profiler::start_task(nullptr);                                                                                 \
    }()

#define DAAL_PROFILER_SERVICE_TASK(...)                                                                                                       \
    daal::internal::profiler_task DAAL_PROFILER_CONCAT(__profiler_task__, DAAL_PROFILER_UNIQUE_ID) = [&]() -> daal::internal::profiler_task { \
        if (daal::internal::is_service_debug_enabled())                                                                                       \
        {                                                                                                                                     \
            if (daal::internal::is_logger_enabled())                                                                                          \
            {                                                                                                                                 \
                DAAL_PROFILER_LOG("Profiler task_name: " << #__VA_ARGS__ << '\n');                                                            \
            }                                                                                                                                 \
            return DAAL_PROFILER_GET_MACRO(__VA_ARGS__, DAAL_PROFILER_MACRO_2, DAAL_PROFILER_MACRO_1, FICTIVE)(__VA_ARGS__);                  \
        }                                                                                                                                     \
        return daal::internal::profiler::start_task(nullptr);                                                                                 \
    }()

inline static constexpr int PROFILER_MODE_OFF       = 0;
inline static constexpr int PROFILER_MODE_LOGGER    = 1;
inline static constexpr int PROFILER_MODE_TRACER    = 2;
inline static constexpr int PROFILER_MODE_ANALYZER  = 3;
inline static constexpr int PROFILER_MODE_ALL_TOOLS = 4;
inline static constexpr int PROFILER_MODE_DEBUG     = 5;

namespace daal
{
struct profiler_region_hooks;

namespace internal
{

// Every shared object that includes this header reads the mode on its own; they all get the same answer.
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
            if (errno == 0 && endptr != verbose_str && *endptr == '\0' && parsed >= 0 && parsed <= 5) result = static_cast<int>(parsed);
        }
        return result;
    }();
    return cached;
}

inline void profiler_write_named_args(std::ostream & /*out*/, const char * /*names*/) {}

template <typename T, typename... Rest>
inline void profiler_write_named_args(std::ostream & out, const char * names, const T & value, Rest &&... rest)
{
    while (*names == ' ') ++names;
    const char * comma = strchr(names, ',');
    out.write(names, comma ? comma - names : static_cast<std::streamsize>(strlen(names)));
    out << ": " << value << "; ";
    if (comma) profiler_write_named_args(out, comma + 1, std::forward<Rest>(rest)...);
}

/// Formats `values...` as "name: value; " pairs, the names taken from the stringized argument list.
template <typename... Args>
inline std::string profiler_named_args(const char * names, Args &&... values)
{
    std::ostringstream out;
    profiler_write_named_args(out, names, std::forward<Args>(values)...);
    return out.str();
}

inline static bool is_service_debug_enabled()
{
    static const bool service_debug_value = [] {
        int value = daal_verbose_mode();
        return value == PROFILER_MODE_DEBUG;
    }();
    return service_debug_value;
}

inline static bool is_logger_enabled()
{
    static const bool logger_value = [] {
        int value = daal_verbose_mode();
        return value == PROFILER_MODE_LOGGER || value == PROFILER_MODE_ALL_TOOLS || value == PROFILER_MODE_DEBUG;
    }();
    return logger_value;
}

inline static bool is_tracer_enabled()
{
    static const bool verbose_value = [] {
        int value = daal_verbose_mode();
        return value == PROFILER_MODE_TRACER || value == PROFILER_MODE_ALL_TOOLS || value == PROFILER_MODE_DEBUG;
    }();
    return verbose_value;
}

inline static bool is_profiler_enabled()
{
    static const bool profiler_value = [] { return daal_verbose_mode() != PROFILER_MODE_OFF; }();
    return profiler_value;
}

inline static bool is_analyzer_enabled()
{
    static const bool profiler_value = [] {
        int value = daal_verbose_mode();
        return value == PROFILER_MODE_ANALYZER || value == PROFILER_MODE_ALL_TOOLS || value == PROFILER_MODE_DEBUG;
    }();
    return profiler_value;
}

/// Whether a call site is to be logged now: the first time `site` is seen, or always at level 5.
///
/// @param[in] site  Key of the call site, such as "file:line"
DAAL_EXPORT bool profiler_claim_log_site(const char * site);

/// Writes `text` to stderr in one piece, so lines from different threads do not interleave.
///
/// @param[in] text    Whole lines
/// @param[in] length  Number of characters in `text`
DAAL_EXPORT void profiler_log(const char * text, std::size_t length);

/// Opens a task on the calling thread, under the task the thread is in or, on a thread running
/// a chunk of a parallel region, under the task that started the region.
///
/// @param[in] task_name  Name of the task; a string literal, kept by address
/// @param[in] threading  Whether the task is the body of a parallel region
///
/// @return The stack depth to pass to profiler_end_task()
DAAL_EXPORT int profiler_begin_task(const char * task_name, bool threading);

/// Closes the task opened at `depth` on the calling thread, and any task still open above it.
///
/// @param[in] depth  Value returned by profiler_begin_task()
DAAL_EXPORT void profiler_end_task(int depth);

/// One task of the merged task tree, with the figures of every thread that ran it.
struct profiler_report_row
{
    std::string name;
    std::int64_t parent; // index of the parent row, -1 for a top-level task
    std::int64_t depth;
    bool parallel;         // ran inside a parallel region
    std::uint64_t count;   // number of runs
    std::uint64_t time_ns; // wall time; for a parallel task, the busiest thread's time
    std::uint64_t self_ns; // wall time not spent in child tasks
    std::uint64_t max_ns;  // longest run
    std::uint64_t threads; // number of threads that ran the task
};

/// Returns the merged task tree recorded so far, parents before children. Open tasks are skipped.
DAAL_EXPORT std::vector<profiler_report_row> profiler_report();

/// Writes the analyzer report for everything recorded so far.
DAAL_EXPORT void profiler_print_report(std::ostream & out);

/// Drops everything recorded so far. No task may be open on any thread.
DAAL_EXPORT void profiler_reset();

/// The callbacks the threading layer calls around every chunk of a parallel region.
DAAL_EXPORT const daal::profiler_region_hooks & profiler_threading_hooks();

/// Writes the version lines when the profiler is on, and arranges for the analyzer report at exit.
DAAL_EXPORT void print_header();

class profiler_task
{
public:
    profiler_task() = default;
    explicit profiler_task(int depth) : depth_(depth) {}
    inline ~profiler_task();

    profiler_task(const profiler_task &)             = delete;
    profiler_task & operator=(const profiler_task &) = delete;

    profiler_task(profiler_task && other) noexcept : depth_(other.depth_) { other.depth_ = -1; }

    profiler_task & operator=(profiler_task && other) noexcept
    {
        if (this != &other)
        {
            depth_       = other.depth_;
            other.depth_ = -1;
        }
        return *this;
    }

private:
    int depth_ = -1;
};

class profiler
{
public:
    /// Starts a profiling task. Invoked by the DAAL_PROFILER_TASK macro.
    ///
    /// @param[in] task_name  The name of the task, or nullptr for a task that records nothing
    ///
    /// @return An object that ends the task when destroyed
    inline static profiler_task start_task(const char * task_name)
    {
        if (!task_name) return profiler_task();
        return profiler_task(profiler_begin_task(task_name, false));
    }

    /// Starts a task that is the body of a parallel region. Invoked by DAAL_PROFILER_THREADING_TASK.
    ///
    /// @param[in] task_name  The name of the task, or nullptr for a task that records nothing
    inline static profiler_task start_threading_task(const char * task_name)
    {
        if (!task_name) return profiler_task();
        return profiler_task(profiler_begin_task(task_name, true));
    }
};

inline profiler_task::~profiler_task()
{
    if (depth_ >= 0)
    {
#ifdef DAAL_PROFILER_EXCEPTIONS_ENABLED
        try
        {
#endif
            profiler_end_task(depth_);
#ifdef DAAL_PROFILER_EXCEPTIONS_ENABLED
        }
        catch (std::exception & e)
        {
            std::cerr << "DAAL KERNEL_PROFILER: unexpected error while ending a task: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "DAAL KERNEL_PROFILER: unexpected error while ending a task" << std::endl;
        }
#endif
    }
}

} // namespace internal
} // namespace daal

extern "C"
{
    /// Writes the analyzer report for everything recorded so far, for callers without C++ linkage
    /// such as Python through ctypes.
    ///
    /// @param[in] path   File to append the report to, or nullptr for stderr
    /// @param[in] clear  If not 0, everything recorded so far is dropped afterwards
    DAAL_EXPORT void _onedal_profiler_report(const char * path, int clear);
}
