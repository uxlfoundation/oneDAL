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
//  Profiler for time and memory measurement of kernels
//--
*/
#pragma once

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <exception>
#include "services/daal_defines.h"
#include "services/library_version_info.h"
#include "src/services/service_profiler_site.h"

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

#define DAAL_PROFILER_MACRO_1(name)                       daal::internal::profiler::start_task(#name)
#define DAAL_PROFILER_MACRO_2(name, queue)                daal::internal::profiler::start_task(#name, queue)
#define DAAL_PROFILER_GET_MACRO(arg_1, arg_2, MACRO, ...) MACRO

#define DAAL_PROFILER_STRINGIFY2(x) #x
#define DAAL_PROFILER_STRINGIFY(x)  DAAL_PROFILER_STRINGIFY2(x)

// LOGGER OUTPUT
// Logs a call of a profiled task: where it was called from, then the streamed message. On stderr each
// call site is logged once (see profiler_log), and is not formatted again after that.
#define DAAL_PROFILER_LOG(...)                                                                                      \
    do                                                                                                              \
    {                                                                                                               \
        static std::atomic<bool> profiler_site_logged_ { false };                                                   \
        if (daal::internal::is_every_call_logged()                                                                  \
            || (!profiler_site_logged_.load(std::memory_order_relaxed) && !profiler_site_logged_.exchange(true)))   \
        {                                                                                                           \
            std::ostringstream profiler_log_line_;                                                                  \
            profiler_log_line_ << "-----------------------------------------------------------------------------\n" \
                               << "File: " << __FILE__ << ", Line: " << __LINE__ << '\n';                           \
            if (daal::internal::is_service_debug_enabled()) profiler_log_line_ << PRETTY_FUNCTION << '\n';          \
            profiler_log_line_ << __VA_ARGS__;                                                                      \
            daal::internal::profiler_log(__FILE__ ":" DAAL_PROFILER_STRINGIFY(__LINE__), profiler_log_line_.str()); \
        }                                                                                                           \
    } while (0)

// ARGS LOGGING
#define DAAL_PROFILER_LOG_ARGS(task_name, ...)                                                                                                    \
    DAAL_PROFILER_LOG("Profiler task_name: " << #task_name << " Printed args: " << daal::internal::profiler_named_args(#__VA_ARGS__, __VA_ARGS__) \
                                             << '\n')

#define DAAL_PROFILER_TASK_WITH_ARGS(task_name, ...)                                                                                               \
    daal::internal::profiler_task DAAL_PROFILER_CONCAT(__profiler_task__, DAAL_PROFILER_UNIQUE_ID) = [&]() -> daal::internal::profiler_task {      \
        if (daal::internal::is_profiler_enabled())                                                                                                 \
        {                                                                                                                                          \
            if (daal::internal::is_logger_enabled())                                                                                               \
            {                                                                                                                                      \
                DAAL_PROFILER_LOG_ARGS(task_name, __VA_ARGS__);                                                                                    \
            }                                                                                                                                      \
            if (daal::internal::is_analyzer_enabled())                                                                                             \
            {                                                                                                                                      \
                return daal::internal::profiler::start_task_with_args(#task_name, daal::internal::profiler_named_args(#__VA_ARGS__, __VA_ARGS__)); \
            }                                                                                                                                      \
            return daal::internal::profiler::start_task(#task_name);                                                                               \
        }                                                                                                                                          \
        return daal::internal::profiler::start_task(nullptr);                                                                                      \
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

namespace daal
{
struct profiler_region_hooks;

namespace internal
{

inline void profiler_write_named_args(std::ostream & /*out*/, const char * /*names*/) {}

template <typename T, typename... Rest>
inline void profiler_write_named_args(std::ostream & out, const char * names, const T & value, Rest &&... rest)
{
    while (std::isspace(static_cast<unsigned char>(*names))) ++names;
    const char * comma = strchr(names, ',');
    out.write(names, comma ? comma - names : static_cast<std::streamsize>(strlen(names)));
    out << ": " << value << "; ";
    if (comma) profiler_write_named_args(out, comma + 1, std::forward<Rest>(rest)...);
}

/// Formats `value...` as "name: value; " pairs, the names taken from the stringized argument list `names`.
template <typename... Args>
inline std::string profiler_named_args(const char * names, Args &&... values)
{
    // One stream per thread: constructing a stream costs more than formatting a few numbers.
    thread_local std::ostringstream out;
    out.str(std::string());
    out.clear();
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

inline static bool is_analyzer_enabled()
{
    static const bool profiler_value = [] {
        int value = daal_verbose_mode();
        return value == PROFILER_MODE_ANALYZER || value == PROFILER_MODE_ALL_TOOLS || value == PROFILER_MODE_DEBUG;
    }();
    return profiler_value;
}

/// Whether allocations are recorded. Only the analyzer report and the timeline show them;
/// ONEDAL_PROFILER_MEMORY=0 turns them off.
inline static bool is_memory_tracking_enabled()
{
    static const bool value = [] {
        const char * memory = std::getenv("ONEDAL_PROFILER_MEMORY");
        const bool off      = memory && std::strcmp(memory, "0") == 0;
        return !off && (is_analyzer_enabled() || profiler_trace_path() != nullptr);
    }();
    return value;
}

/// Form of the analyzer report, set by ONEDAL_PROFILER_REPORT.
enum class profiler_report_format
{
    text,   // summary, task tree, hotspots and memory
    tree,   // the task tree alone
    folded, // one line per call path with its self time, for flame graph tools
    json,   // every row of the task tree with all its figures
    health  // the health checks and the memory summary alone
};

inline static profiler_report_format report_format()
{
    static const profiler_report_format value = [] {
        const char * report = std::getenv("ONEDAL_PROFILER_REPORT");
        if (!report) return profiler_report_format::text;
        if (std::strcmp(report, "tree") == 0) return profiler_report_format::tree;
        if (std::strcmp(report, "folded") == 0) return profiler_report_format::folded;
        if (std::strcmp(report, "json") == 0) return profiler_report_format::json;
        if (std::strcmp(report, "health") == 0) return profiler_report_format::health;
        return profiler_report_format::text;
    }();
    return value;
}

/// Whether the report is something other than the full text report, which has no room for the
/// version lines.
inline static bool is_tree_only_report()
{
    return report_format() != profiler_report_format::text;
}

/// Whether ONEDAL_PROFILER_REPORT_AT=call asks for a report after each outermost call instead of
/// one at exit.
inline static bool is_report_per_call()
{
    static const bool value = [] {
        const char * at = std::getenv("ONEDAL_PROFILER_REPORT_AT");
        return at && std::strcmp(at, "call") == 0;
    }();
    return value;
}

/// Whether ONEDAL_PROFILER_OUTPUT asks for every call to be logged, rather than each call site once.
inline static bool is_every_call_logged()
{
    static const bool value = [] {
        const char * path = std::getenv("ONEDAL_PROFILER_OUTPUT");
        return path && *path;
    }();
    return value;
}

/// Writes `text` to the profiler output: the file named by ONEDAL_PROFILER_OUTPUT, or stderr.
///
/// @param[in] once_key  If not nullptr, `text` is written only the first time this key is seen,
///                      unless the output is a file, which takes every call.
/// @param[in] text      Whole lines, written in one piece.
DAAL_EXPORT void profiler_log(const char * once_key, const std::string & text);

inline void print_header()
{
    if (is_profiler_enabled() && !is_tree_only_report())
    {
        daal::services::LibraryVersionInfo ver;
        std::ostringstream out;
        out << "Major version:          " << ver.majorVersion << '\n';
        out << "Minor version:          " << ver.minorVersion << '\n';
        out << "Update version:         " << ver.updateVersion << '\n';
        out << "Product status:         " << ver.productStatus << '\n';
        out << "Build:                  " << ver.build << '\n';
        out << "Build revision:         " << ver.build_rev << '\n';
        out << "Name:                   " << ver.name << '\n';
        out << "Processor optimization: " << ver.processor << '\n';
        out << '\n';
        profiler_log(nullptr, out.str());
    }
}

/// Memory an allocation lives in. USM shared memory counts as device memory.
enum class profiler_memory_kind : int
{
    host   = 0,
    device = 1
};

/// Direction of an explicit copy between host and device memory.
enum class profiler_transfer_kind : int
{
    host_to_device = 0,
    device_to_host = 1
};

/// Opens a task on the calling thread.
///
/// @param[in] task_name    Name of the task; copied the first time it is seen at a tree position
/// @param[in] threading    Whether the task is the body of a parallel region
///
/// @return The stack depth to pass to profiler_end_task()
DAAL_EXPORT int profiler_begin_task(const char * task_name, bool threading);

/// Opens a task that the analyzer reports separately for each distinct `args`, up to a few per task.
///
/// @param[in] task_name    Name of the task
/// @param[in] args         The task's arguments as "name: value; " pairs
///
/// @return The stack depth to pass to profiler_end_task()
DAAL_EXPORT int profiler_begin_task_with_args(const char * task_name, const std::string & args);

/// Closes the task opened at `depth` on the calling thread, and any task still open above it.
///
/// @param[in] depth    Value returned by profiler_begin_task()
DAAL_EXPORT void profiler_end_task(int depth);

/// Records an allocation, charged to the innermost task open on the calling thread.
///
/// @param[in] ptr      Address of the allocation; nullptr is ignored
/// @param[in] bytes    Size of the allocation
/// @param[in] kind     Memory the allocation lives in
DAAL_EXPORT void profiler_record_alloc(const void * ptr, std::size_t bytes, profiler_memory_kind kind);

/// Records a deallocation. Pointers whose allocation was not recorded are ignored.
///
/// @param[in] ptr      Address passed to the deallocation function
DAAL_EXPORT void profiler_record_free(const void * ptr);

/// Records an explicit copy between host and device memory.
///
/// @param[in] bytes    Size of the copy
/// @param[in] kind     Direction of the copy
DAAL_EXPORT void profiler_record_transfer(std::size_t bytes, profiler_transfer_kind kind);

/// One task of the merged task tree. Arrays are indexed by profiler_memory_kind or
/// profiler_transfer_kind; memory figures include the task's children.
struct profiler_report_row
{
    std::string name;
    std::int64_t parent; // index of the parent row, -1 for a top-level task
    std::int64_t depth;
    bool parallel;                // ran inside a parallel region
    std::uint64_t count;          // number of runs
    std::uint64_t time_ns;        // wall time; for a parallel task, along the busiest thread of each run
    std::uint64_t cpu_ns;         // time summed over all threads
    std::uint64_t self_ns;        // wall time not spent in child tasks
    std::uint64_t min_ns;         // shortest run
    std::uint64_t max_ns;         // longest run
    std::uint64_t first_ns;       // the first run, which on a GPU includes compiling its kernels
    std::uint64_t threads;        // number of threads; for a parallel task, the most in one run of the region
    std::uint64_t busiest_ns;     // for a parallel task: the busiest thread's time, summed over runs
    std::uint64_t mean_thread_ns; // for a parallel task: the average thread's time, summed over runs
    std::uint64_t alloc_bytes[2];
    std::uint64_t alloc_count[2];
    std::uint64_t free_bytes[2];
    std::uint64_t free_count[2];
    std::uint64_t peak_bytes[2]; // highest live memory above the level at task start
    std::uint64_t live_bytes[2]; // allocated in the task and not freed when the report was built
    std::uint64_t live_count[2];
    std::uint64_t transfer_bytes[2];
    std::uint64_t transfer_count[2];
};

/// Returns the merged task tree recorded so far, parents before children. Open tasks are skipped.
DAAL_EXPORT std::vector<profiler_report_row> profiler_report();

/// Writes the analyzer report for everything recorded so far.
///
/// @param[in] out  Stream to write to
DAAL_EXPORT void profiler_print_report(std::ostream & out);

/// Writes the report in the given form, as ONEDAL_PROFILER_REPORT and ONEDAL_PROFILER_MIN_PERCENT
/// would select it.
///
/// @param[in] out          Stream to write to
/// @param[in] format       Form of the report
/// @param[in] min_percent  Rows shorter than this share of the profiled time are folded into one per parent
DAAL_EXPORT void profiler_print_report(std::ostream & out, profiler_report_format format, double min_percent);

/// Drops everything recorded so far. No task may be open on any thread.
DAAL_EXPORT void profiler_reset();

/// Sets every figure recorded so far to zero, keeping the timeline. No task may be open on any thread.
DAAL_EXPORT void profiler_clear();

/// Callbacks the threading layer calls around every chunk of a parallel region.
DAAL_EXPORT const daal::profiler_region_hooks & profiler_threading_hooks();

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
    /// Starts a profiling task
    ///
    /// @param[in] task_name The name of the task, or nullptr for a task that records nothing
    ///
    /// @return An object that ends the task when destroyed. Invoked by the DAAL_PROFILER_TASK macro.
    inline static profiler_task start_task(const char * task_name)
    {
        if (!task_name) return profiler_task();
        return profiler_task(profiler_begin_task(task_name, false));
    }

    /// Starts a profiling task reported separately for each distinct set of arguments
    ///
    /// @param[in] task_name The name of the task
    /// @param[in] args      The task's arguments, as formatted by profiler_named_args()
    inline static profiler_task start_task_with_args(const char * task_name, const std::string & args)
    {
        return profiler_task(profiler_begin_task_with_args(task_name, args));
    }

    /// Starts a task that is the body of a parallel region. Invoked by DAAL_PROFILER_THREADING_TASK.
    ///
    /// @param[in] task_name The name of the task, or nullptr for a task that records nothing
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
    /// @param[in] path     File to append the report to, or nullptr for the profiler output
    /// @param[in] clear    If not 0, the figures are set to zero afterwards, as profiler_clear() does
    DAAL_EXPORT void _onedal_profiler_report(const char * path, int clear);
}
