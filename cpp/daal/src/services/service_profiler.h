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

#include <chrono>
#include <thread>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <algorithm>
#include <exception>
#include <unordered_map>
#include <unordered_set>
#include "services/library_version_info.h"

#ifdef _WIN32
    #define PRETTY_FUNCTION __FUNCSIG__
#else
    #define PRETTY_FUNCTION __PRETTY_FUNCTION__
#endif

#define __SERVICE_PROFILER_H__

#define DAAL_PROFILER_CONCAT2(x, y) x##y
#define DAAL_PROFILER_CONCAT(x, y)  DAAL_PROFILER_CONCAT2(x, y)
#define DAAL_PROFILER_UNIQUE_ID     __LINE__

#define DAAL_PROFILER_MACRO_1(name)                       daal::internal::profiler::start_task(#name)
#define DAAL_PROFILER_MACRO_2(name, queue)                daal::internal::profiler::start_task(#name, queue)
#define DAAL_PROFILER_GET_MACRO(arg_1, arg_2, MACRO, ...) MACRO

// HEADER OUTPUT
#define DAAL_PROFILER_PRINT_HEADER()                                                                          \
    do                                                                                                        \
    {                                                                                                         \
        std::cerr << "-----------------------------------------------------------------------------" << '\n'; \
        std::cerr << "File: " << __FILE__ << ", Line: " << __LINE__ << '\n';                                  \
        if (daal::internal::is_service_debug_enabled())                                                       \
        {                                                                                                     \
            std::cerr << PRETTY_FUNCTION << '\n';                                                             \
        }                                                                                                     \
    } while (0)

// ARGS LOGGING
#define DAAL_PROFILER_LOG_ARGS(task_name, ...)                                  \
    do                                                                          \
    {                                                                           \
        DAAL_PROFILER_PRINT_HEADER();                                           \
        std::cerr << "Profiler task_name: " << #task_name << " Printed args: "; \
        daal::internal::profiler_log_named_args(#__VA_ARGS__, __VA_ARGS__);     \
        std::cerr << '\n';                                                      \
    } while (0)

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
                DAAL_PROFILER_PRINT_HEADER();                                                                                                 \
                std::cerr << "Profiler task_name: " << #__VA_ARGS__ << std::endl;                                                             \
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
                DAAL_PROFILER_PRINT_HEADER();                                                                                                 \
                std::cerr << "Profiler task_name: " << #__VA_ARGS__ << std::endl;                                                             \
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
namespace internal
{

// Reads ONEDAL_VERBOSE once per process; the C++17 function-local static gives thread-safe
// single initialization and a single storage instance across all TUs (fixes the previous
// namespace-scope `static volatile int` that gave every TU its own copy).
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

inline static std::string format_time_for_output(std::uint64_t time_ns)
{
    std::ostringstream out;
    double time = static_cast<double>(time_ns);
    if (time <= 0)
        out << "0.00s";
    else if (time > 1e9)
        out << std::fixed << std::setprecision(2) << time / 1e9 << "s";
    else if (time > 1e6)
        out << std::fixed << std::setprecision(2) << time / 1e6 << "ms";
    else if (time > 1e3)
        out << std::fixed << std::setprecision(2) << time / 1e3 << "us";
    else
        out << static_cast<std::uint64_t>(time) << "ns";
    return out.str();
}

inline void profiler_log_named_args(const char * /*names*/) {}

template <typename T, typename... Rest>
inline void profiler_log_named_args(const char * names, const T & value, Rest &&... rest)
{
    const char * comma = strchr(names, ',');
    std::string name   = comma ? std::string(names, comma) : std::string(names);
    name.erase(name.begin(), std::find_if(name.begin(), name.end(), [](unsigned char ch) { return !std::isspace(ch); }));
    std::cerr << name << ": " << value << "; ";
    if (comma) profiler_log_named_args(comma + 1, std::forward<Rest>(rest)...);
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
        std::ios::sync_with_stdio(false);
        int value = daal_verbose_mode();
        return value == PROFILER_MODE_TRACER || value == PROFILER_MODE_ALL_TOOLS || value == PROFILER_MODE_DEBUG;
    }();
    return verbose_value;
}

inline static bool is_profiler_enabled()
{
    static const bool profiler_value = [] {
        int value = daal_verbose_mode();
        return value == PROFILER_MODE_LOGGER || value == PROFILER_MODE_TRACER || value == PROFILER_MODE_ANALYZER || value == PROFILER_MODE_ALL_TOOLS
               || value == PROFILER_MODE_DEBUG;
    }();
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

inline void print_header()
{
    if (is_profiler_enabled())
    {
        daal::services::LibraryVersionInfo ver;
        std::cerr << "Major version:          " << ver.majorVersion << '\n';
        std::cerr << "Minor version:          " << ver.minorVersion << '\n';
        std::cerr << "Update version:         " << ver.updateVersion << '\n';
        std::cerr << "Product status:         " << ver.productStatus << '\n';
        std::cerr << "Build:                  " << ver.build << '\n';
        std::cerr << "Build revision:         " << ver.build_rev << '\n';
        std::cerr << "Name:                   " << ver.name << '\n';
        std::cerr << "Processor optimization: " << ver.processor << '\n';
        std::cerr << '\n';
    }
}

struct task_entry
{
    std::int64_t idx;
    std::string name;
    std::uint64_t duration;
    std::int64_t level;
    std::int64_t count;
    bool threading_task;
    std::int64_t parent_idx; // -1 for top-level; index into task::kernels otherwise
};

struct task
{
    std::vector<task_entry> kernels;
};

class profiler_task
{
public:
    inline profiler_task(const char * task_name, int idx) : task_name_(task_name), idx_(idx) {}
    inline profiler_task(const char * task_name, int idx, bool thread) : task_name_(task_name), idx_(idx), is_thread_(thread) {}
    inline ~profiler_task();

    profiler_task(const profiler_task &)             = delete;
    profiler_task & operator=(const profiler_task &) = delete;

    profiler_task(profiler_task && other) noexcept : task_name_(other.task_name_), idx_(other.idx_), is_thread_(other.is_thread_)
    {
        other.task_name_ = nullptr;
        other.idx_       = -1;
        other.is_thread_ = false;
    }

    profiler_task & operator=(profiler_task && other) noexcept
    {
        if (this != &other)
        {
            task_name_ = other.task_name_;
            idx_       = other.idx_;
            is_thread_ = other.is_thread_;

            other.task_name_ = nullptr;
            other.idx_       = -1;
            other.is_thread_ = false;
        }
        return *this;
    }

private:
    const char * task_name_ = nullptr;
    int idx_                = -1;
    bool is_thread_         = false;
};

// Per-thread stacks of currently-open tasks. Two stacks per thread — regular and threading — kept
// separate so a worker thread's threading task can find its own inner nesting without colliding
// with regular tasks opened on the main thread.
//
// Storage: an `unordered_map<std::thread::id, vector>` protected by the profiler's global mutex.
// We do NOT use `thread_local`: this header is included into multiple shared libraries
// (libonedal_core, libonedal_thread, libonedal_dpc, libonedal, per-CPU-variant translation units)
// and different .so boundaries can end up owning different TLS instances for the same OS thread,
// which corrupts the push/pop invariant. The map avoids that entirely — all threads share one
// well-defined instance in the singleton, and mutex-guarded access makes it thread-safe.
using per_thread_stack_map = std::unordered_map<std::thread::id, std::vector<std::int64_t>>;

class profiler
{
public:
    inline profiler() { daal_verbose_mode(); }

    inline ~profiler()
    {
        if (is_analyzer_enabled())
        {
#if (!defined(DAAL_NOTHROW_EXCEPTIONS))
            try
            {
#endif
                merge_tasks();
                const auto & tasks_info  = get_instance()->get_task();
                std::uint64_t total_time = 0;
                std::cerr << "Algorithm tree analyzer" << '\n';

                for (size_t i = 0; i < tasks_info.kernels.size(); ++i)
                {
                    const auto & entry = tasks_info.kernels[i];
                    if (entry.level == 0) total_time += entry.duration;
                }

                for (size_t i = 0; i < tasks_info.kernels.size(); ++i)
                {
                    const auto & entry = tasks_info.kernels[i];
                    std::string prefix;
                    for (std::int64_t lvl = 0; lvl < entry.level; ++lvl) prefix += "|  ";
                    prefix += "|-- ";
                    std::cerr << prefix << entry.name << " time: " << format_time_for_output(entry.duration) << " " << std::fixed
                              << std::setprecision(2) << (total_time > 0 ? (double(entry.duration) / total_time) * 100 : 0.0) << "% " << entry.count
                              << " times in a " << (entry.threading_task ? "parallel" : "sequential") << " region" << '\n';
                }
                std::cerr << "|--(end)" << '\n';
                std::cerr << "DAAL KERNEL_PROFILER: kernels total time " << format_time_for_output(total_time) << '\n';

#if (!defined(DAAL_NOTHROW_EXCEPTIONS))
            }
            catch (std::exception & e)
            {
                std::cerr << e.what() << std::endl;
            }
#endif
        }
    }

    /// Starts a profiling task with the given task name and returns a profiler_task object
    ///
    /// @param[in] task_name The name of the task to be profiled
    ///
    /// @return A profiler_task object containing the task name and a unique task ID. Returns an invalid
    /// profiler_task (nullptr, -1) if task_name is nullptr
    ///
    /// @note Captures the start time before acquiring the profiler mutex (so lock-wait time is not
    /// charged to the task), records the parent task index from the calling thread's regular-task stack
    /// (falling back to the innermost open regular task across threads), derives the nesting level from
    /// the parent, and pushes the entry into tasks_info.kernels.
    /// Invoked by the DAAL_PROFILER_TASK macro.
    inline static profiler_task start_task(const char * task_name)
    {
        if (!task_name) return profiler_task(nullptr, -1);

        const std::uint64_t ns_start = get_time();
        std::lock_guard<std::mutex> lock(global_mutex());
        auto & inst          = *get_instance();
        auto & tasks_info    = inst.get_task();
        auto & rstack        = inst.regular_stacks_[std::this_thread::get_id()];
        std::int64_t new_idx = static_cast<std::int64_t>(tasks_info.kernels.size());

        const std::int64_t parent_idx = rstack.empty() ? inst.innermost_regular_task_idx_ : rstack.back();
        const std::int64_t level      = (parent_idx < 0) ? 0 : tasks_info.kernels[parent_idx].level + 1;

        tasks_info.kernels.push_back({ new_idx, task_name, ns_start, level, 1, false, parent_idx });
        rstack.push_back(new_idx);
        inst.innermost_regular_task_idx_ = new_idx;
        return profiler_task(task_name, static_cast<int>(new_idx));
    }

    /// Starts a threading-specific profiling task with the given task name and returns a profiler_task object
    ///
    /// @param[in] task_name The name of the threading task to be profiled
    ///
    /// @return A profiler_task object containing the task name, a unique task ID, and a threading flag.
    /// Returns an invalid profiler_task (nullptr, -1) if task_name is nullptr
    ///
    /// @note Captures the start time before acquiring the mutex. Parent index comes from the calling
    /// worker thread's threading-task stack (allowing nested threading tasks on the same worker);
    /// if that stack is empty, falls back to the innermost open regular task recorded across threads
    /// (i.e. the task that spawned the parallel region). Logs unique task names once per process
    /// if logging is enabled. Invoked by the DAAL_PROFILER_THREADING_TASK macro.
    inline static profiler_task start_threading_task(const char * task_name)
    {
        if (!task_name) return profiler_task(nullptr, -1);

        const std::uint64_t ns_start = get_time();
        std::lock_guard<std::mutex> lock(global_mutex());

        if (is_logger_enabled())
        {
            auto & inst = *get_instance();
            if (!is_service_debug_enabled())
            {
                if (inst.unique_threading_task_names_.insert(task_name).second)
                {
                    std::cerr << "-----------------------------------------------------------------------------" << '\n';
                    std::cerr << "THREADING Profiler task started on the main rank: " << task_name << '\n';
                }
            }
            else
            {
                std::cerr << "-----------------------------------------------------------------------------" << '\n';
                std::cerr << "THREADING Profiler task started " << task_name << '\n';
            }
        }

        auto & inst          = *get_instance();
        auto & tasks_info    = inst.get_task();
        auto & tstack        = inst.threading_stacks_[std::this_thread::get_id()];
        std::int64_t new_idx = static_cast<std::int64_t>(tasks_info.kernels.size());

        const std::int64_t parent_idx = tstack.empty() ? inst.innermost_regular_task_idx_ : tstack.back();
        const std::int64_t level      = (parent_idx < 0) ? 0 : tasks_info.kernels[parent_idx].level + 1;

        tasks_info.kernels.push_back({ new_idx, task_name, ns_start, level, 1, true, parent_idx });
        tstack.push_back(new_idx);
        return profiler_task(task_name, static_cast<int>(new_idx), true);
    }

    /// Terminates a profiling task and records its duration
    ///
    /// @param[in] task_name The name of the task to end
    /// @param[in] idx_ The index of the task in the tasks_info.kernels vector
    ///
    /// @note If task_name is nullptr, the function returns immediately. Captures the end time
    /// before acquiring the mutex (so the recorded duration excludes lock-wait time), pops the
    /// calling thread's regular-task stack, and restores the global innermost-regular-task index
    /// to the popped task's parent. Logs the task name and duration if tracing is enabled.
    /// Invoked by macros such as DAAL_PROFILER_TASK.
    inline static void end_task(const char * task_name, int idx_)
    {
        if (!task_name) return;
        const std::uint64_t ns_end = get_time();

        std::lock_guard<std::mutex> lock(global_mutex());
        auto & inst       = *get_instance();
        auto & tasks_info = inst.get_task();
        if (idx_ < 0 || static_cast<std::size_t>(idx_) >= tasks_info.kernels.size()) return;

        auto & entry           = tasks_info.kernels[idx_];
        const auto duration    = ns_end - entry.duration;
        entry.duration         = duration;

        auto & rstack = inst.regular_stacks_[std::this_thread::get_id()];
        if (!rstack.empty() && rstack.back() == idx_)
        {
            rstack.pop_back();
        }
        else if (!rstack.empty())
        {
            // RAII invariant violated (should not happen); fall back to scan+erase to stay safe.
            rstack.erase(std::remove(rstack.begin(), rstack.end(), static_cast<std::int64_t>(idx_)), rstack.end());
        }
        inst.innermost_regular_task_idx_ = entry.parent_idx;

        if (is_tracer_enabled()) std::cerr << task_name << " " << format_time_for_output(duration) << '\n';
    }

    /// Terminates a threading-specific profiling task and records its duration
    ///
    /// @param[in] task_name The name of the threading task to end
    /// @param[in] idx_ The index of the task in the tasks_info.kernels vector
    ///
    /// @note If task_name is nullptr or idx_ is invalid, the function returns immediately.
    /// Captures the end time before the mutex, calculates duration, updates the entry, and pops
    /// the calling worker thread's threading-task stack. Logs unique task names and durations if
    /// tracing is enabled. Invoked by the DAAL_PROFILER_THREADING_TASK macro.
    inline static void end_threading_task(const char * task_name, int idx_)
    {
        if (!task_name) return;
        const std::uint64_t ns_end = get_time();

        std::lock_guard<std::mutex> lock(global_mutex());
        auto & inst       = *get_instance();
        auto & tasks_info = inst.get_task();
        if (idx_ < 0 || static_cast<std::size_t>(idx_) >= tasks_info.kernels.size()) return;

        auto & entry        = tasks_info.kernels[idx_];
        const auto duration = ns_end - entry.duration;
        entry.duration      = duration;

        auto & tstack = inst.threading_stacks_[std::this_thread::get_id()];
        if (!tstack.empty() && tstack.back() == idx_)
        {
            tstack.pop_back();
        }
        else if (!tstack.empty())
        {
            tstack.erase(std::remove(tstack.begin(), tstack.end(), static_cast<std::int64_t>(idx_)), tstack.end());
        }

        if (is_tracer_enabled())
        {
            if (inst.unique_threading_task_end_names_.insert(task_name).second)
            {
                std::cerr << "THREADING " << task_name
                          << " finished on the main rank(time could be different for other ranks): " << format_time_for_output(duration) << '\n';
            }
        }
    }

    inline static std::uint64_t get_time()
    {
        auto now = std::chrono::steady_clock::now();
        auto ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
        return static_cast<std::uint64_t>(ns);
    }

    inline static profiler * get_instance()
    {
        static profiler instance;
        return &instance;
    }

    /// Merges sibling tasks with identical names (same parent, same name) into a single entry.
    ///
    /// @note Combines matching tasks in the tasks_info.kernels vector to simplify profiling output.
    /// For non-threading tasks, durations are summed; for threading tasks, the maximum duration is
    /// taken (rough estimate of wall-clock time under parallelism). Skipped in service-debug mode
    /// to preserve per-instance detail. Uses an unordered_map keyed on (parent_idx, name) so the
    /// merge is O(n) instead of the previous O(n^2), and matches on true parent identity rather
    /// than the ambiguous (level, name) pair.
    inline void merge_tasks()
    {
        if (is_service_debug_enabled())
        {
            return;
        }
        auto & tasks_info = get_instance()->get_task();
        auto & kernels    = tasks_info.kernels;

        // Map (parent_idx, name) -> canonical index in the compacted output.
        struct key_t
        {
            std::int64_t parent;
            std::string name;
            bool operator==(const key_t & o) const { return parent == o.parent && name == o.name; }
        };
        struct key_hash
        {
            size_t operator()(const key_t & k) const noexcept
            {
                return std::hash<std::int64_t> {}(k.parent) ^ (std::hash<std::string> {}(k.name) << 1);
            }
        };

        std::vector<task_entry> merged;
        merged.reserve(kernels.size());
        std::unordered_map<key_t, std::int64_t, key_hash> canonical;
        std::vector<std::int64_t> remap(kernels.size(), -1);

        for (size_t i = 0; i < kernels.size(); ++i)
        {
            const auto & e = kernels[i];
            key_t k { e.parent_idx == -1 ? -1 : remap[e.parent_idx], e.name };
            auto it = canonical.find(k);
            if (it != canonical.end())
            {
                auto & target = merged[it->second];
                if (target.threading_task)
                    target.duration = std::max(target.duration, e.duration);
                else
                    target.duration += e.duration;
                target.count++;
                remap[i] = it->second;
            }
            else
            {
                task_entry ne  = e;
                ne.parent_idx  = k.parent;
                ne.idx         = static_cast<std::int64_t>(merged.size());
                remap[i]       = static_cast<std::int64_t>(merged.size());
                canonical[k]   = static_cast<std::int64_t>(merged.size());
                merged.push_back(std::move(ne));
            }
        }
        kernels = std::move(merged);
    }

    inline task & get_task() { return task_; }

public:
    // Per-thread stacks of currently-open regular and threading tasks, keyed by std::thread::id.
    // These are made public solely so the static start_/end_task methods (which run on many
    // threads but share this singleton) can index into them under global_mutex(). Guarded by
    // global_mutex().
    per_thread_stack_map regular_stacks_;
    per_thread_stack_map threading_stacks_;

    // Innermost open regular task across all threads; used as fallback parent for threading tasks
    // opened on worker threads that have no regular task on their own stack. Guarded by global_mutex().
    std::int64_t innermost_regular_task_idx_ = -1;

    // Once-per-process suppression sets for the logger/tracer THREADING messages. Live on the
    // singleton so their lifetime is bounded to the profiler's, not the process, and they can be
    // inspected. Guarded by global_mutex().
    std::unordered_set<std::string> unique_threading_task_names_;
    std::unordered_set<std::string> unique_threading_task_end_names_;

private:
    task task_;

    static std::mutex & global_mutex()
    {
        static std::mutex m;
        return m;
    }
};

inline profiler_task::~profiler_task()
{
    if (task_name_)
    {
#if (!defined(DAAL_NOTHROW_EXCEPTIONS))
        try
        {
#endif
            if (is_thread_)
                profiler::end_threading_task(task_name_, idx_);
            else
                profiler::end_task(task_name_, idx_);
#if (!defined(DAAL_NOTHROW_EXCEPTIONS))
        }
        catch (std::exception & e)
        {
            std::cerr << e.what() << std::endl;
        }
#endif
    }
}

} // namespace internal
} // namespace daal
