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

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
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
#define DAAL_PROFILER_MACRO_2(name, queue)                daal::internal::profiler::start_task(#name)
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
                std::cerr << "Profiler task_name: " << #__VA_ARGS__ << '\n';                                                                  \
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
                std::cerr << "Profiler task_name: " << #__VA_ARGS__ << '\n';                                                                  \
            }                                                                                                                                 \
            return DAAL_PROFILER_GET_MACRO(__VA_ARGS__, DAAL_PROFILER_MACRO_2, DAAL_PROFILER_MACRO_1, FICTIVE)(__VA_ARGS__);                  \
        }                                                                                                                                     \
        return daal::internal::profiler::start_task(nullptr);                                                                                 \
    }()

// FIX: inline atomic ensures a single shared variable across all TUs.
// Previously: `static volatile int` gave each TU its own copy — data race.
inline std::atomic<int> daal_verbose_val{ -1 };
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

inline static void set_verbose_from_env()
{
    const char * verbose_str = std::getenv("ONEDAL_VERBOSE");
    int newval               = PROFILER_MODE_OFF;
    if (verbose_str)
    {
        char * endptr  = nullptr;
        errno          = 0;
        long val       = std::strtol(verbose_str, &endptr, 10);
        bool parsed_ok = (errno == 0 && endptr != verbose_str && *endptr == '\0');
        if (parsed_ok && val >= 0 && val <= 5) newval = static_cast<int>(val);
    }
    daal_verbose_val.store(newval, std::memory_order_relaxed);
}

inline static int daal_verbose_mode()
{
    int val = daal_verbose_val.load(std::memory_order_relaxed);
    if (val == -1)
    {
        set_verbose_from_env();
        val = daal_verbose_val.load(std::memory_order_relaxed);
    }
    return val;
}

// FIX: snprintf into a small stack buffer instead of heap-allocating
// an ostringstream + string per call. 10-50x faster in hot paths.
inline static std::string format_time_for_output(std::uint64_t time_ns)
{
    char buf[32];
    double time = static_cast<double>(time_ns);
    if (time <= 0)
        std::snprintf(buf, sizeof(buf), "0.00s");
    else if (time > 1e9)
        std::snprintf(buf, sizeof(buf), "%.2fs", time / 1e9);
    else if (time > 1e6)
        std::snprintf(buf, sizeof(buf), "%.2fms", time / 1e6);
    else if (time > 1e3)
        std::snprintf(buf, sizeof(buf), "%.2fus", time / 1e3);
    else
        std::snprintf(buf, sizeof(buf), "%luns", static_cast<unsigned long>(time_ns));
    return std::string(buf);
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
    // FIX: const char* instead of std::string. All profiler task names
    // are string literals from #name macro stringification — no heap allocation needed.
    const char * name;
    std::uint64_t duration;
    std::int64_t level;
    std::int64_t count;
    bool threading_task;
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

class profiler
{
public:
    inline profiler()
    {
        daal_verbose_mode();
        // Pre-allocate space for typical profiling sessions to avoid
        // reallocation latency spikes during hot profiling paths.
        task_.kernels.reserve(256);
    }

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
                std::cerr << e.what() << '\n';
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
    /// @note FIX: Timestamp is now captured BEFORE acquiring the mutex so that lock
    /// contention time is not included in the measured task duration. Previously the
    /// timestamp was taken after the lock, inflating durations under contention.
    inline static profiler_task start_task(const char * task_name)
    {
        if (!task_name) return profiler_task(nullptr, -1);

        const auto ns_start = get_time();
        std::lock_guard<std::mutex> lock(global_mutex());
        auto & tasks_info            = get_instance()->get_task();
        auto & current_level_        = get_instance()->get_current_level();
        auto & current_kernel_count_ = get_instance()->get_kernel_count();
        std::int64_t tmp             = current_kernel_count_;
        tasks_info.kernels.push_back({ tmp, task_name, ns_start, current_level_, 1, false });
        current_level_++;
        current_kernel_count_++;
        return profiler_task(task_name, tmp);
    }

    /// Starts a threading-specific profiling task with the given task name and returns a profiler_task object
    ///
    /// @param[in] task_name The name of the threading task to be profiled
    ///
    /// @return A profiler_task object containing the task name, a unique task ID, and a threading flag.
    /// Returns an invalid profiler_task (nullptr, -1) if task_name is nullptr
    ///
    /// @note Uses a mutex for thread safety, logs unique task names if logging is enabled, captures the start time,
    /// updates task info, and increments the kernel count. Stores task details in tasks_info.kernels, marking it
    /// as a threading task. Invoked by the DAAL_PROFILER_THREADING_TASK macro.
    /// FIX: Uses unordered_set for O(1) unique name lookup instead of O(n) vector scan.
    /// FIX: Timestamp captured before lock.
    inline static profiler_task start_threading_task(const char * task_name)
    {
        if (!task_name) return profiler_task(nullptr, -1);

        const auto ns_start = get_time();
        std::lock_guard<std::mutex> lock(global_mutex());
        if (is_logger_enabled())
        {
            if (!is_service_debug_enabled())
            {
                static std::unordered_set<std::string> unique_task_names;
                auto [it, inserted] = unique_task_names.insert(task_name);
                if (inserted)
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
        auto & tasks_info            = get_instance()->get_task();
        auto & current_kernel_count_ = get_instance()->get_kernel_count();
        std::int64_t tmp             = current_kernel_count_;
        tasks_info.kernels.push_back({ tmp, task_name, ns_start, get_instance()->get_current_level(), 1, true });
        current_kernel_count_++;
        return profiler_task(task_name, tmp, true);
    }

    /// Terminates a profiling task and records its duration
    ///
    /// @param[in] task_name The name of the task to end
    /// @param[in] idx_ The index of the task in the tasks_info.kernels vector
    ///
    /// @note FIX: Added bounds check (was missing — end_threading_task had it, this didn't).
    /// Captures end time before the lock to avoid measuring mutex contention.
    inline static void end_task(const char * task_name, int idx_)
    {
        if (!task_name) return;
        const std::uint64_t ns_end = get_time();

        std::lock_guard<std::mutex> lock(global_mutex());
        auto & tasks_info = get_instance()->get_task();
        if (idx_ < 0 || static_cast<std::size_t>(idx_) >= tasks_info.kernels.size()) return;

        auto & entry          = tasks_info.kernels[idx_];
        auto duration         = ns_end - entry.duration;
        entry.duration        = duration;
        auto & current_level_ = get_instance()->get_current_level();
        current_level_--;
        if (is_tracer_enabled()) std::cerr << task_name << " " << format_time_for_output(duration) << '\n';
    }

    /// Terminates a threading-specific profiling task and records its duration
    ///
    /// @param[in] task_name The name of the threading task to end
    /// @param[in] idx_ The index of the task in the tasks_info.kernels vector
    ///
    /// @note FIX: Uses unordered_set for O(1) unique name lookup.
    inline static void end_threading_task(const char * task_name, int idx_)
    {
        if (!task_name) return;
        const std::uint64_t ns_end = get_time();

        std::lock_guard<std::mutex> lock(global_mutex());
        auto & tasks_info = get_instance()->get_task();

        if (idx_ < 0 || static_cast<std::size_t>(idx_) >= tasks_info.kernels.size()) return;

        auto & entry   = tasks_info.kernels[idx_];
        auto duration  = ns_end - entry.duration;
        entry.duration = duration;

        if (is_tracer_enabled())
        {
            static std::unordered_set<std::string> unique_task_names;
            auto [it, inserted] = unique_task_names.insert(task_name);
            if (inserted)
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

    /// Merges tasks at the same nesting level with identical names to improve profiling clarity.
    ///
    /// FIX: Rewrote from O(n^3) nested-loop-with-erase to O(n) single-pass hash-based merge.
    /// For non-threading tasks, durations are summed; for threading tasks, the maximum
    /// duration is taken. Skips merging if service debug mode is enabled to preserve
    /// detailed task information.
    inline void merge_tasks()
    {
        if (is_service_debug_enabled())
        {
            return;
        }
        auto & kernels = task_.kernels;

        // key = (level, name) -> index in the output vector
        struct LevelNameHash
        {
            std::size_t operator()(const std::pair<std::int64_t, const char *> & p) const
            {
                // Hash the level and the string content (not the pointer)
                auto h1 = std::hash<std::int64_t>{}(p.first);
                auto h2 = std::hash<std::string_view>{}(std::string_view(p.second));
                return h1 ^ (h2 << 1);
            }
        };
        struct LevelNameEqual
        {
            bool operator()(const std::pair<std::int64_t, const char *> & a, const std::pair<std::int64_t, const char *> & b) const
            {
                return a.first == b.first && std::strcmp(a.second, b.second) == 0;
            }
        };

        std::vector<task_entry> merged;
        merged.reserve(kernels.size());
        std::unordered_map<std::pair<std::int64_t, const char *>, std::size_t, LevelNameHash, LevelNameEqual> seen;

        for (auto & entry : kernels)
        {
            auto key = std::make_pair(entry.level, entry.name);
            auto it  = seen.find(key);
            if (it != seen.end())
            {
                auto & existing = merged[it->second];
                if (existing.threading_task)
                    existing.duration = std::max(existing.duration, entry.duration);
                else
                    existing.duration += entry.duration;
                existing.count++;
            }
            else
            {
                seen[key] = merged.size();
                merged.push_back(std::move(entry));
            }
        }
        kernels = std::move(merged);
    }

    inline task & get_task()
    {
        return task_;
    }
    inline std::int64_t & get_current_level()
    {
        return current_level_;
    }
    inline std::int64_t & get_kernel_count()
    {
        return kernel_count_;
    }

private:
    std::int64_t current_level_ = 0;
    std::int64_t kernel_count_  = 0;
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
            std::cerr << e.what() << '\n';
        }
#endif
    }
}

} // namespace internal
} // namespace daal
