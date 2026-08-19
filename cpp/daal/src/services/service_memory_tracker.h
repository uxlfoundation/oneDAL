/* file: service_memory_tracker.h */
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
//  Standalone memory allocation tracker with leak detection.
//  Controlled by ONEDAL_MEMORY_TRACKER env var, independent of profiler.
//
//  ONEDAL_MEMORY_TRACKER=0 (default) — off
//  ONEDAL_MEMORY_TRACKER=1           — summary + leak report (at program exit)
//  ONEDAL_MEMORY_TRACKER=2           — every allocation/free logged to stderr
//--
*/
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <unordered_map>

namespace daal
{
namespace internal
{

inline static constexpr int MEMORY_TRACKER_OFF     = 0;
inline static constexpr int MEMORY_TRACKER_SUMMARY = 1;
inline static constexpr int MEMORY_TRACKER_VERBOSE = 2;

inline std::atomic<int> memory_tracker_mode{ -1 };

inline static void init_memory_tracker_from_env()
{
    const char * env = std::getenv("ONEDAL_MEMORY_TRACKER");
    int newval       = MEMORY_TRACKER_OFF;
    if (env)
    {
        char * endptr  = nullptr;
        errno          = 0;
        long val       = std::strtol(env, &endptr, 10);
        bool parsed_ok = (errno == 0 && endptr != env && *endptr == '\0');
        if (parsed_ok && val >= 0 && val <= 2) newval = static_cast<int>(val);
    }
    memory_tracker_mode.store(newval, std::memory_order_relaxed);
}

inline static int get_memory_tracker_mode()
{
    int val = memory_tracker_mode.load(std::memory_order_relaxed);
    if (val == -1)
    {
        init_memory_tracker_from_env();
        val = memory_tracker_mode.load(std::memory_order_relaxed);
    }
    return val;
}

inline static bool is_memory_tracker_enabled()
{
    static const bool enabled = (get_memory_tracker_mode() >= MEMORY_TRACKER_SUMMARY);
    return enabled;
}

inline static bool is_memory_tracker_verbose()
{
    static const bool verbose = (get_memory_tracker_mode() >= MEMORY_TRACKER_VERBOSE);
    return verbose;
}

/// Formats a byte count into human-readable form (e.g. "1.50 MB").
inline static const char * format_bytes(std::size_t bytes, char * buf, std::size_t buf_size)
{
    if (bytes >= std::size_t(1) << 30)
        std::snprintf(buf, buf_size, "%.2f GB", static_cast<double>(bytes) / (std::size_t(1) << 30));
    else if (bytes >= std::size_t(1) << 20)
        std::snprintf(buf, buf_size, "%.2f MB", static_cast<double>(bytes) / (std::size_t(1) << 20));
    else if (bytes >= std::size_t(1) << 10)
        std::snprintf(buf, buf_size, "%.2f KB", static_cast<double>(bytes) / (std::size_t(1) << 10));
    else
        std::snprintf(buf, buf_size, "%zu B", bytes);
    return buf;
}

/// Per-allocation record stored in the pointer map.
struct alloc_record
{
    std::size_t bytes;
    const char * kind; // string literal, never freed
};

/// Singleton that tracks allocation statistics and detects leaks.
class memory_tracker
{
public:
    static memory_tracker & instance()
    {
        static memory_tracker inst;
        return inst;
    }

    void record_alloc(const char * kind, void * ptr, std::size_t bytes)
    {
        if (!ptr) return;

        total_allocated_.fetch_add(bytes, std::memory_order_relaxed);
        alloc_count_.fetch_add(1, std::memory_order_relaxed);

        {
            std::lock_guard<std::mutex> lock(map_mutex_);
            live_allocs_[ptr] = { bytes, kind };

            std::size_t cur = current_live_.fetch_add(bytes, std::memory_order_relaxed) + bytes;
            // Update peak (lock-free CAS loop)
            std::size_t prev_peak = peak_live_.load(std::memory_order_relaxed);
            while (cur > prev_peak && !peak_live_.compare_exchange_weak(prev_peak, cur, std::memory_order_relaxed))
            {
            }
        }

        if (is_memory_tracker_verbose())
        {
            char buf[32];
            std::lock_guard<std::mutex> lock(io_mutex_);
            std::cerr << "[MEMORY] alloc " << kind << " " << format_bytes(bytes, buf, sizeof(buf)) << " (" << bytes << " bytes)  ptr="
                      << ptr << '\n';
        }
    }

    void record_free(const char * kind, void * ptr)
    {
        if (!ptr) return;

        std::size_t bytes = 0;
        {
            std::lock_guard<std::mutex> lock(map_mutex_);
            auto it = live_allocs_.find(ptr);
            if (it != live_allocs_.end())
            {
                bytes = it->second.bytes;
                live_allocs_.erase(it);
            }
        }

        total_freed_.fetch_add(bytes, std::memory_order_relaxed);
        free_count_.fetch_add(1, std::memory_order_relaxed);
        if (bytes > 0)
        {
            current_live_.fetch_sub(bytes, std::memory_order_relaxed);
        }

        if (is_memory_tracker_verbose())
        {
            char buf[32];
            std::lock_guard<std::mutex> lock(io_mutex_);
            if (bytes > 0)
            {
                std::cerr << "[MEMORY] free  " << kind << " " << format_bytes(bytes, buf, sizeof(buf)) << " (" << bytes
                          << " bytes)  ptr=" << ptr << '\n';
            }
            else
            {
                std::cerr << "[MEMORY] free  " << kind << " (untracked ptr)  ptr=" << ptr << '\n';
            }
        }
    }

    ~memory_tracker()
    {
        if (!is_memory_tracker_enabled()) return;

        char buf1[32], buf2[32], buf3[32], buf4[32];
        std::cerr << "\n=== oneDAL Memory Tracker Summary ===\n";
        std::cerr << "Total allocated:  " << format_bytes(total_allocated_.load(), buf1, sizeof(buf1)) << " (" << total_allocated_.load()
                  << " bytes)\n";
        std::cerr << "Total freed:      " << format_bytes(total_freed_.load(), buf2, sizeof(buf2)) << " (" << total_freed_.load() << " bytes)\n";
        std::cerr << "Peak live:        " << format_bytes(peak_live_.load(), buf3, sizeof(buf3)) << " (" << peak_live_.load() << " bytes)\n";
        std::cerr << "Alloc calls:      " << alloc_count_.load() << '\n';
        std::cerr << "Free calls:       " << free_count_.load() << '\n';

        // Leak report
        std::lock_guard<std::mutex> lock(map_mutex_);
        if (live_allocs_.empty())
        {
            std::cerr << "Leaked:           0 B (no leaks detected)\n";
        }
        else
        {
            std::size_t leaked_bytes = 0;
            for (auto & p : live_allocs_)
                leaked_bytes += p.second.bytes;
            std::cerr << "Leaked:           " << format_bytes(leaked_bytes, buf4, sizeof(buf4)) << " (" << leaked_bytes << " bytes) in "
                      << live_allocs_.size() << " allocation(s)\n";

            // Print up to 20 leaked allocations
            std::size_t shown = 0;
            for (auto & p : live_allocs_)
            {
                if (shown >= 20)
                {
                    std::cerr << "  ... and " << (live_allocs_.size() - shown) << " more\n";
                    break;
                }
                char b[32];
                std::cerr << "  LEAK: ptr=" << p.first << "  " << format_bytes(p.second.bytes, b, sizeof(b)) << " (" << p.second.bytes
                          << " bytes)  kind=" << p.second.kind << '\n';
                ++shown;
            }
        }
        std::cerr << "=====================================\n";
    }

private:
    memory_tracker() = default;

    std::atomic<std::size_t> total_allocated_{ 0 };
    std::atomic<std::size_t> total_freed_{ 0 };
    std::atomic<std::size_t> current_live_{ 0 };
    std::atomic<std::size_t> peak_live_{ 0 };
    std::atomic<std::size_t> alloc_count_{ 0 };
    std::atomic<std::size_t> free_count_{ 0 };
    std::unordered_map<void *, alloc_record> live_allocs_;
    std::mutex map_mutex_;
    std::mutex io_mutex_;
};

} // namespace internal
} // namespace daal

// Convenience macros — no-ops when tracker is disabled (branch predicted away).
// ALLOC takes the pointer returned by the allocator so we can track it.
// FREE takes the pointer being freed so we can match it to the original alloc.
#define ONEDAL_MEMORY_TRACKER_ALLOC(kind, ptr, bytes)                                  \
    do                                                                                 \
    {                                                                                  \
        if (daal::internal::is_memory_tracker_enabled())                               \
        {                                                                              \
            daal::internal::memory_tracker::instance().record_alloc(kind, ptr, bytes); \
        }                                                                              \
    } while (0)

#define ONEDAL_MEMORY_TRACKER_FREE(kind, ptr)                                        \
    do                                                                               \
    {                                                                                \
        if (daal::internal::is_memory_tracker_enabled())                             \
        {                                                                            \
            daal::internal::memory_tracker::instance().record_free(kind, ptr);       \
        }                                                                            \
    } while (0)
