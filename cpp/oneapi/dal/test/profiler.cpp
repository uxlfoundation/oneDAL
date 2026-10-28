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

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "src/threading/threading.h"

namespace oneapi::dal::test {

namespace prof = daal::internal;
namespace te = dal::test::engine;

using rows_t = std::vector<prof::profiler_report_row>;

// The threading layer reports its chunks once the hooks are attached, which the profiler does on
// its first task when ONEDAL_VERBOSE is set. The tests attach them regardless of the environment.
static void start_recording() {
    static const bool attached = [] {
        _daal_threader_set_profiler_hooks(&prof::profiler_threading_hooks());
        return true;
    }();
    (void)attached;
    prof::profiler_reset();
}

/// Index of the row named `name` under the row `parent`, or -1.
static std::int64_t find_row(const rows_t& rows, const std::string& name, std::int64_t parent) {
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].name == name && rows[i].parent == parent) {
            return static_cast<std::int64_t>(i);
        }
    }
    return -1;
}

static void busy_wait(std::chrono::microseconds length) {
    const auto end = std::chrono::steady_clock::now() + length;
    while (std::chrono::steady_clock::now() < end) {
    }
}

TEST("profiler merges repeated siblings and splits self time from child time") {
    start_recording();
    {
        auto outer = prof::profiler::start_task("outer");
        busy_wait(std::chrono::microseconds(300));
        for (int i = 0; i < 3; ++i) {
            auto inner = prof::profiler::start_task("inner");
            busy_wait(std::chrono::microseconds(100));
        }
    }
    const rows_t rows = prof::profiler_report();
    REQUIRE(rows.size() == 2);
    const auto outer = find_row(rows, "outer", -1);
    const auto inner = find_row(rows, "inner", outer);
    REQUIRE(outer >= 0);
    REQUIRE(inner >= 0);
    REQUIRE(rows[outer].count == 1);
    REQUIRE(rows[inner].count == 3);
    REQUIRE(rows[inner].depth == 1);
    REQUIRE_FALSE(rows[outer].parallel);
    REQUIRE(rows[inner].time_ns >= 300'000);
    REQUIRE(rows[inner].max_ns >= 100'000);
    REQUIRE(rows[inner].max_ns <= rows[inner].time_ns);
    REQUIRE(rows[outer].time_ns >= rows[inner].time_ns + 300'000);
    REQUIRE(rows[outer].self_ns + rows[inner].time_ns == rows[outer].time_ns);
    REQUIRE(rows[inner].self_ns == rows[inner].time_ns);
}

TEST("profiler attaches the tasks of a parallel loop to the task that started it") {
    start_recording();
    constexpr std::int64_t n = 64;
    std::atomic<int> runs{ 0 };
    {
        auto outer = prof::profiler::start_task("outer");
        daal::threader_for(n, 1, [&](std::int64_t) {
            auto body = prof::profiler::start_task("body");
            busy_wait(std::chrono::microseconds(50));
            runs.fetch_add(1);
        });
    }
    REQUIRE(runs.load() == n);
    const rows_t rows = prof::profiler_report();
    REQUIRE(rows.size() == 2);
    const auto outer = find_row(rows, "outer", -1);
    const auto body = find_row(rows, "body", outer);
    REQUIRE(outer >= 0);
    REQUIRE(body >= 0);
    REQUIRE(rows[body].parallel);
    REQUIRE_FALSE(rows[outer].parallel);
    REQUIRE(rows[body].count == n);
    REQUIRE(rows[body].threads >= 1);
    REQUIRE(rows[body].threads <= static_cast<std::uint64_t>(daal::threader_get_threads_number()));
    // The busiest thread's time, so the row fits in the task that started the loop.
    REQUIRE(rows[body].time_ns <= rows[outer].time_ns);
    REQUIRE(rows[body].time_ns >= 50'000);
}

TEST("profiler keeps a nested parallel loop inside the task that started it") {
    start_recording();
    {
        auto outer = prof::profiler::start_task("outer");
        daal::threader_for(8, 1, [&](std::int64_t) {
            auto first = prof::profiler::start_task("first");
            daal::threader_for(8, 1, [&](std::int64_t) {
                auto second = prof::profiler::start_task("second");
                busy_wait(std::chrono::microseconds(20));
            });
        });
    }
    const rows_t rows = prof::profiler_report();
    REQUIRE(rows.size() == 3);
    const auto outer = find_row(rows, "outer", -1);
    const auto first = find_row(rows, "first", outer);
    const auto second = find_row(rows, "second", first);
    REQUIRE(outer >= 0);
    REQUIRE(first >= 0);
    REQUIRE(second >= 0);
    REQUIRE(rows[first].parallel);
    REQUIRE(rows[second].parallel);
    REQUIRE(rows[first].count == 8);
    REQUIRE(rows[second].count == 64);
    REQUIRE(rows[second].time_ns <= rows[first].time_ns);
    REQUIRE(rows[first].time_ns <= rows[outer].time_ns);
}

TEST("profiler takes a chunk run while the thread waited in another task out of that task") {
    start_recording();
    const auto& hooks = prof::profiler_threading_hooks();
    const void* region = nullptr;
    {
        auto started = prof::profiler::start_task("started");
        busy_wait(std::chrono::microseconds(500));
        region = hooks.capture();
    }
    {
        auto waiting = prof::profiler::start_task("waiting");
        // A chunk of the region started from `started`, as the threading layer hands it to a
        // thread that waits inside `waiting`.
        hooks.enter(region);
        {
            auto chunk = prof::profiler::start_task("chunk");
            busy_wait(std::chrono::microseconds(400));
        }
        hooks.leave();
        busy_wait(std::chrono::microseconds(100));
    }
    const rows_t rows = prof::profiler_report();
    REQUIRE(rows.size() == 3);
    const auto started = find_row(rows, "started", -1);
    const auto waiting = find_row(rows, "waiting", -1);
    REQUIRE(started >= 0);
    REQUIRE(waiting >= 0);
    const auto chunk = find_row(rows, "chunk", started);
    REQUIRE(chunk >= 0);
    REQUIRE(rows[chunk].parallel);
    REQUIRE(rows[chunk].time_ns >= 400'000);
    REQUIRE(rows[waiting].time_ns >= 100'000);
    REQUIRE(rows[waiting].time_ns < 400'000);
}

TEST("profiler keeps the tasks of independent calling threads apart") {
    start_recording();
    std::atomic<int> arrived{ 0 };
    auto call = [&](const char* outer_name, const char* inner_name) {
        auto outer = prof::profiler::start_task(outer_name);
        // Both outer tasks are open at the same time before either inner task starts.
        arrived.fetch_add(1);
        while (arrived.load() < 2) {
        }
        auto inner = prof::profiler::start_task(inner_name);
        busy_wait(std::chrono::microseconds(100));
    };
    std::thread a(call, "a_outer", "a_inner");
    std::thread b(call, "b_outer", "b_inner");
    a.join();
    b.join();
    const rows_t rows = prof::profiler_report();
    REQUIRE(rows.size() == 4);
    const auto a_outer = find_row(rows, "a_outer", -1);
    const auto b_outer = find_row(rows, "b_outer", -1);
    REQUIRE(a_outer >= 0);
    REQUIRE(b_outer >= 0);
    REQUIRE(find_row(rows, "a_inner", a_outer) >= 0);
    REQUIRE(find_row(rows, "b_inner", b_outer) >= 0);
    for (const auto& row : rows) {
        REQUIRE_FALSE(row.parallel);
        REQUIRE(row.count == 1);
    }
}

TEST("profiler closes tasks left open above the one being closed") {
    start_recording();
    {
        auto outer = prof::profiler::start_task("outer");
        auto inner = prof::profiler::start_task("inner");
        {
            // Ends `outer` while `inner` is still open, as moving a task out of its scope can.
            auto moved = std::move(outer);
        }
        auto after = prof::profiler::start_task("after");
    }
    const rows_t rows = prof::profiler_report();
    REQUIRE(rows.size() == 3);
    const auto outer = find_row(rows, "outer", -1);
    REQUIRE(outer >= 0);
    REQUIRE(find_row(rows, "inner", outer) >= 0);
    REQUIRE(find_row(rows, "after", -1) >= 0);
    for (const auto& row : rows) {
        REQUIRE(row.count == 1);
    }
}

TEST("profiler logs a call site once") {
    SKIP_IF(prof::is_service_debug_enabled());
    REQUIRE(prof::profiler_claim_log_site("profiler.cpp:once"));
    REQUIRE_FALSE(prof::profiler_claim_log_site("profiler.cpp:once"));
    REQUIRE(prof::profiler_claim_log_site("profiler.cpp:twice"));
}

TEST("profiler report can be requested through the C entry point and cleared") {
    start_recording();
    {
        auto task = prof::profiler::start_task("requested");
        busy_wait(std::chrono::microseconds(10));
    }
    const std::string path = "profiler_report_test.txt";
    std::remove(path.c_str());
    _onedal_profiler_report(path.c_str(), 1);
    std::ifstream file(path);
    REQUIRE(file.is_open());
    const std::string text((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
    file.close();
    std::remove(path.c_str());
    REQUIRE(text.find("Algorithm tree analyzer") != std::string::npos);
    REQUIRE(text.find("|-- requested time: ") != std::string::npos);
    REQUIRE(text.find("1 times in a sequential region") != std::string::npos);
    REQUIRE(prof::profiler_report().empty());
}

TEST("profiler prints parallel rows with their thread count") {
    start_recording();
    {
        auto outer = prof::profiler::start_task("outer");
        daal::threader_for(16, 1, [&](std::int64_t) {
            auto body = prof::profiler::start_task("body");
            busy_wait(std::chrono::microseconds(20));
        });
    }
    std::ostringstream out;
    prof::profiler_print_report(out);
    const std::string text = out.str();
    REQUIRE(text.find("|-- outer time: ") != std::string::npos);
    REQUIRE(text.find("|  |-- body time: ") != std::string::npos);
    REQUIRE(text.find("16 times in a parallel region on ") != std::string::npos);
    REQUIRE(text.find("DAAL KERNEL_PROFILER: kernels total time ") != std::string::npos);
}

#ifdef ONEDAL_DATA_PARALLEL
TEST("profiler task given a queue covers the device work submitted inside it") {
    start_recording();
    auto& queue = te::test_queue_provider::get_instance().get_global_queue();
    sycl::event event;
    {
        auto task = detail::queue_sync_guard::make(queue, "device_task", true);
        event = queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(1 << 16), [=](sycl::id<1> idx) {
                volatile float sink = static_cast<float>(idx[0]);
                for (int i = 0; i < 64; ++i) {
                    sink = sink * 1.0001f + 1.0f;
                }
            });
        });
    }
    REQUIRE(event.get_info<sycl::info::event::command_execution_status>() ==
            sycl::info::event_command_status::complete);
    const rows_t rows = prof::profiler_report();
    REQUIRE(rows.size() == 1);
    REQUIRE(rows[0].name == "device_task");
    REQUIRE(rows[0].count == 1);
    REQUIRE(rows[0].time_ns > 0);
}
#endif

} // namespace oneapi::dal::test
