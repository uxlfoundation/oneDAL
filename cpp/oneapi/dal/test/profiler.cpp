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

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <sstream>
#include <thread>

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "src/threading/threading.h"

namespace oneapi::dal::test {

namespace prof = daal::internal;

// The profiler reads its mode once, at the first call. Running the analyzer for the whole binary
// attaches the threading layer's hooks, as in a profiled program; an explicit setting wins.
[[maybe_unused]] static const bool analyzer_mode = [] {
#ifdef _WIN32
    if (!std::getenv("ONEDAL_VERBOSE")) {
        _putenv_s("ONEDAL_VERBOSE", "3");
    }
#else
    setenv("ONEDAL_VERBOSE", "3", 0);
#endif
    return true;
}();

using rows_t = std::vector<prof::profiler_report_row>;

/// Index of the row named `name` under the row `parent`, or -1.
static std::int64_t find_row(const rows_t& rows, const std::string& name, std::int64_t parent) {
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].name == name && rows[i].parent == parent) {
            return static_cast<std::int64_t>(i);
        }
    }
    return -1;
}

/// Number of rows that are tasks, not the rows the report adds for waits on parallel regions.
static std::size_t task_count(const rows_t& rows) {
    return static_cast<std::size_t>(std::count_if(rows.begin(), rows.end(), [](const auto& row) {
        return row.name.front() != '(';
    }));
}

static void busy_wait(std::chrono::microseconds length) {
    const auto end = std::chrono::steady_clock::now() + length;
    while (std::chrono::steady_clock::now() < end) {
    }
}

// Runs `body` on a new thread, inside a chunk of `region`, and keeps the task it opens alive until
// `release()` is called, so several workers can hold open tasks at the same time.
class held_worker {
public:
    template <typename Body>
    held_worker(const daal::profiler_region& region, Body body)
            : thread_([this, region, body]() {
                  const auto& hooks = prof::profiler_threading_hooks();
                  hooks.enter(region);
                  {
                      auto task = body();
                      {
                          std::lock_guard<std::mutex> lock(m_);
                          opened_ = true;
                      }
                      cv_.notify_all();
                      std::unique_lock<std::mutex> lock(m_);
                      cv_.wait(lock, [this] {
                          return released_;
                      });
                  }
                  hooks.leave();
              }) {
        std::unique_lock<std::mutex> lock(m_);
        cv_.wait(lock, [this] {
            return opened_;
        });
    }

    void release() {
        {
            std::lock_guard<std::mutex> lock(m_);
            released_ = true;
        }
        cv_.notify_all();
        thread_.join();
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    bool opened_ = false;
    bool released_ = false;
    std::thread thread_;
};

// Calls the threading layer's hooks around `body`, as it does around a chunk of a parallel region.
template <typename Body>
static void run_chunk(const daal::profiler_region& region, Body body) {
    const auto& hooks = prof::profiler_threading_hooks();
    hooks.enter(region);
    body();
    hooks.leave();
}

TEST("profiler merges repeated siblings and splits self time from child time") {
    prof::profiler_reset();
    {
        auto outer = prof::profiler::start_task("outer");
        for (int i = 0; i < 3; ++i) {
            auto inner = prof::profiler::start_task("inner");
            busy_wait(std::chrono::microseconds(200));
        }
        busy_wait(std::chrono::microseconds(200));
    }
    const auto rows = prof::profiler_report();

    const auto outer = find_row(rows, "outer", -1);
    REQUIRE(outer >= 0);
    const auto inner = find_row(rows, "inner", outer);
    REQUIRE(inner >= 0);
    REQUIRE(task_count(rows) == 2);

    REQUIRE(rows[outer].count == 1);
    REQUIRE(rows[inner].count == 3);
    REQUIRE(rows[inner].depth == 1);
    REQUIRE(!rows[outer].parallel);
    REQUIRE(rows[inner].min_ns <= rows[inner].max_ns);
    REQUIRE(rows[inner].time_ns >= 3 * 200000);
    REQUIRE(rows[outer].time_ns >= rows[inner].time_ns + 200000);
    // Each figure is converted from clock ticks on its own, so they agree to a nanosecond or two.
    const auto expected_self = std::int64_t(rows[outer].time_ns - rows[inner].time_ns);
    REQUIRE(std::llabs(std::int64_t(rows[outer].self_ns) - expected_self) <= 2);
    REQUIRE(rows[inner].self_ns == rows[inner].time_ns);
}

TEST("profiler attaches concurrent worker tasks to the task that started the region") {
    prof::profiler_reset();
    const auto& hooks = prof::profiler_threading_hooks();
    {
        auto outer = prof::profiler::start_task("outer");
        const daal::profiler_region region = hooks.capture(true);

        // Worker threads have nothing open on their own stacks, like TBB workers joining a region.
        held_worker first(region, []() {
            return prof::profiler::start_threading_task("worker");
        });
        held_worker second(region, []() {
            return prof::profiler::start_threading_task("worker");
        });
        first.release();
        second.release();
        hooks.finish(region);

        // A task opened after the region still nests under `outer`, not under a worker task.
        auto after = prof::profiler::start_task("after");
    }
    const auto rows = prof::profiler_report();

    const auto outer = find_row(rows, "outer", -1);
    REQUIRE(outer >= 0);
    const auto worker = find_row(rows, "worker", outer);
    REQUIRE(worker >= 0);
    REQUIRE(find_row(rows, "after", outer) >= 0);
    REQUIRE(task_count(rows) == 3);

    REQUIRE(rows[worker].parallel);
    REQUIRE(rows[worker].count == 2);
    REQUIRE(rows[worker].threads == 2);
    // Wall time is the busier thread, CPU time is both.
    REQUIRE(rows[worker].cpu_ns >= rows[worker].time_ns);
    REQUIRE(rows[worker].time_ns <= rows[outer].time_ns);
}

TEST("profiler nests a worker's regular task under its own threading task") {
    prof::profiler_reset();
    const auto& hooks = prof::profiler_threading_hooks();
    {
        auto outer = prof::profiler::start_task("outer");
        const daal::profiler_region region = hooks.capture(true);
        std::thread worker([&]() {
            run_chunk(region, []() {
                auto threading = prof::profiler::start_threading_task("body");
                auto inner = prof::profiler::start_task("inner");
            });
        });
        worker.join();
        hooks.finish(region);
    }
    const auto rows = prof::profiler_report();

    const auto outer = find_row(rows, "outer", -1);
    const auto body = find_row(rows, "body", outer);
    REQUIRE(body >= 0);
    const auto inner = find_row(rows, "inner", body);
    REQUIRE(inner >= 0);
    REQUIRE(rows[inner].parallel);
}

TEST("profiler attaches each chunk to the task its region was started from") {
    prof::profiler_reset();
    const auto& hooks = prof::profiler_threading_hooks();
    {
        auto outer = prof::profiler::start_task("outer");
        const daal::profiler_region region = hooks.capture(true);

        // The thread that started the region runs a chunk itself, and so does a worker.
        run_chunk(region, []() {
            auto body = prof::profiler::start_task("body");
        });
        std::thread worker([&]() {
            run_chunk(region, []() {
                auto body = prof::profiler::start_task("body");
            });
        });
        worker.join();
        hooks.finish(region);
    }
    const auto rows = prof::profiler_report();

    const auto outer = find_row(rows, "outer", -1);
    const auto body = find_row(rows, "body", outer);
    REQUIRE(body >= 0);
    REQUIRE(task_count(rows) == 2);
    REQUIRE(rows[body].parallel);
    REQUIRE(rows[body].count == 2);
    REQUIRE(rows[body].threads == 2);
}

TEST("profiler keeps concurrent chunks of a region started outside any task apart") {
    prof::profiler_reset();
    const auto& hooks = prof::profiler_threading_hooks();
    const daal::profiler_region region = hooks.capture(true);

    // Both chunks open the same tasks at the same time, as in a parallel loop over independent fits.
    std::mutex m;
    std::condition_variable cv;
    int opened = 0;
    auto chunk_body = [&]() {
        run_chunk(region, [&]() {
            auto train = prof::profiler::start_task("train");
            auto inner = prof::profiler::start_task("inner");
            std::unique_lock<std::mutex> lock(m);
            ++opened;
            cv.notify_all();
            cv.wait(lock, [&] {
                return opened == 2;
            });
        });
    };
    std::thread first(chunk_body);
    std::thread second(chunk_body);
    first.join();
    second.join();
    hooks.finish(region);
    const auto rows = prof::profiler_report();

    const auto train = find_row(rows, "train", -1);
    REQUIRE(train >= 0);
    REQUIRE(find_row(rows, "inner", train) >= 0);
    REQUIRE(task_count(rows) == 2);
    REQUIRE(rows[train].count == 2);
    REQUIRE(rows[train].threads == 2);
}

TEST("profiler does not add up nested regions of tasks that ran at the same time") {
    prof::profiler_reset();
    const auto& hooks = prof::profiler_threading_hooks();
    const daal::profiler_region outer_region = hooks.capture(true);

    // Two fits run as chunks of one region, and each waits for a nested region of its own.
    auto fit = [&]() {
        run_chunk(outer_region, [&]() {
            auto train = prof::profiler::start_task("train");
            const daal::profiler_region nested = hooks.capture(true);
            std::thread helper([&]() {
                run_chunk(nested, []() {
                    auto rows = prof::profiler::start_task("rows");
                    busy_wait(std::chrono::microseconds(2000));
                });
            });
            helper.join();
            hooks.finish(nested);
        });
    };
    std::thread first(fit);
    std::thread second(fit);
    first.join();
    second.join();
    hooks.finish(outer_region);
    const auto rows = prof::profiler_report();

    const auto train = find_row(rows, "train", -1);
    REQUIRE(train >= 0);
    const auto nested = find_row(rows, "rows", train);
    REQUIRE(nested >= 0);
    REQUIRE(rows[train].threads == 2);
    REQUIRE(rows[nested].count == 2);
    // The two nested regions overlap in time, so together they take as long as one.
    REQUIRE(rows[nested].time_ns <= rows[train].time_ns);
    REQUIRE(rows[nested].cpu_ns >= 2 * 2000000);
    // Waiting for the nested region is not the fit's own time; it shows as a row of its own.
    REQUIRE(rows[train].self_ns < 1000000);
    const auto waiting = find_row(rows, "(in parallel regions, outside tasks)", train);
    REQUIRE(waiting >= 0);
    REQUIRE(rows[waiting].count == 2);
    REQUIRE(rows[waiting].time_ns <= rows[train].time_ns);
    // The rows under a task never add up to more than the task.
    REQUIRE(rows[train].self_ns + rows[nested].time_ns + rows[waiting].time_ns <=
            rows[train].time_ns + 2);
}

TEST("profiler takes a chunk run while the thread waited elsewhere out of the waiting tasks") {
    prof::profiler_reset();
    const auto& hooks = prof::profiler_threading_hooks();
    {
        auto outer = prof::profiler::start_task("outer");
        const daal::profiler_region region = hooks.capture(true);
        std::thread worker([&]() {
            run_chunk(region, [&]() {
                auto chunk = prof::profiler::start_task("chunk");
                auto wait = prof::profiler::start_task("wait");
                // While waiting, the worker is given another chunk of the same region.
                run_chunk(region, []() {
                    auto stolen = prof::profiler::start_task("chunk");
                    busy_wait(std::chrono::microseconds(20000));
                });
                busy_wait(std::chrono::microseconds(100));
            });
        });
        worker.join();
        hooks.finish(region);
    }
    const auto rows = prof::profiler_report();

    const auto outer = find_row(rows, "outer", -1);
    const auto chunk = find_row(rows, "chunk", outer);
    REQUIRE(chunk >= 0);
    const auto wait = find_row(rows, "wait", chunk);
    REQUIRE(wait >= 0);
    REQUIRE(find_row(rows, "chunk", wait) == -1);
    REQUIRE(task_count(rows) == 3);

    REQUIRE(rows[chunk].count == 2);
    REQUIRE(rows[wait].time_ns >= 100000);
    // Well below the stolen chunk, with room for the thread being preempted on a loaded machine.
    REQUIRE(rows[wait].time_ns < 10000000);
    REQUIRE(rows[chunk].time_ns >= 20000000 + rows[wait].time_ns);
}

TEST(
    "profiler keeps a nested region inside the task that started it when that thread is taken away") {
    prof::profiler_reset();
    const auto& hooks = prof::profiler_threading_hooks();
    {
        auto outer = prof::profiler::start_task("outer");
        const daal::profiler_region region = hooks.capture(true);
        run_chunk(region, [&]() {
            auto update = prof::profiler::start_task("update");
            const daal::profiler_region nested = hooks.capture(true);
            std::thread worker([&]() {
                run_chunk(nested, []() {
                    auto inner = prof::profiler::start_task("inner");
                    busy_wait(std::chrono::microseconds(3000));
                });
            });
            // While it waits, this thread runs another chunk of the outer region, which is not part
            // of `update`; the worker's chunk of the nested region runs meanwhile.
            run_chunk(region, []() {
                auto other = prof::profiler::start_task("other");
                busy_wait(std::chrono::microseconds(2500));
            });
            worker.join();
            hooks.finish(nested);
        });
        hooks.finish(region);
    }
    const auto rows = prof::profiler_report();
    const auto outer = find_row(rows, "outer", -1);
    const auto update = find_row(rows, "update", outer);
    REQUIRE(update >= 0);
    const auto inner = find_row(rows, "inner", update);
    REQUIRE(inner >= 0);
    REQUIRE(rows[inner].time_ns <= rows[update].time_ns);
    REQUIRE(rows[update].time_ns <= rows[outer].time_ns);
}

TEST("profiler takes the busiest thread of each run of a parallel region") {
    prof::profiler_reset();
    const auto& hooks = prof::profiler_threading_hooks();
    for (int run = 0; run < 2; ++run) {
        auto outer = prof::profiler::start_task("outer");
        const daal::profiler_region region = hooks.capture(true);
        std::thread slow([&]() {
            run_chunk(region, []() {
                auto chunk = prof::profiler::start_threading_task("chunk");
                busy_wait(std::chrono::microseconds(3000));
            });
        });
        slow.join();
        std::thread fast([&]() {
            run_chunk(region, []() {
                auto chunk = prof::profiler::start_threading_task("chunk");
                busy_wait(std::chrono::microseconds(1000));
            });
        });
        fast.join();
        hooks.finish(region);
    }
    const auto rows = prof::profiler_report();

    const auto outer = find_row(rows, "outer", -1);
    const auto chunk = find_row(rows, "chunk", outer);
    REQUIRE(chunk >= 0);
    REQUIRE(rows[chunk].count == 4);
    REQUIRE(rows[chunk].threads == 2);
    // Each run lasts as long as its slow thread, although every thread ran in only one of the runs.
    REQUIRE(rows[chunk].time_ns >= 2 * 3000000);
    REQUIRE(rows[chunk].time_ns < rows[chunk].cpu_ns);
    REQUIRE(rows[chunk].mean_thread_ns >= 2 * 2000000);
    REQUIRE(rows[chunk].mean_thread_ns < rows[chunk].time_ns);
}

TEST("profiler charges memory to the open task and counts it in every enclosing task") {
    prof::profiler_reset();
    char host_a = 0, host_b = 0, device = 0;
    {
        auto outer = prof::profiler::start_task("outer");
        prof::profiler_record_alloc(&host_a, 1000, prof::profiler_memory_kind::host);
        {
            auto inner = prof::profiler::start_task("inner");
            prof::profiler_record_alloc(&host_b, 500, prof::profiler_memory_kind::host);
            prof::profiler_record_alloc(&device, 64, prof::profiler_memory_kind::device);
            prof::profiler_record_free(&host_b);
            prof::profiler_record_transfer(4096, prof::profiler_transfer_kind::host_to_device);
        }
        prof::profiler_record_free(&host_a);
        prof::profiler_record_free(&device);
    }
    const auto rows = prof::profiler_report();

    const auto outer = find_row(rows, "outer", -1);
    const auto inner = find_row(rows, "inner", outer);
    REQUIRE(inner >= 0);
    const int host = static_cast<int>(prof::profiler_memory_kind::host);
    const int dev = static_cast<int>(prof::profiler_memory_kind::device);
    const int to_device = static_cast<int>(prof::profiler_transfer_kind::host_to_device);

    REQUIRE(rows[inner].alloc_bytes[host] == 500);
    REQUIRE(rows[inner].free_bytes[host] == 500);
    REQUIRE(rows[inner].peak_bytes[host] == 500);
    REQUIRE(rows[inner].alloc_bytes[dev] == 64);
    REQUIRE(rows[inner].free_bytes[dev] == 0);
    REQUIRE(rows[inner].transfer_bytes[to_device] == 4096);
    REQUIRE(rows[inner].transfer_count[to_device] == 1);

    REQUIRE(rows[outer].alloc_bytes[host] == 1500);
    REQUIRE(rows[outer].alloc_count[host] == 2);
    REQUIRE(rows[outer].free_bytes[host] == 1500);
    REQUIRE(rows[outer].peak_bytes[host] == 1500);
    REQUIRE(rows[outer].free_bytes[dev] == 64);
    REQUIRE(rows[outer].transfer_bytes[to_device] == 4096);
}

TEST(
    "profiler charges a worker's allocations outside any task to the task that started the region") {
    prof::profiler_reset();
    const auto& hooks = prof::profiler_threading_hooks();
    char small_block = 0, large_block = 0;
    const std::size_t large = std::size_t(1) << 20;
    {
        auto outer = prof::profiler::start_task("outer");
        const daal::profiler_region region = hooks.capture(true);
        std::thread worker([&]() {
            run_chunk(region, [&]() {
                prof::profiler_record_alloc(&small_block, 256, prof::profiler_memory_kind::host);
                prof::profiler_record_alloc(&large_block, large, prof::profiler_memory_kind::host);
                prof::profiler_record_free(&large_block);
                prof::profiler_record_free(&small_block);
            });
        });
        worker.join();
        hooks.finish(region);
    }
    const auto rows = prof::profiler_report();

    const auto outer = find_row(rows, "outer", -1);
    REQUIRE(outer >= 0);
    REQUIRE(rows[outer].alloc_bytes[0] == large + 256);
    REQUIRE(rows[outer].free_bytes[0] == large + 256);
    // A worker's small blocks reach the process-wide count in batches; a large one at once, so it
    // shows in the peak of the task that started the region.
    REQUIRE(rows[outer].peak_bytes[0] >= large);
    REQUIRE(rows[outer].peak_bytes[0] <= large + 256);
}

TEST("profiler reports memory still allocated in the task that allocated it") {
    prof::profiler_reset();
    const auto& hooks = prof::profiler_threading_hooks();
    char kept = 0, freed = 0, in_chunk = 0;
    {
        auto outer = prof::profiler::start_task("outer");
        {
            auto inner = prof::profiler::start_task("inner");
            prof::profiler_record_alloc(&kept, 1000, prof::profiler_memory_kind::host);
            prof::profiler_record_alloc(&freed, 64, prof::profiler_memory_kind::device);
            prof::profiler_record_free(&freed);
        }
        // A worker's allocation outside any task belongs to the task that started the region.
        const daal::profiler_region region = hooks.capture(true);
        std::thread worker([&]() {
            run_chunk(region, [&]() {
                prof::profiler_record_alloc(&in_chunk, 300, prof::profiler_memory_kind::device);
            });
        });
        worker.join();
        hooks.finish(region);
    }
    auto rows = prof::profiler_report();
    const auto outer = find_row(rows, "outer", -1);
    const auto inner = find_row(rows, "inner", outer);
    REQUIRE(inner >= 0);
    REQUIRE(rows[inner].live_bytes[0] == 1000);
    REQUIRE(rows[inner].live_count[0] == 1);
    REQUIRE(rows[inner].live_bytes[1] == 0);
    REQUIRE(rows[outer].live_bytes[0] == 1000);
    REQUIRE(rows[outer].live_bytes[1] == 300);

    prof::profiler_record_free(&kept);
    prof::profiler_record_free(&in_chunk);
    rows = prof::profiler_report();
    REQUIRE(rows[find_row(rows, "outer", -1)].live_bytes[0] == 0);
    REQUIRE(rows[find_row(rows, "outer", -1)].live_bytes[1] == 0);
}

TEST("profiler health names the task that holds memory") {
    prof::profiler_reset();
    char block = 0;
    {
        auto outer = prof::profiler::start_task("outer");
        auto inner = prof::profiler::start_task("inner");
        prof::profiler_record_alloc(&block, 4096, prof::profiler_memory_kind::host);
    }
    std::ostringstream out;
    prof::profiler_print_report(out, prof::profiler_report_format::text, 0.0);
    std::ostringstream health_only;
    prof::profiler_print_report(health_only, prof::profiler_report_format::health, 0.0);
    prof::profiler_record_free(&block);
    REQUIRE(health_only.str().rfind("oneDAL profiler health\n", 0) == 0);
    REQUIRE(health_only.str().find("outer > inner is still allocated") != std::string::npos);
    REQUIRE(health_only.str().find("Hotspots") == std::string::npos);
    const std::string text = out.str();
    REQUIRE(text.find("\nHealth\n") != std::string::npos);
    REQUIRE(
        text.find(
            "4.00 KiB of host memory in 1 block allocated in outer > inner is still allocated") !=
        std::string::npos);
    REQUIRE(text.find("host live") != std::string::npos);
}

TEST("profiler health reports a parallel task whose threads do unequal work") {
    prof::profiler_reset();
    const auto& hooks = prof::profiler_threading_hooks();
    {
        auto outer = prof::profiler::start_task("outer");
        const daal::profiler_region region = hooks.capture(true);
        std::vector<std::thread> workers;
        for (const int length : { 3000, 100, 100 }) {
            workers.emplace_back([&, length]() {
                run_chunk(region, [length]() {
                    auto chunk = prof::profiler::start_threading_task("uneven");
                    busy_wait(std::chrono::microseconds(length));
                });
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }
        hooks.finish(region);
    }
    std::ostringstream out;
    prof::profiler_print_report(out, prof::profiler_report_format::text, 0.0);
    REQUIRE(out.str().find("outer > uneven: load imbalance") != std::string::npos);
}

TEST("profiler health finds nothing in a plain sequential run") {
    prof::profiler_reset();
    {
        auto outer = prof::profiler::start_task("outer");
        auto inner = prof::profiler::start_task("inner");
        busy_wait(std::chrono::microseconds(100));
    }
    std::ostringstream out;
    prof::profiler_print_report(out, prof::profiler_report_format::text, 0.0);
    REQUIRE(out.str().find("Health\n  nothing unusual found\n") != std::string::npos);
}

TEST("profiler closes tasks left open above the one being closed") {
    prof::profiler_reset();
    {
        const int outer = prof::profiler_begin_task("outer", false);
        prof::profiler_begin_task("leaked", false);
        prof::profiler_end_task(outer);
    }
    const auto rows = prof::profiler_report();

    const auto outer = find_row(rows, "outer", -1);
    REQUIRE(outer >= 0);
    const auto leaked = find_row(rows, "leaked", outer);
    REQUIRE(leaked >= 0);
    REQUIRE(rows[outer].count == 1);
    REQUIRE(rows[leaked].count == 1);

    // Nothing is left open: the next task is top-level again.
    {
        auto next = prof::profiler::start_task("next");
    }
    REQUIRE(find_row(prof::profiler_report(), "next", -1) >= 0);
}

TEST("profiler logs a call site once unless every call is asked for") {
    // Without ONEDAL_PROFILER_OUTPUT, the log goes to stderr one line per call
    // site, so a pipe nobody reads while oneDAL runs cannot fill up and block the process.
    std::ostringstream captured;
    auto* previous = std::cerr.rdbuf(captured.rdbuf());
    for (int i = 0; i < 3; ++i) {
        prof::profiler_log("profiler test call site", "logged line\n");
    }
    prof::profiler_log(nullptr, "unkeyed line\n");
    prof::profiler_log(nullptr, "unkeyed line\n");
    std::cerr.rdbuf(previous);

    const std::string text = captured.str();
    const auto count = [&](const std::string& what) {
        std::size_t n = 0;
        for (auto at = text.find(what); at != std::string::npos; at = text.find(what, at + 1)) {
            ++n;
        }
        return n;
    };
    REQUIRE(count("logged line") == 1);
    REQUIRE(count("unkeyed line") == 2);
}

TEST("profiler reports a task with arguments once per distinct set of arguments") {
    prof::profiler_reset();
    {
        auto outer = prof::profiler::start_task("outer");
        for (int i = 0; i < 2; ++i) {
            auto gemm =
                prof::profiler::start_task_with_args("gemm",
                                                     prof::profiler_named_args("m, n", 100, 20));
        }
        {
            auto gemm =
                prof::profiler::start_task_with_args("gemm",
                                                     prof::profiler_named_args("m, n", 5, 20));
        }
        // Past a few variants, further argument sets share one row.
        for (int m = 0; m < 12; ++m) {
            auto other =
                prof::profiler::start_task_with_args("syrk", prof::profiler_named_args("m", m));
        }
    }
    const auto rows = prof::profiler_report();

    const auto outer = find_row(rows, "outer", -1);
    const auto big = find_row(rows, "gemm [m: 100; n: 20]", outer);
    const auto small = find_row(rows, "gemm [m: 5; n: 20]", outer);
    REQUIRE(big >= 0);
    REQUIRE(small >= 0);
    REQUIRE(rows[big].count == 2);
    REQUIRE(rows[small].count == 1);

    std::size_t syrk_rows = 0;
    for (const auto& row : rows) {
        syrk_rows += row.name.rfind("syrk [", 0) == 0 ? 1 : 0;
    }
    const auto rest = find_row(rows, "syrk [other args]", outer);
    REQUIRE(rest >= 0);
    REQUIRE(rows[rest].count == 4);
    REQUIRE(syrk_rows == 9);
}

TEST("profiler report lists the tree and the hotspots") {
    prof::profiler_reset();
    {
        auto outer = prof::profiler::start_task("outer");
        auto inner = prof::profiler::start_task("inner");
    }
    std::ostringstream out;
    prof::profiler_print_report(out);
    const std::string text = out.str();

    REQUIRE(text.find("oneDAL profiler report") != std::string::npos);
    REQUIRE(text.find("  outer\n") != std::string::npos);
    REQUIRE(text.find("`-- inner\n") != std::string::npos);
    REQUIRE(text.find("Hotspots by self time") != std::string::npos);
}

TEST("profiler gives the chunks of a region started at a known call site a row of their own") {
    prof::profiler_reset();
    const auto& hooks = prof::profiler_threading_hooks();
    const prof::profiler_site site = { "/path/to/kernel_impl.i", 42, "compute" };
    {
        auto outer = prof::profiler::start_task("outer");
        prof::profiler_set_region_site(&site);
        const daal::profiler_region region = hooks.capture(true);
        // The site names one region only.
        const daal::profiler_region unnamed = hooks.capture(true);
        REQUIRE(unnamed.name == nullptr);
        hooks.finish(unnamed);

        auto chunk = [&]() {
            run_chunk(region, []() {
                auto body = prof::profiler::start_task("body");
                busy_wait(std::chrono::microseconds(500));
            });
        };
        chunk();
        std::thread worker(chunk);
        worker.join();
        hooks.finish(region);
    }
    const auto rows = prof::profiler_report();

    const auto outer = find_row(rows, "outer", -1);
    const auto loop = find_row(rows, "parallel loop: compute (kernel_impl.i:42)", outer);
    REQUIRE(loop >= 0);
    REQUIRE(rows[loop].parallel);
    REQUIRE(rows[loop].count == 2);
    REQUIRE(rows[loop].threads == 2);
    const auto body = find_row(rows, "body", loop);
    REQUIRE(body >= 0);
    REQUIRE(rows[body].count == 2);
    REQUIRE(rows[loop].time_ns >= rows[body].time_ns);
}

TEST("threading layer names its regions after the parallel loop's call site") {
    prof::profiler_reset();
    {
        auto outer = prof::profiler::start_task("outer");
        daal::threader_for(64, 1, [](std::int64_t) {
            busy_wait(std::chrono::microseconds(50));
        });
    }
    const auto rows = prof::profiler_report();
    const auto outer = find_row(rows, "outer", -1);
    REQUIRE(outer >= 0);
    // Only a multi-threaded run starts a region.
    SKIP_IF(daal::threader_get_threads_number() < 2);
    bool found = false;
    for (const auto& row : rows) {
        found |= row.parent == outer && row.name.rfind("parallel loop: ", 0) == 0 &&
                 row.name.find("profiler.cpp:") != std::string::npos;
    }
    REQUIRE(found);
}

TEST("profiler keeps the tasks of independent calling threads apart") {
    prof::profiler_reset();
    std::mutex m;
    std::condition_variable cv;
    int step = 0;
    const auto wait_for = [&](int value) {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [&] {
            return step >= value;
        });
    };
    const auto advance = [&]() {
        {
            std::lock_guard<std::mutex> lock(m);
            ++step;
        }
        cv.notify_all();
    };
    // Two threads call oneDAL at once, as two Python threads may; their tasks overlap in time and
    // close out of order.
    std::thread first([&]() {
        {
            auto fit = prof::profiler::start_task("fit_first");
            advance();
            wait_for(2);
        }
        advance();
    });
    std::thread second([&]() {
        wait_for(1);
        auto fit = prof::profiler::start_task("fit_second");
        advance();
        wait_for(3);
    });
    first.join();
    second.join();
    const auto rows = prof::profiler_report();

    const auto first_fit = find_row(rows, "fit_first", -1);
    const auto second_fit = find_row(rows, "fit_second", -1);
    REQUIRE(first_fit >= 0);
    REQUIRE(second_fit >= 0);
    REQUIRE(!rows[first_fit].parallel);
    REQUIRE(!rows[second_fit].parallel);
    REQUIRE(task_count(rows) == 2);
}

TEST("profiler matches every free to its allocation across many live blocks") {
    prof::profiler_reset();
    // Enough blocks to grow the address table several times and to collide in it.
    constexpr std::size_t count = 50000;
    std::vector<char> blocks(count);
    std::vector<std::size_t> order(count);
    for (std::size_t i = 0; i < count; ++i) {
        order[i] = (i * 7919) % count;
    }
    {
        auto outer = prof::profiler::start_task("outer");
        for (std::size_t i = 0; i < count; ++i) {
            prof::profiler_record_alloc(&blocks[i], i + 1, prof::profiler_memory_kind::host);
        }
        for (const std::size_t i : order) {
            prof::profiler_record_free(&blocks[i]);
        }
    }
    const auto rows = prof::profiler_report();
    const auto outer = find_row(rows, "outer", -1);
    REQUIRE(outer >= 0);
    const std::uint64_t bytes = count * (count + 1) / 2;
    REQUIRE(rows[outer].alloc_count[0] == count);
    REQUIRE(rows[outer].free_count[0] == count);
    REQUIRE(rows[outer].alloc_bytes[0] == bytes);
    REQUIRE(rows[outer].free_bytes[0] == bytes);
    REQUIRE(rows[outer].peak_bytes[0] == bytes);
}

TEST("profiler writes the call paths and their self times for flame graph tools") {
    prof::profiler_reset();
    {
        auto outer = prof::profiler::start_task("outer");
        auto inner =
            prof::profiler::start_task_with_args("inner", prof::profiler_named_args("m, n", 3, 4));
        busy_wait(std::chrono::microseconds(200));
    }
    std::ostringstream out;
    prof::profiler_print_report(out, prof::profiler_report_format::folded, 0.0);
    const std::string text = out.str();
    // Semicolons separate the frames, so the ones in the arguments become commas.
    const auto line = text.find("outer;inner [m: 3, n: 4] ");
    REQUIRE(line != std::string::npos);
    const auto value = std::stoull(text.substr(text.find(' ', text.find(']', line)) + 1));
    REQUIRE(value >= 200000);
}

TEST("profiler writes every row with its figures as JSON") {
    prof::profiler_reset();
    {
        auto outer = prof::profiler::start_task("outer \"quoted\"");
        auto inner = prof::profiler::start_task("inner");
    }
    std::ostringstream out;
    prof::profiler_print_report(out, prof::profiler_report_format::json, 0.0);
    const std::string text = out.str();
    REQUIRE(text.rfind("{\"rows\":[", 0) == 0);
    REQUIRE(text.find("\"name\":\"outer \\\"quoted\\\"\",\"parent\":-1") != std::string::npos);
    REQUIRE(text.find("\"name\":\"inner\",\"parent\":0,\"depth\":1") != std::string::npos);
    REQUIRE(text.find("\"first_ns\":") != std::string::npos);
    REQUIRE(text.find("\"live_bytes\":[") != std::string::npos);
}

TEST("profiler report folds the rows below the threshold into one per parent") {
    prof::profiler_reset();
    {
        auto outer = prof::profiler::start_task("outer");
        {
            auto big = prof::profiler::start_task("big");
            busy_wait(std::chrono::microseconds(5000));
        }
        for (const char* name : { "tiny_a", "tiny_b" }) {
            auto tiny = prof::profiler::start_task(name);
        }
    }
    std::ostringstream out;
    prof::profiler_print_report(out, prof::profiler_report_format::tree, 5.0);
    const std::string text = out.str();
    REQUIRE(text.find("big") != std::string::npos);
    REQUIRE(text.find("tiny_a") == std::string::npos);
    REQUIRE(text.find("(2 more tasks under the report threshold)") != std::string::npos);
}

TEST("profiler reports the first run of a task apart from the rest") {
    prof::profiler_reset();
    {
        auto outer = prof::profiler::start_task("outer");
        for (int i = 0; i < 4; ++i) {
            auto step = prof::profiler::start_task("step");
            busy_wait(std::chrono::microseconds(i == 0 ? 3000 : 100));
        }
    }
    const auto rows = prof::profiler_report();
    const auto step = find_row(rows, "step", find_row(rows, "outer", -1));
    REQUIRE(step >= 0);
    REQUIRE(rows[step].first_ns >= 3000000);
    REQUIRE(rows[step].first_ns == rows[step].max_ns);
}

TEST("profiler clear sets the figures to zero and keeps recording") {
    prof::profiler_reset();
    {
        auto before = prof::profiler::start_task("before");
    }
    prof::profiler_clear();
    REQUIRE(prof::profiler_report().empty());
    {
        auto after = prof::profiler::start_task("after");
    }
    const auto rows = prof::profiler_report();
    REQUIRE(find_row(rows, "after", -1) >= 0);
    REQUIRE(find_row(rows, "before", -1) == -1);
}

TEST("profiler report can be requested through the C entry point") {
    prof::profiler_reset();
    {
        auto task = prof::profiler::start_task("c_entry_point_task");
    }
    const std::string path = "profiler_c_entry_point_report.txt";
    std::remove(path.c_str());
    _onedal_profiler_report(path.c_str(), 1);
    std::ifstream in(path);
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::remove(path.c_str());
    REQUIRE(text.find("c_entry_point_task") != std::string::npos);
    REQUIRE(prof::profiler_report().empty());
}

#ifndef _WIN32
// The signatures below are in the GCC and Clang form; Windows builds read __FUNCSIG__.
TEST("profiler names a kernel after its type without the namespaces") {
    REQUIRE(oneapi::dal::detail::profiler_short_type_name(
                "const char* oneapi::dal::detail::profiler_type_name() [with T = "
                "oneapi::dal::kmeans::backend::train_kernel_cpu<float, "
                "oneapi::dal::kmeans::method::v1::lloyd_dense, "
                "oneapi::dal::kmeans::task::v1::clustering>]") ==
            "kmeans::train_kernel_cpu<float, lloyd_dense, clustering>");
    REQUIRE(
        oneapi::dal::detail::profiler_short_type_name(
            "const char *oneapi::dal::detail::profiler_type_name() [T = "
            "oneapi::dal::pca::backend::train_kernel_gpu<double,oneapi::dal::pca::method::v1::cov>]") ==
        "pca::train_kernel_gpu<double, cov>");
}
#endif

} // namespace oneapi::dal::test
