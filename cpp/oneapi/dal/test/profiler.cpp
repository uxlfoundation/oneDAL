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
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/detail/memory.hpp"
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

TEST("profiler keeps the recording when the report file cannot be written") {
    start_recording();
    {
        auto task = prof::profiler::start_task("kept");
    }
    REQUIRE(_onedal_profiler_report("profiler_no_such_dir/report.txt", 1) == -1);
    const rows_t rows = prof::profiler_report();
    REQUIRE(find_row(rows, "kept", -1) >= 0);
}

TEST("profiler report can be built while another thread adds tasks") {
    start_recording();
    // Distinct names make the worker add a node on every task, which grows its node list.
    constexpr int task_count = 4000;
    static std::vector<std::string> names = [] {
        std::vector<std::string> result(task_count);
        for (int i = 0; i < task_count; ++i) {
            result[i] = "concurrent_" + std::to_string(i);
        }
        return result;
    }();
    std::atomic<bool> started{ false };
    std::atomic<bool> done{ false };
    std::thread worker([&] {
        while (!started.load()) {
        }
        for (int i = 0; i < task_count; ++i) {
            auto task = prof::profiler::start_task(names[i].c_str());
        }
        done.store(true);
    });
    std::size_t max_rows = 0;
    started.store(true);
    while (!done.load()) {
        max_rows = std::max(max_rows, prof::profiler_report().size());
    }
    worker.join();
    REQUIRE(max_rows <= static_cast<std::size_t>(task_count));
    REQUIRE(prof::profiler_report().size() == static_cast<std::size_t>(task_count));
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
    REQUIRE(rows[0].on_queue);
}
#endif

TEST("profiler labels a task timed on a SYCL queue apart from host tasks") {
    start_recording();
    {
        auto host = prof::profiler::start_task("host_task");
        auto queued = prof::profiler::start_queue_task("queue_task");
    }
    const rows_t rows = prof::profiler_report();
    const auto host = find_row(rows, "host_task", -1);
    const auto queued = find_row(rows, "queue_task", host);
    REQUIRE(host >= 0);
    REQUIRE(queued >= 0);
    REQUIRE_FALSE(rows[host].on_queue);
    REQUIRE(rows[queued].on_queue);
    REQUIRE_FALSE(rows[queued].parallel);
    std::ostringstream out;
    prof::profiler_print_report(out);
    const std::string text = out.str();
    REQUIRE(text.find("host_task time: ") != std::string::npos);
    REQUIRE(text.find("1 times in a sequential region") != std::string::npos);
    REQUIRE(text.find("1 times on a SYCL queue") != std::string::npos);
}

constexpr int host_memory = static_cast<int>(prof::profiler_memory_kind::host);
constexpr int device_memory = static_cast<int>(prof::profiler_memory_kind::device);

TEST("profiler charges recorded blocks to the open task and keeps the peak of a sequential task") {
    start_recording();
    // Only the addresses are recorded, so any distinct objects stand for the blocks.
    char first = 0, second = 0, third = 0;
    {
        auto outer = prof::profiler::start_task("outer");
        prof::profiler_record_alloc(&first, 1000, prof::profiler_memory_kind::host);
        {
            auto inner = prof::profiler::start_task("inner");
            prof::profiler_record_alloc(&second, 4000, prof::profiler_memory_kind::host);
            prof::profiler_record_free(&second);
            prof::profiler_record_alloc(&third, 500, prof::profiler_memory_kind::device);
        }
        prof::profiler_record_free(&first);
    }
    const rows_t rows = prof::profiler_report();
    const auto outer = find_row(rows, "outer", -1);
    const auto inner = find_row(rows, "inner", outer);
    REQUIRE(outer >= 0);
    REQUIRE(inner >= 0);
    // A row covers its children.
    REQUIRE(rows[outer].alloc_bytes[host_memory] == 5000);
    REQUIRE(rows[outer].alloc_count[host_memory] == 2);
    REQUIRE(rows[outer].alloc_bytes[device_memory] == 500);
    REQUIRE(rows[inner].alloc_bytes[host_memory] == 4000);
    REQUIRE(rows[inner].alloc_count[host_memory] == 1);
    // `first` and `second` were live together; `inner` started after `first`.
    REQUIRE(rows[outer].peak_bytes[host_memory] == 5000);
    REQUIRE(rows[inner].peak_bytes[host_memory] == 4000);
    REQUIRE(rows[inner].peak_bytes[device_memory] == 500);
    // `third` is still allocated, by `inner`.
    REQUIRE(rows[inner].held_bytes[device_memory] == 500);
    REQUIRE(rows[outer].held_bytes[device_memory] == 500);
    REQUIRE(rows[outer].held_bytes[host_memory] == 0);
    prof::profiler_record_free(&third);
    REQUIRE(prof::profiler_report()[static_cast<std::size_t>(inner)].held_bytes[device_memory] ==
            0);
}

TEST(
    "profiler charges a block allocated in a chunk with no task open to the task that started the region") {
    start_recording();
    constexpr std::int64_t n = 64;
    std::vector<char> blocks(n);
    {
        auto outer = prof::profiler::start_task("outer");
        daal::threader_for(n, 1, [&](std::int64_t i) {
            prof::profiler_record_alloc(&blocks[i], 100, prof::profiler_memory_kind::host);
        });
        daal::threader_for(n, 1, [&](std::int64_t i) {
            prof::profiler_record_free(&blocks[i]);
        });
    }
    const rows_t rows = prof::profiler_report();
    REQUIRE(rows.size() == 1);
    REQUIRE(rows[0].alloc_bytes[host_memory] == 100 * n);
    REQUIRE(rows[0].alloc_count[host_memory] == n);
    REQUIRE(rows[0].peak_bytes[host_memory] <= 100 * n);
    REQUIRE(rows[0].held_bytes[host_memory] == 0);
}

TEST("profiler ignores a free of a block it never recorded") {
    start_recording();
    char block = 0, other = 0;
    {
        auto outer = prof::profiler::start_task("outer");
        prof::profiler_record_alloc(&block, 64, prof::profiler_memory_kind::host);
        prof::profiler_record_free(&other);
    }
    const rows_t rows = prof::profiler_report();
    REQUIRE(rows.size() == 1);
    REQUIRE(rows[0].held_bytes[host_memory] == 64);
    prof::profiler_record_free(&block);
}

constexpr int host_to_device = static_cast<int>(prof::profiler_transfer_kind::host_to_device);
constexpr int device_to_host = static_cast<int>(prof::profiler_transfer_kind::device_to_host);
constexpr int device_to_device = static_cast<int>(prof::profiler_transfer_kind::device_to_device);

TEST("profiler charges recorded copies to the open task by direction") {
    start_recording();
    constexpr std::int64_t n = 64;
    prof::profiler_record_transfer(7, prof::profiler_transfer_kind::host_to_device);
    {
        auto outer = prof::profiler::start_task("outer");
        prof::profiler_record_transfer(1000, prof::profiler_transfer_kind::host_to_device);
        {
            auto inner = prof::profiler::start_task("inner");
            prof::profiler_record_transfer(3000, prof::profiler_transfer_kind::host_to_device);
            prof::profiler_record_transfer(40, prof::profiler_transfer_kind::device_to_host);
        }
        // A chunk with no task open charges the task that started the region.
        daal::threader_for(n, 1, [&](std::int64_t) {
            prof::profiler_record_transfer(10, prof::profiler_transfer_kind::device_to_device);
        });
    }
    const rows_t rows = prof::profiler_report();
    const auto outer = find_row(rows, "outer", -1);
    const auto inner = find_row(rows, "inner", outer);
    REQUIRE(rows.size() == 2);
    REQUIRE(outer >= 0);
    REQUIRE(inner >= 0);
    // A row covers its children; the copy made outside any task is on no row.
    REQUIRE(rows[outer].transfer_bytes[host_to_device] == 4000);
    REQUIRE(rows[outer].transfer_count[host_to_device] == 2);
    REQUIRE(rows[outer].transfer_bytes[device_to_host] == 40);
    REQUIRE(rows[outer].transfer_count[device_to_host] == 1);
    REQUIRE(rows[outer].transfer_bytes[device_to_device] == 10 * n);
    REQUIRE(rows[outer].transfer_count[device_to_device] == n);
    REQUIRE(rows[inner].transfer_bytes[host_to_device] == 3000);
    REQUIRE(rows[inner].transfer_count[host_to_device] == 1);
    REQUIRE(rows[inner].transfer_count[device_to_host] == 1);
    REQUIRE(rows[inner].transfer_count[device_to_device] == 0);
    std::ostringstream out;
    prof::profiler_print_report(out);
    const std::string text = out.str();
    REQUIRE(text.find(", host to device 3.91KB in 2 copies, device to host 40B in 1 copy, "
                      "device to device 640B in 64 copies\n") != std::string::npos);
    REQUIRE(text.find(", host to device 2.93KB in 1 copy, device to host 40B in 1 copy\n") !=
            std::string::npos);
}

#ifdef ONEDAL_DATA_PARALLEL
TEST("profiler takes the direction of a copy from the USM types of its pointers") {
    start_recording();
    auto& queue = te::test_queue_provider::get_instance().get_global_queue();
    constexpr std::int64_t n = 256;
    constexpr std::size_t bytes = n * sizeof(float);
    std::vector<float> host(n, 1.0f), host_back(n);
    float* device = sycl::malloc_device<float>(n, queue);
    float* shared = sycl::malloc_shared<float>(n, queue);
    float* usm_host = sycl::malloc_host<float>(n, queue);
    REQUIRE(device);
    REQUIRE(shared);
    REQUIRE(usm_host);
    {
        auto task = prof::profiler::start_task("copies");
        detail::profiler_record_copy(queue, device, host.data(), bytes);
        detail::profiler_record_copy(queue, device, usm_host, bytes);
        detail::profiler_record_copy(queue, host_back.data(), device, bytes);
        detail::profiler_record_copy(queue, shared, device, bytes);
        detail::profiler_record_copy(queue, device, shared, bytes);
        // Neither side is device memory.
        detail::profiler_record_copy(queue, host_back.data(), usm_host, bytes);
    }
    sycl::free(device, queue);
    sycl::free(shared, queue);
    sycl::free(usm_host, queue);
    const rows_t rows = prof::profiler_report();
    REQUIRE(rows.size() == 1);
    REQUIRE(rows[0].transfer_count[host_to_device] == 2);
    REQUIRE(rows[0].transfer_bytes[host_to_device] == 2 * bytes);
    REQUIRE(rows[0].transfer_count[device_to_host] == 1);
    REQUIRE(rows[0].transfer_bytes[device_to_host] == bytes);
    REQUIRE(rows[0].transfer_count[device_to_device] == 2);
    REQUIRE(rows[0].transfer_bytes[device_to_device] == 2 * bytes);
}

TEST("profiler records the copies of the memcpy helpers only when it is on") {
    start_recording();
    auto& queue = te::test_queue_provider::get_instance().get_global_queue();
    const detail::data_parallel_policy policy{ queue };
    constexpr std::int64_t n = 256;
    constexpr std::size_t bytes = n * sizeof(float);
    std::vector<float> host(n, 2.0f), host_back(n, 0.0f);
    float* device = sycl::malloc_device<float>(n, queue);
    float* other = sycl::malloc_device<float>(n, queue);
    REQUIRE(device);
    REQUIRE(other);
    {
        auto task = prof::profiler::start_task("copies");
        detail::memcpy_host2usm(policy, device, host.data(), bytes);
        detail::memcpy(policy, other, device, bytes);
        detail::memcpy_usm2host(policy, host_back.data(), other, bytes);
    }
    sycl::free(device, queue);
    sycl::free(other, queue);
    REQUIRE(host_back == host);
    const rows_t rows = prof::profiler_report();
    REQUIRE(rows.size() == 1);
    // ONEDAL_VERBOSE decides, as for the library; with it unset nothing is recorded.
    const std::uint64_t expected = prof::is_transfer_tracking_enabled() ? 1 : 0;
    for (int k : { host_to_device, device_to_host, device_to_device }) {
        REQUIRE(rows[0].transfer_count[k] == expected);
        REQUIRE(rows[0].transfer_bytes[k] == expected * bytes);
    }
}
#endif

} // namespace oneapi::dal::test
