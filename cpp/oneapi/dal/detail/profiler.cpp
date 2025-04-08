/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifdef _WIN
#include <windows.h>
#define PATH_MAX MAX_PATH
#else
#ifndef _GNU_SOURCE // must be before all includes or features.h to define __USE_GNU
#define _GNU_SOURCE
#endif
#include <linux/limits.h>
#endif

#include <sstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include "oneapi/dal/detail/profiler.hpp"
#include <daal/include/services/library_version_info.h>

namespace oneapi::dal::detail {

#ifdef ONEDAL_REF
static volatile int onedal_verbose_val __attribute__((aligned(64))) = -1;
#else
__declspec(align(64)) static volatile int onedal_verbose_val = -1;
#endif

static bool device_info_printed = false;
static bool device_info_needed = false;
static bool kernel_info_needed = false;
//__declspec(align(64)) static volatile char verbose_file_val[PATH_MAX] = {'\0'};

#define ONEDAL_VERBOSE_ENV      "ONEDAL_VERBOSE"
#define ONEDAL_VERBOSE_FILE_ENV "ONEDAL_VERBOSE_OUTPUT_FILE"

/**
* Returns the pointer to variable that holds oneDAL verbose mode information (enabled/disabled)
*
*  @returns pointer to mode
*                      0 disabled
*                      1 enabled
*                      2 enabled with device and library information(will be added soon)
*/

static void set_verbose_from_env(void) {
    static volatile int read_done = 0;
    if (read_done)
        return;

    const char* verbose_str = std::getenv("ONEDAL_VERBOSE");
    int newval = 0;
    if (verbose_str) {
        newval = std::atoi(verbose_str);
        if (newval < 0 || newval > 2) {
            newval = 0;
        }
    }

    onedal_verbose_val = newval;
    read_done = 1;
}

void print_device_info(sycl::device dev) {
    std::cout << "Platfrom: " << dev.get_platform().get_info<sycl::info::platform::name>()
              << std::endl;

    std::cout << "\tName:\t" << dev.get_info<sycl::info::device::name>() << std::endl;

    std::string device = "UNKNOWN";
    switch (dev.get_info<sycl::info::device::device_type>()) {
        case sycl::info::device_type::gpu: device = "GPU"; break;
        case sycl::info::device_type::cpu: device = "CPU"; break;
        case sycl::info::device_type::accelerator: device = "ACCELERATOR"; break;
        case sycl::info::device_type::host: device = "HOST"; break;
        default: break;
    }
    std::cout << "\tType:\t" << device << std::endl;
    std::cout << "\tVendor:\t" << dev.get_info<sycl::info::device::vendor>() << std::endl;
    std::cout << "\tVersion:\t" << dev.get_info<sycl::info::device::version>() << std::endl;
    std::cout << "\tDriver:\t" << dev.get_info<sycl::info::device::driver_version>() << std::endl;

    std::cout << "\tMax freq:\t" << dev.get_info<sycl::info::device::max_clock_frequency>()
              << std::endl;

    std::cout << "\tMax comp units:\t" << dev.get_info<sycl::info::device::max_compute_units>()
              << std::endl;
    std::cout << "\tMax work item dims:\t"
              << dev.get_info<sycl::info::device::max_work_item_dimensions>() << std::endl;
    std::cout << "\tMax work group size:\t"
              << dev.get_info<sycl::info::device::max_work_group_size>() << std::endl;
    std::cout << "\tGlobal mem size:\t" << dev.get_info<sycl::info::device::global_mem_size>()
              << std::endl;
    std::cout << "\tGlobal mem cache size:\t"
              << dev.get_info<sycl::info::device::global_mem_cache_size>() << std::endl;
    std::cout << std::endl;
}

std::string format_time_for_output(std::uint64_t time_ns) {
    std::ostringstream out;
    double time = static_cast<double>(time_ns);

    if (time <= 0) {
        out << "0.00s";
    }
    else {
        if (time > 1e9) {
            out << std::fixed << std::setprecision(2) << time / 1e9 << "s";
        }
        else if (time > 1e6) {
            out << std::fixed << std::setprecision(2) << time / 1e6 << "ms";
        }
        else if (time > 1e3) {
            out << std::fixed << std::setprecision(2) << time / 1e3 << "us";
        }
        else {
            out << static_cast<std::uint64_t>(time) << "ns";
        }
    }

    return out.str();
}

void print_header() {
    daal::services::LibraryVersionInfo ver;

    std::cout << "Major version:          " << ver.majorVersion << std::endl;
    std::cout << "Minor version:          " << ver.minorVersion << std::endl;
    std::cout << "Update version:         " << ver.updateVersion << std::endl;
    std::cout << "Product status:         " << ver.productStatus << std::endl;
    std::cout << "Build:                  " << ver.build << std::endl;
    std::cout << "Build revision:         " << ver.build_rev << std::endl;
    std::cout << "Name:                   " << ver.name << std::endl;
    std::cout << "Processor optimization: " << ver.processor << std::endl;
    std::cout << std::endl;
}

int* onedal_verbose_mode() {
    if (__builtin_expect((onedal_verbose_val == -1), 0)) {
        // ADD MUTEX
        if (onedal_verbose_val == -1)
            set_verbose_from_env();
        // DISABLE MUTEX
    }
    return (int*)&onedal_verbose_val;
}

int onedal_verbose(int option) {
    int* retVal = onedal_verbose_mode();
    if (option != 0 && option != 1 && option != 2) {
        return -1;
    }
    if (option != onedal_verbose_val) {
        // ADD MUTEX
        if (option != onedal_verbose_val)
            onedal_verbose_val = option;
    }
    return *retVal;
}

profiler::profiler() {
    int verbose = *onedal_verbose_mode();
    if (verbose == 1) {
        kernel_info_needed = true;
    }
    else if (verbose == 2) {
        device_info_needed = true;
        kernel_info_needed = true;
    }

    if (device_info_needed) {
        print_header();
    }
    start_time = get_time();
}

profiler::~profiler() {
    if (kernel_info_needed) {
        std::cerr << "ONEDAL KERNEL_PROFILER: ALL KERNELS total time "
                  << format_time_for_output(total_time) << std::endl;
    }
}

std::uint64_t profiler::get_time() {
#ifdef _WIN32
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (counter.QuadPart * 1000000000) / frequency.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000000000 + t.tv_nsec;
#endif
}

profiler* profiler::get_instance() {
    static profiler instance;
    return &instance;
}

task& profiler::get_task() {
    return task_;
}

#ifdef ONEDAL_DATA_PARALLEL
sycl::queue& profiler::get_queue() {
    return queue_;
}

void profiler::set_queue(const sycl::queue& q) {
    queue_ = q;
}
#endif

profiler_task profiler::start_task(const char* task_name) {
    auto ns_start = get_time();
    auto& tasks_info = get_instance()->get_task();
    tasks_info.time_kernels.push_back(ns_start);
    return profiler_task(task_name);
}

void profiler::end_task(const char* task_name) {
    const std::uint64_t ns_end = get_time();
    auto& tasks_info = get_instance()->get_task();
    if (tasks_info.time_kernels.empty()) {
        throw std::runtime_error("Attempting to end a task when no tasks are running");
    }

#ifdef ONEDAL_DATA_PARALLEL
    auto& queue = get_instance()->get_queue();
    queue.wait_and_throw();
#endif

    const std::uint64_t ns_start = tasks_info.time_kernels.back();
    tasks_info.time_kernels.pop_back();
    const std::uint64_t times = ns_end - ns_start;

    auto it = tasks_info.kernels.find(task_name);
    if (it == tasks_info.kernels.end()) {
        tasks_info.kernels.insert({ std::string(task_name), times });
    }
    else {
        it->second += times;
    }
    get_instance()->total_time += times;
    if (kernel_info_needed) {
        std::cerr << "ONEDAL KERNEL_PROFILER: " << std::string(task_name) << " ";
        std::cerr << format_time_for_output(times) << std::endl;
    }
}

#ifdef ONEDAL_DATA_PARALLEL
profiler_task profiler::start_task(const char* task_name, sycl::queue& task_queue) {
    task_queue.wait_and_throw();
    if (device_info_printed == false && device_info_needed == true) {
        auto device = task_queue.get_device();
        print_device_info(device);
        device_info_printed = true;
    }
    get_instance()->set_queue(task_queue);
    return start_task(task_name);
}

profiler_task::profiler_task(const char* task_name, const sycl::queue& task_queue)
        : task_name_(task_name),
          task_queue_(task_queue),
          has_queue_(true) {}
#endif

profiler_task::profiler_task(const char* task_name) : task_name_(task_name) {}

profiler_task::~profiler_task() {
#ifdef ONEDAL_DATA_PARALLEL
    if (has_queue_)
        task_queue_.wait_and_throw();
#endif // ONEDAL_DATA_PARALLEL
    profiler::end_task(task_name_);
}

} // namespace oneapi::dal::detail
