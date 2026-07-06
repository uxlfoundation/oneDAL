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

#include <chrono>
#include <thread>
#include <iostream>
#include <optional>

#include <fmt/format.h>

#include "zexx.hpp"
#include "utils.hpp"

void display_frequency(std::uint32_t index, const zexx::frequency_domain& domain) {
    const auto properties = domain.get_properties();
    fmt::print("   Min frequency ({}): {}MHz\n", properties.get_type_name(), properties.get_min());
    fmt::print("   Max frequency ({}): {}MHz\n", properties.get_type_name(), properties.get_max());
}

void display_devices_info(std::uint32_t index, const zexx::device& device) {
    const auto properties = device.get_properties();
    fmt::print("[{}:{}] {} {}\n",
               properties.get_type_name(),
               index,
               properties.get_name(),
               properties.is_integrated() ? "(integrated)" : "");
    fmt::print("   Threads per EU: {}\n", properties.get_thread_count_per_unit());
    fmt::print("   EU SIMD width (32-bit lane): {}\n", properties.get_simd_width());
    fmt::print("   EUs layout (EUs x sub-slices x slices): {} x {} x {} = {}\n",
               properties.get_unit_count_per_subslice(),
               properties.get_subslices_count_per_slice(),
               properties.get_slice_count(),
               properties.get_total_unit_count());
    fmt::print("   Peak f32_mad/clock: {}\n", properties.get_float_mads_per_clock());

    const auto freq_domains = device.get_frequency_domains();
    for (auto [i, domain] : enumerate{ freq_domains }) {
        display_frequency(i, domain);
        if (i + 1 < freq_domains.size()) {
            fmt::print("\n");
        }
    }
}

void display_devices_info(std::uint32_t index, const zexx::driver& driver) {
    const auto devices = driver.get_devices();
    if (devices.empty()) {
        throw std::runtime_error{ "Cannot find any device" };
    }

    for (auto [i, device] : enumerate{ devices }) {
        display_devices_info(i, device);
        if (i + 1 < devices.size()) {
            fmt::print("\n");
        }
    }
}

void display_devices_info() {
    zexx::system_manager sysman;

    const auto drivers = sysman.get_drivers();
    if (drivers.empty()) {
        throw std::runtime_error{ "Cannot find any driver" };
    }

    for (auto [i, driver] : enumerate{ drivers }) {
        display_devices_info(i, driver);
        if (i + 1 < drivers.size()) {
            fmt::print("\n");
        }
    }
}

class frequency_monitor {
public:
    explicit frequency_monitor(std::uint32_t index, const zexx::device& device) {
        name_ = fmt::format("{}:{}", device.get_properties().get_type_name(), index);
        for (auto domain : device.get_frequency_domains()) {
            if (domain.get_properties().get_type_name() == "core") {
                core_ = domain;
            }
        }
        if (!core_) {
            throw std::runtime_error{ "Cannot query GPU frequency" };
        }
    }

    double get_core_frequency() const {
        return core_->get_state().get_actual();
    }

    const std::string& get_name() const {
        return name_;
    }

private:
    std::optional<zexx::frequency_domain> core_;
    std::string name_;
};

std::vector<frequency_monitor> create_frequency_monitors() {
    zexx::system_manager sysman;
    const auto devices = sysman.get_devices();

    std::vector<frequency_monitor> monitors;
    monitors.reserve(devices.size());

    for (auto [i, device] : enumerate{ devices }) {
        monitors.emplace_back(i, device);
    }

    return monitors;
}

void display_frequency_monitors() {
    constexpr char move_cursor_up[] = "\033[A\033[K";

    const auto monitors = create_frequency_monitors();
    while (true) {
        for (auto& monitor : monitors) {
            fmt::print("[{}]: {}MHz\n", monitor.get_name(), monitor.get_core_frequency());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::cout << move_cursor_up << move_cursor_up << std::flush;
    }
}

void dispatch(int argc, char const* argv[]) {
    if (argc <= 1) {
        return display_devices_info();
    }

    const std::string command = argv[1];

    if (command == "info") {
        display_devices_info();
    }
    else if (command == "freq") {
        display_frequency_monitors();
    }
    else {
        throw std::runtime_error{ "Unknown command: `" + command + "`" };
    }
}

int main(int argc, char const* argv[]) {
    try {
        dispatch(argc, argv);
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }
}
