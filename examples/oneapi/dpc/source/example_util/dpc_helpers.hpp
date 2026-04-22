/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#pragma once

#include <vector>
#include <sycl/sycl.hpp>

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/detail/array_utils.hpp"

void try_add_device(std::vector<sycl::device>& devices, int (*selector)(const sycl::device&)) {
    try {
        devices.push_back(sycl::ext::oneapi::detail::select_device(selector));
    }
    catch (...) {
    }
}

std::vector<sycl::device> list_devices() {
    std::vector<sycl::device> devices;
    try_add_device(devices, &sycl::cpu_selector_v);
    try_add_device(devices, &sycl::gpu_selector_v);
    return devices;
}
