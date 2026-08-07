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

#pragma once

#include <cassert>
#include <stdexcept>
#include <vector>
#include <optional>

#include <fmt/format.h>
#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>

namespace zexx {

inline std::string get_ze_error_message(ze_result_t status) {
    switch (status) {
        case ZE_RESULT_SUCCESS: return "success";
        case ZE_RESULT_NOT_READY: return "synchronization primitive not signaled";
        case ZE_RESULT_ERROR_DEVICE_LOST:
            return "device hung, reset, was removed, or driver update occurred";
        case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY: return "insufficient host memory to satisfy call";
        case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
            return "insufficient device memory to satisfy call";
        case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:
            return "error occurred when building module, see build log for details";
        case ZE_RESULT_ERROR_MODULE_LINK_FAILURE:
            return "error occurred when linking modules, see build log for details";
        case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS:
            return "access denied due to permission level";
        case ZE_RESULT_ERROR_NOT_AVAILABLE:
            return "resource already in use and simultaneous access not allowed or resource was removed";
        case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE:
            return "external required dependency is unavailable or missing";
        case ZE_RESULT_ERROR_UNINITIALIZED: return "driver is not initialized";
        case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:
            return "generic error code for unsupported versions";
        case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
            return "generic error code for unsupported features";
        case ZE_RESULT_ERROR_INVALID_ARGUMENT: return "generic error code for invalid arguments";
        case ZE_RESULT_ERROR_INVALID_NULL_HANDLE: return "handle argument is not valid";
        case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
            return "object pointed to by handle still in-use by device";
        case ZE_RESULT_ERROR_INVALID_NULL_POINTER: return "pointer argument may not be nullptr";
        case ZE_RESULT_ERROR_INVALID_SIZE:
            return "size argument is invalid (e.g., must not be zero)";
        case ZE_RESULT_ERROR_UNSUPPORTED_SIZE:
            return "size argument is not supported by the device (e.g., too large)";
        case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
            return "alignment argument is not supported by the device (e.g.,";
        case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
            return "synchronization object in invalid state";
        case ZE_RESULT_ERROR_INVALID_ENUMERATION: return "enumerator argument is not valid";
        case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
            return "enumerator argument is not supported by the device";
        case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
            return "image format is not supported by the device";
        case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY:
            return "native binary is not supported by the device";
        case ZE_RESULT_ERROR_INVALID_GLOBAL_NAME:
            return "global variable is not found in the module";
        case ZE_RESULT_ERROR_INVALID_KERNEL_NAME: return "kernel name is not found in the module";
        case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME:
            return "function name is not found in the module";
        case ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
            return "group size dimension is not valid for the kernel or";
        case ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
            return "global width dimension is not valid for the kernel or";
        case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
            return "kernel argument index is not valid for kernel";
        case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
            return "kernel argument size does not match kernel";
        case ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
            return "value of kernel attribute is not valid for the kernel or";
        case ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED:
            return "module with imports needs to be linked before kernels can";
        case ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE:
            return "command list type does not match command queue type";
        case ZE_RESULT_ERROR_OVERLAPPING_REGIONS:
            return "copy operations do not support overlapping regions of";
        default: return "unknown or internal error";
    }
}

inline void check_ze(ze_result_t status) {
    if (status != ZE_RESULT_SUCCESS) {
        throw std::runtime_error{ fmt::format("L0 returned error: {}",
                                              get_ze_error_message(status)) };
    }
}

template <typename T, typename Init>
inline T get_or_init(std::optional<T>& optional_value, Init&& init) {
    if (!optional_value) {
        optional_value = T{ init() };
    }
    return *optional_value;
}

template <typename Handle>
class handled {
public:
    explicit handled(const Handle& handle) : handle_(handle) {
        if constexpr (std::is_pointer_v<Handle>) {
            assert(handle != nullptr);
        }
    }

protected:
    const Handle& get_handle() const {
        return handle_;
    }

private:
    Handle handle_;
};

class frequency_domain_properties : public handled<zes_freq_properties_t> {
public:
    using handled<zes_freq_properties_t>::handled;

    std::string get_type_name() const {
        return get_or_init(type_name_, [&]() {
            switch (get_handle().type) {
                case ZES_FREQ_DOMAIN_GPU: return "core";
                case ZES_FREQ_DOMAIN_MEMORY: return "memory";
                default: return "Unknown";
            }
        });
    }

    double get_min() const {
        return get_handle().min;
    }

    double get_max() const {
        return get_handle().max;
    }

private:
    mutable std::optional<std::string> type_name_;
};

class frequency_domain_state : public handled<zes_freq_state_t> {
public:
    using handled<zes_freq_state_t>::handled;

    double get_actual() const {
        return get_handle().actual;
    }
};

class frequency_domain : public handled<zes_freq_handle_t> {
public:
    using handled<zes_freq_handle_t>::handled;

    frequency_domain_properties get_properties() const {
        return get_or_init(properties_, [&]() {
            zes_freq_properties_t properties;
            check_ze(zesFrequencyGetProperties(get_handle(), &properties));
            return properties;
        });
    }

    frequency_domain_state get_state() const {
        zes_freq_state_t state;
        check_ze(zesFrequencyGetState(get_handle(), &state));
        return frequency_domain_state{ state };
    }

private:
    mutable std::optional<frequency_domain_properties> properties_;
};

class device_properties : public handled<ze_device_properties_t> {
public:
    using handled<ze_device_properties_t>::handled;

    std::string get_name() const {
        return get_or_init(name_, [&]() {
            return get_handle().name;
        });
    }

    std::string get_type_name() const {
        return get_or_init(type_name_, [&]() {
            switch (get_handle().type) {
                case ZE_DEVICE_TYPE_CPU: return "CPU";
                case ZE_DEVICE_TYPE_GPU: return "GPU";
                case ZE_DEVICE_TYPE_FPGA: return "FPGA";
                default: return "Unknown";
            }
        });
    }

    std::uint32_t get_id() const {
        return get_handle().deviceId;
    }

    std::uint32_t get_thread_count_per_unit() const {
        return get_handle().numThreadsPerEU;
    }

    std::uint32_t get_simd_width() const {
        return get_handle().physicalEUSimdWidth;
    }

    std::uint32_t get_unit_count_per_subslice() const {
        return get_handle().numEUsPerSubslice;
    }

    std::uint32_t get_subslices_count_per_slice() const {
        return get_handle().numSubslicesPerSlice;
    }

    std::uint32_t get_slice_count() const {
        return get_handle().numSlices;
    }

    std::uint32_t get_total_unit_count() const {
        return get_unit_count_per_subslice() * //
               get_subslices_count_per_slice() * //
               get_slice_count();
    }

    std::uint32_t get_float_mads_per_clock() const {
        return get_simd_width() * get_total_unit_count();
    }

    bool is_integrated() const {
        return bool(get_handle().flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED);
    }

private:
    mutable std::optional<std::string> name_;
    mutable std::optional<std::string> type_name_;
};

class device : public handled<ze_device_handle_t> {
public:
    using handled<ze_device_handle_t>::handled;

    device_properties get_properties() const {
        return get_or_init(properties_, [&]() {
            ze_device_properties_t handle{};
            handle.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            handle.pNext = nullptr;
            check_ze(zeDeviceGetProperties(get_handle(), &handle));
            return handle;
        });
    }

    std::vector<frequency_domain> get_frequency_domains() const {
        std::uint32_t domain_count = 0;
        check_ze(zesDeviceEnumFrequencyDomains(get_zes_handle(), &domain_count, nullptr));

        std::vector<zes_freq_handle_t> domain_handles(domain_count);
        check_ze(
            zesDeviceEnumFrequencyDomains(get_zes_handle(), &domain_count, domain_handles.data()));

        std::vector<frequency_domain> domains;
        domains.reserve(domain_handles.size());
        for (auto handle : domain_handles) {
            domains.emplace_back(handle);
        }
        return domains;
    }

private:
    zes_device_handle_t get_zes_handle() const {
        return static_cast<zes_device_handle_t>(get_handle());
    }

    mutable std::optional<device_properties> properties_;
};

class driver : public handled<ze_driver_handle_t> {
public:
    using handled<ze_driver_handle_t>::handled;

    std::vector<device> get_devices() const {
        std::uint32_t device_count = 0;
        check_ze(zeDeviceGet(get_handle(), &device_count, nullptr));

        std::vector<ze_device_handle_t> device_handles(device_count);
        check_ze(zeDeviceGet(get_handle(), &device_count, device_handles.data()));

        std::vector<device> devices;
        devices.reserve(device_handles.size());
        for (auto handle : device_handles) {
            devices.emplace_back(handle);
        }
        return devices;
    }
};

class system_manager {
public:
    explicit system_manager() {
        setenv("ZES_ENABLE_SYSMAN", "1", true);
        check_ze(zeInit(0));
    }

    std::vector<driver> get_drivers() const {
        std::uint32_t driver_count = 0;
        check_ze(zeDriverGet(&driver_count, nullptr));

        std::vector<ze_driver_handle_t> driver_handles(driver_count);
        check_ze(zeDriverGet(&driver_count, driver_handles.data()));

        std::vector<driver> drivers;
        drivers.reserve(driver_handles.size());
        for (auto handle : driver_handles) {
            drivers.emplace_back(handle);
        }
        return drivers;
    }

    std::vector<device> get_devices() const {
        std::vector<device> devices;
        for (auto driver : get_drivers()) {
            for (auto device : driver.get_devices()) {
                devices.push_back(device);
            }
        }
        return devices;
    }
};

} // namespace zexx
