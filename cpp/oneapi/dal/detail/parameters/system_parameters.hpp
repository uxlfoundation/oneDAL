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

#pragma once

#include <string>

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/parameters/system_parameters_impl.hpp"

namespace oneapi::dal {

namespace detail {

/// Stores system-related parameters that affect the performance of the algorithms.
/// Those parameters can differ from the `get_global_context().get_cpu_info()`.
///
/// `cpu_info` reports the parameters available in hardware, where `system_parameters`
/// are the software-enabled parameters that can differ from `cpu_info`.
class ONEDAL_EXPORT system_parameters : public base {
public:
    /// Creates a new default `system_parameters` instance.
    explicit system_parameters();

    /// Host related parameters.

    /// Top enabled CPU instruction set.
    cpu_extension get_top_enabled_cpu_extension() const;

    /// Maximal number of threads available to the algorithm.
    std::uint32_t get_max_number_of_threads() const;

#ifdef ONEDAL_DATA_PARALLEL
    /// Device related parameters.

    /// Maximal SYCL workgroup size on the device.
    ///
    /// @param queue                  The SYCL* queue object
    std::uint32_t get_max_workgroup_size(sycl::queue& queue) const;
#endif

    /// Logs host parameters in the format: name_1: value_1; ... ; name_N: value_N.
    std::string dump() const;

#ifdef ONEDAL_DATA_PARALLEL
    /// Logs host and device parameters in the format: name_1: value_1; ... ; name_N: value_N.
    ///
    /// @param queue                  The SYCL* queue object
    std::string dump(sycl::queue& queue) const;
#endif

private:
    detail::pimpl<system_parameters_impl> impl_;
};

} // namespace detail
} // namespace oneapi::dal
