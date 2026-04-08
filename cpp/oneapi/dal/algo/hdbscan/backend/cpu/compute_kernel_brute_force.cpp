/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "oneapi/dal/algo/hdbscan/backend/cpu/compute_kernel.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::hdbscan::backend {

using dal::backend::context_cpu;

using descriptor_t = detail::descriptor_base<task::clustering>;
using result_t = compute_result<task::clustering>;
using input_t = compute_input<task::clustering>;

template <typename Float>
struct compute_kernel_cpu<Float, method::brute_force, task::clustering> {
    result_t operator()(const context_cpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        throw unimplemented(
            dal::detail::error_messages::algorithm_is_not_implemented_for_this_device());
    }
};

template struct compute_kernel_cpu<float, method::brute_force, task::clustering>;
template struct compute_kernel_cpu<double, method::brute_force, task::clustering>;

} // namespace oneapi::dal::hdbscan::backend
