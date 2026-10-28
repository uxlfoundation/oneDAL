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

#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/algo/hdbscan/compute_types.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::hdbscan::backend {

/// HDBSCAN GPU compute kernel functor, specialized in `compute_kernel_brute_force_dpc.cpp` and
/// `compute_kernel_tree_dpc.cpp`; every method runs `compute_gpu` from `compute_pipeline.hpp`.
///
/// @tparam Float  Floating-point type
/// @tparam Method Method tag (`method::brute_force`, `method::kd_tree`, `method::ball_tree`)
/// @tparam Task   Task tag (currently `task::clustering`)
template <typename Float, typename Method, typename Task>
struct compute_kernel_gpu {
    /// Run the HDBSCAN compute pipeline on the GPU.
    ///
    /// @param[in] ctx    GPU dispatch context (carries the SYCL queue)
    /// @param[in] params Algorithm descriptor
    /// @param[in] input  Compute input (carries the data table)
    ///
    /// @return Compute result with the requested outputs
    compute_result<Task> operator()(const dal::backend::context_gpu& ctx,
                                    const detail::descriptor_base<Task>& params,
                                    const compute_input<Task>& input) const;
};

} // namespace oneapi::dal::hdbscan::backend
