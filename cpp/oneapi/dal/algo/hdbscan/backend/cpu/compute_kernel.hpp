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

#include "oneapi/dal/algo/hdbscan/compute_types.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::hdbscan::backend {

/// HDBSCAN CPU compute kernel functor.
///
/// Specialized per (Float, Method, Task) tuple by the per-method TUs in
/// `compute_kernel_brute_force.cpp` / `compute_kernel_kd_tree.cpp` /
/// `compute_kernel_ball_tree.cpp`. Each specialization forwards to the DAAL
/// `HDBSCANBatchKernel` after converting the oneAPI descriptor and input.
///
/// @tparam Float  Floating-point type
/// @tparam Method Method tag (`method::brute_force`, `method::kd_tree`, `method::ball_tree`)
/// @tparam Task   Task tag (currently `task::clustering`)
template <typename Float, typename Method, typename Task>
struct compute_kernel_cpu {
    /// Run the HDBSCAN compute pipeline on the CPU.
    ///
    /// @param[in] ctx    CPU dispatch context
    /// @param[in] params Algorithm descriptor (carries metric, mcs, min_samples, etc.)
    /// @param[in] input  Compute input (carries the data table)
    ///
    /// @return Compute result containing the response and cluster-count tables
    compute_result<Task> operator()(const dal::backend::context_cpu& ctx,
                                    const detail::descriptor_base<Task>& params,
                                    const compute_input<Task>& input) const;
};

} // namespace oneapi::dal::hdbscan::backend
