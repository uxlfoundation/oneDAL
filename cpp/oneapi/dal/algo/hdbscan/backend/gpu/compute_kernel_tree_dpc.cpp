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

/// GPU HDBSCAN kd_tree and ball_tree. Both run the same exact GPU search, with alpha scaling only
/// the pairwise distance.

#include "oneapi/dal/algo/hdbscan/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/algo/hdbscan/backend/gpu/compute_pipeline.hpp"

namespace oneapi::dal::hdbscan::backend {

using dal::backend::context_gpu;

using descriptor_t = detail::descriptor_base<task::clustering>;
using result_t = compute_result<task::clustering>;
using input_t = compute_input<task::clustering>;

template <typename Float>
struct compute_kernel_gpu<Float, method::kd_tree, task::clustering> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute_gpu<Float>(ctx.get_queue(), desc, input.get_data(), false);
    }
};

template <typename Float>
struct compute_kernel_gpu<Float, method::ball_tree, task::clustering> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute_gpu<Float>(ctx.get_queue(), desc, input.get_data(), false);
    }
};

template struct compute_kernel_gpu<float, method::kd_tree, task::clustering>;
template struct compute_kernel_gpu<double, method::kd_tree, task::clustering>;
template struct compute_kernel_gpu<float, method::ball_tree, task::clustering>;
template struct compute_kernel_gpu<double, method::ball_tree, task::clustering>;

} // namespace oneapi::dal::hdbscan::backend
