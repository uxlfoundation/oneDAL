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

#include "oneapi/dal/algo/basic_statistics/backend/gpu/finalize_compute_kernel.hpp"
#include "oneapi/dal/algo/basic_statistics/backend/gpu/finalize_compute_kernel_dense_impl.hpp"

namespace oneapi::dal::basic_statistics::backend {

namespace bk = dal::backend;
using bk::context_gpu;
using task_t = task::compute;
using input_t = partial_compute_result<task_t>;
using result_t = compute_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

/// CSR finalize on GPU: the CSR partial_compute writes the same six fields
/// (partial_n_rows, partial_min, partial_max, partial_sum, partial_sum_squares,
/// partial_sum_squares_centered) that the dense partial_compute does, so the
/// post-processing step (allreduce + mean / variance / stddev / variation /
/// second_order_raw_moment / recompute sum_squares_centered) is identical.
/// Delegate to the dense finalize impl.
template <typename Float>
struct finalize_compute_kernel_gpu<Float, method::sparse, task::compute> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return finalize_compute_kernel_dense_impl<Float>(ctx)(desc, input);
    }
};

template struct finalize_compute_kernel_gpu<float, method::sparse, task::compute>;
template struct finalize_compute_kernel_gpu<double, method::sparse, task::compute>;

} // namespace oneapi::dal::basic_statistics::backend
