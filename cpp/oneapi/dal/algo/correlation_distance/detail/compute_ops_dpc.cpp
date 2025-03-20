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

#include "oneapi/dal/algo/correlation_distance/backend/cpu/compute_kernel.hpp"
#include "oneapi/dal/algo/correlation_distance/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/algo/correlation_distance/detail/compute_ops.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::correlation_distance::detail {
namespace v1 {

using dal::detail::data_parallel_policy;

template <typename Float, typename Method, typename Task>
struct compute_ops_dispatcher<data_parallel_policy, Float, Method, Task> {
    compute_result<Task> operator()(const data_parallel_policy& ctx,
                                    const descriptor_base<Task>& params,
                                    const compute_input<Task>& input) const {
        using kernel_dispatcher_t = dal::backend::kernel_dispatcher<
            KERNEL_SINGLE_NODE_CPU(backend::compute_kernel_cpu<Float, Method, Task>),
            KERNEL_SINGLE_NODE_GPU(backend::compute_kernel_gpu<Float, Method, Task>)>;
        return kernel_dispatcher_t{}(ctx, params, input);
    }

#ifdef ONEDAL_DATA_PARALLEL
    void operator()(const data_parallel_policy& ctx,
                    const descriptor_base<Task>& params,
                    const table& x,
                    const table& y,
                    homogen_table& res) {
        using kernel_dispatcher_t = dal::backend::kernel_dispatcher<
            KERNEL_SINGLE_NODE_CPU(backend::compute_kernel_cpu<Float, Method, Task>),
            KERNEL_SINGLE_NODE_GPU(backend::compute_kernel_gpu<Float, Method, Task>)>;
        kernel_dispatcher_t{}(ctx, params, x, y, res);
    }
#endif
};

#define INSTANTIATE(F, M, T) \
    template struct ONEDAL_EXPORT compute_ops_dispatcher<data_parallel_policy, F, M, T>;

INSTANTIATE(float, method::dense, task::compute)
INSTANTIATE(double, method::dense, task::compute)

} // namespace v1
} // namespace oneapi::dal::correlation_distance::detail
