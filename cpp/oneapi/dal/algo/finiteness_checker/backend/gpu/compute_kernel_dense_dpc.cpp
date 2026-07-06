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

#include "oneapi/dal/algo/finiteness_checker/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/detail/profiler.hpp"

namespace oneapi::dal::finiteness_checker::backend {

using dal::backend::context_gpu;
using input_t = compute_input<task::compute>;
using result_t = compute_result<task::compute>;
using descriptor_t = detail::descriptor_base<task::compute>;

namespace pr = dal::backend::primitives;

template <typename Float>
bool compute_finiteness(sycl::queue& queue,
                        const pr::ndview<Float, 1>& data_1d,
                        bool allowNaN,
                        const dal::backend::event_vector& deps = {}) {
    Float out;

    if (allowNaN) {
        ONEDAL_PROFILER_TASK(finiteness_checker.reduce, queue);
        out = pr::reduce_1d(queue, data_1d, pr::logical_or<Float>{}, pr::isinf<Float>{}, deps);
    }
    else {
        ONEDAL_PROFILER_TASK(finiteness_checker.reduce, queue);
        out = pr::reduce_1d(queue, data_1d, pr::logical_or<Float>{}, pr::isinfornan<Float>{}, deps);
    }
    // invert out to match daal implementation (assert result is finite)
    return !static_cast<bool>(out);
}

template <typename Float>
static result_t compute(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    auto& queue = ctx.get_queue();
    const auto data = input.get_data();
    const auto data_1d = pr::flatten_table_by_layout<Float>(queue, data, sycl::usm::alloc::device);
    return result_t{}.set_finite(compute_finiteness(queue, data_1d, desc.get_allow_NaN()));
}

template <typename Float>
struct compute_kernel_gpu<Float, method::dense, task::compute> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute<Float>(ctx, desc, input);
    }

#ifdef ONEDAL_DATA_PARALLEL
    void operator()(const context_gpu& ctx,
                    const descriptor_t& desc,
                    const table& data,
                    bool& res) {
        auto& queue = ctx.get_queue();
        const auto data_1d =
            pr::flatten_table_by_layout<Float>(queue, data, sycl::usm::alloc::device);
        res = compute_finiteness(queue, data_1d, desc.get_allow_NaN());
    }
#endif
};

template struct compute_kernel_gpu<float, method::dense, task::compute>;
template struct compute_kernel_gpu<double, method::dense, task::compute>;

} // namespace oneapi::dal::finiteness_checker::backend
