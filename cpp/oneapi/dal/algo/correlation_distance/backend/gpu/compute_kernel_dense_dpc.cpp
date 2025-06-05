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

#include "oneapi/dal/algo/correlation_distance/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/backend/primitives/distance.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/math.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/detail/profiler.hpp"

namespace oneapi::dal::correlation_distance::backend {

using dal::backend::context_gpu;
using input_t = compute_input<task::compute>;
using result_t = compute_result<task::compute>;
using descriptor_t = detail::descriptor_base<task::compute>;

namespace pr = dal::backend::primitives;

template <typename Float>
static result_t compute(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    ONEDAL_PROFILER_TASK(correlation_distance.compute, ctx.get_queue());

    auto& queue = ctx.get_queue();
    const auto x = input.get_x();
    const auto y = input.get_y();

    const std::int64_t x_row_count = x.get_row_count();
    const std::int64_t y_row_count = y.get_row_count();

    ONEDAL_ASSERT(x.get_column_count() == y.get_column_count());
    dal::detail::check_mul_overflow(x_row_count, y_row_count);

    // Convert tables to ndarray format
    const auto x_nd = pr::table2ndarray<Float>(queue, x, sycl::usm::alloc::device);
    const auto y_nd = pr::table2ndarray<Float>(queue, y, sycl::usm::alloc::device);

    // Create result array
    auto res_nd =
        pr::ndarray<Float, 2>::empty(queue, { x_row_count, y_row_count }, sycl::usm::alloc::device);

    // Use correlation distance primitive
    pr::correlation_distance<Float> distance(queue);
    distance(x_nd, y_nd, res_nd);

    // Convert result back to table format
    const auto res_array = res_nd.flatten(queue);
    auto res_table = homogen_table::wrap(res_array, x_row_count, y_row_count);

    return result_t{}.set_values(res_table);
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
                    const table& x,
                    const table& y,
                    homogen_table& res) {
        ONEDAL_ASSERT(x.get_row_count() == res.get_row_count());
        ONEDAL_ASSERT(y.get_row_count() == res.get_column_count());
        ONEDAL_ASSERT(x.get_column_count() == y.get_column_count());

        auto& queue = ctx.get_queue();
        const auto x_nd = pr::table2ndarray<Float>(queue, x, sycl::usm::alloc::device);
        const auto y_nd = pr::table2ndarray<Float>(queue, y, sycl::usm::alloc::device);

        auto res_ptr = res.get_data<Float>();

        // Temporary workaround until the table_builder approach is ready
        auto res_nd = pr::ndarray<Float, 2>::wrap(const_cast<Float*>(res_ptr),
                                                  { res.get_row_count(), res.get_column_count() });

        // Use correlation distance primitive
        pr::correlation_distance<Float> distance(queue);
        distance(x_nd, y_nd, res_nd);
    }
#endif
};

template struct compute_kernel_gpu<float, method::dense, task::compute>;
template struct compute_kernel_gpu<double, method::dense, task::compute>;

} // namespace oneapi::dal::correlation_distance::backend
