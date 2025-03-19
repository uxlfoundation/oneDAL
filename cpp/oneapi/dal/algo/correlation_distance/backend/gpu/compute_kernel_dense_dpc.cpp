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
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/math.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/backend/primitives/distance.hpp

namespace oneapi::dal::correlation_distance::backend {

using dal::backend::context_gpu;
using input_t = compute_input<task::compute>;
using result_t = compute_result<task::compute>;
using descriptor_t = detail::descriptor_base<task::compute>;

namespace pr = dal::backend::primitives;

template <typename Float>
void compute_coefficient(sycl::queue& queue,
                         const pr::ndview<Float, 1>& inv_norm_x_nd,
                         const pr::ndview<Float, 1>& inv_norm_y_nd,
                         pr::ndview<Float, 2>& res_nd,
                         const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(correlation_distance.compute_coefficient, queue);
    const std::int64_t x_row_count = inv_norm_x_nd.get_dimension(0);
    const std::int64_t y_row_count = inv_norm_y_nd.get_dimension(0);
    ONEDAL_ASSERT(res_nd.get_count() == x_row_count * y_row_count);

    const Float* inv_norm_x_ptr = inv_norm_x_nd.get_data();
    const Float* inv_norm_y_ptr = inv_norm_y_nd.get_data();
    Float* res_ptr = res_nd.get_mutable_data();
    const auto res_stride = res_nd.get_leading_stride();
    const auto range = dal::backend::make_range_2d(x_row_count, y_row_count);

    queue
        .submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.parallel_for(range, [=](sycl::id<2> idx) {
                auto& res_nd = *(res_ptr + res_stride * idx[0] + idx[1]);
                res_nd = 1 - res_nd * inv_norm_x_ptr[idx[0]] * inv_norm_y_ptr[idx[1]];
            });
        })
        .wait_and_throw();
}

template <typename Float>
void compute_correlation_distance(sycl::queue& queue,
                                  const pr::ndview<Float, 2>& x_nd,
                                  const pr::ndview<Float, 2>& y_nd,
                                  pr::ndview<Float, 2>& res_nd,
                                  const dal::backend::event_vector& deps = {}) {
    ONEDAL_ASSERT(x_nd.get_dimension(0) == res_nd.get_dimension(0));
    ONEDAL_ASSERT(y_nd.get_dimension(0) == res_nd.get_dimension(1));
    ONEDAL_ASSERT(x_nd.get_dimension(1) == y_nd.get_dimension(1));

    /*const auto x_row_count = x_nd.get_dimension(0);
    const auto y_row_count = y_nd.get_dimension(0);*/

    auto [dev_x_nd, dev_x_nd_event] =
        pr::distance<Float, correlation_metric<Float>>::get_deviation(x_nd, deps);
    auto [dev_y_nd, dev_y_nd_event] =
        pr::distance<Float, correlation_metric<Float>>::get_deviation(y_nd, deps);

    auto [inv_norm_x_nd, inv_norm_x_nd_event] =
        pr::distance<Float, correlation_metric<Float>>::get_inversed_norms(dev_x_nd,
                                                                           { dev_x_nd_event });
    auto [inv_norm_y_nd, inv_norm_y_nd_event] =
        pr::distance<Float, correlation_metric<Float>>::get_inversed_norms(dev_y_nd,
                                                                           { dev_y_nd_event });

    constexpr Float alpha = 1.0;
    constexpr Float beta = 0.0;
    sycl::event gemm_event;
    {
        ONEDAL_PROFILER_TASK(correlation_distance.gemm, queue);

        gemm_event = pr::gemm(queue,
                              dev_x_nd,
                              dev_y_nd.t(),
                              res_nd,
                              alpha,
                              beta,
                              { dev_x_nd_event, dev_y_nd_event });
    }
    compute_coefficient(queue, inv_norm_x_nd, inv_norm_y_nd, res_nd, { gemm_event });
}

template <typename Float>
static result_t compute(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    auto& queue = ctx.get_queue();
    const auto x = input.get_x();
    const auto y = input.get_y();

    const std::int64_t x_row_count = x.get_row_count();
    const std::int64_t y_row_count = y.get_row_count();

    ONEDAL_ASSERT(x.get_column_count() == y.get_column_count());
    dal::detail::check_mul_overflow(x_row_count, y_row_count);

    const auto x_nd = pr::table2ndarray<Float>(queue, x, sycl::usm::alloc::device);
    const auto y_nd = pr::table2ndarray<Float>(queue, y, sycl::usm::alloc::device);

    auto res_nd =
        pr::ndarray<Float, 2>::empty(queue, { x_row_count, y_row_count }, sycl::usm::alloc::device);

    compute_correlation_distance(queue, x_nd, y_nd, res_nd, desc.get_sigma());

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

        compute_correlation_distance(queue, x_nd, y_nd, res_nd, desc.get_sigma());
    }
#endif
};

template struct compute_kernel_gpu<float, method::dense, task::compute>;
template struct compute_kernel_gpu<double, method::dense, task::compute>;

} // namespace oneapi::dal::correlation_distance::backend
