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

#include "oneapi/dal/algo/linear_regression/backend/gpu/finalize_train_kernel_norm_eq_impl.hpp"
#include "oneapi/dal/algo/linear_regression/backend/gpu/misc.hpp"
#include "oneapi/dal/algo/linear_regression/backend/model_impl.hpp"

#include "oneapi/dal/backend/primitives/lapack.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::linear_regression::backend {

namespace be = dal::backend;
namespace pr = be::primitives;

using be::context_gpu;

template <typename Float, typename Task>
train_result<Task> finalize_train_kernel_norm_eq_impl<Float, Task>::operator()(
    const detail::descriptor_base<Task>& desc,
    const detail::train_parameters<Task>& params,
    const partial_train_result<Task>& input) {
    using dal::detail::check_mul_overflow;

    using model_t = model<Task>;
    using model_impl_t = detail::model_impl<Task>;

    const bool compute_intercept = desc.get_compute_intercept();

    constexpr auto uplo = pr::mkl::uplo::upper;
    constexpr auto alloc = sycl::usm::alloc::device;

    const auto response_count = input.get_partial_xty().get_row_count();
    const auto ext_feature_count = input.get_partial_xty().get_column_count();
    const auto feature_count = ext_feature_count - compute_intercept;

    const pr::ndshape<2> xtx_shape{ ext_feature_count, ext_feature_count };
    const pr::ndshape<2> betas_shape{ response_count, feature_count + 1 };

    auto xtx_nd = pr::table2ndarray<Float>(q, input.get_partial_xtx(), sycl::usm::alloc::device);
    auto xty_nd = pr::table2ndarray<Float, pr::ndorder::f>(q,
                                                           input.get_partial_xty(),
                                                           sycl::usm::alloc::device);

    const auto betas_size = check_mul_overflow(response_count, feature_count + 1);
    auto betas_arr = array<Float>::zeros(q, betas_size, alloc);

    if (comm_.get_rank_count() > 1) {
        auto xtx_nd_copy = pr::ndarray<Float, 2>::empty(q, xtx_shape, sycl::usm::alloc::device);
        auto copy_event = copy(q, xtx_nd_copy, xtx_nd, {});
        copy_event.wait_and_throw();
        xtx_nd = xtx_nd_copy;
        {
            ONEDAL_PROFILER_TASK(xtx_allreduce);
            auto xtx_arr =
                dal::array<Float>::wrap(q, xtx_nd.get_mutable_data(), xtx_nd.get_count());
            comm_.allreduce(xtx_arr).wait();
        }
        auto xty_nd_copy =
            pr::ndarray<Float, 2, pr::ndorder::f>::empty(q, betas_shape, sycl::usm::alloc::device);
        copy_event = copy(q, xty_nd_copy, xty_nd, {});
        copy_event.wait_and_throw();
        xty_nd = xty_nd_copy;
        {
            ONEDAL_PROFILER_TASK(xty_allreduce);
            auto xty_arr =
                dal::array<Float>::wrap(q, xty_nd.get_mutable_data(), xty_nd.get_count());
            comm_.allreduce(xty_arr).wait();
        }
    }

    double alpha = desc.get_alpha();
    sycl::event ridge_event;
    if (alpha != 0.0) {
        ridge_event = add_ridge_penalty<Float>(q, xtx_nd, compute_intercept, alpha);
    }

    auto nxtx = pr::ndarray<Float, 2>::empty(q, xtx_shape, alloc);
    auto nxty = pr::ndview<Float, 2>::wrap_mutable(betas_arr, betas_shape);
    auto solve_event =
        pr::solve_system<uplo>(q, compute_intercept, xtx_nd, xty_nd, nxtx, nxty, { ridge_event });
    sycl::event::wait_and_throw({ solve_event });

    auto betas = homogen_table::wrap(betas_arr, response_count, feature_count + 1);

    const auto model_impl = std::make_shared<model_impl_t>(betas);
    const auto model = dal::detail::make_private<model_t>(model_impl);

    const auto options = desc.get_result_options();
    auto result = train_result<Task>().set_model(model).set_result_options(options);

    if (options.test(result_options::intercept)) {
        auto arr = array<Float>::zeros(q, response_count, alloc);
        auto dst = pr::ndview<Float, 2>::wrap_mutable(arr, { 1l, response_count });
        const auto src = nxty.get_col_slice(0l, 1l).t();

        pr::copy(q, dst, src).wait_and_throw();

        auto intercept = homogen_table::wrap(arr, 1l, response_count);
        result.set_intercept(intercept);
    }

    if (options.test(result_options::coefficients)) {
        const auto size = check_mul_overflow(response_count, feature_count);

        auto arr = array<Float>::zeros(q, size, alloc);
        const auto src = nxty.get_col_slice(1l, feature_count + 1);
        auto dst = pr::ndview<Float, 2>::wrap_mutable(arr, { response_count, feature_count });

        pr::copy(q, dst, src).wait_and_throw();

        auto coefficients = homogen_table::wrap(arr, response_count, feature_count);
        result.set_coefficients(coefficients);
    }

    return result;
}

template class finalize_train_kernel_norm_eq_impl<float, task::regression>;
template class finalize_train_kernel_norm_eq_impl<double, task::regression>;

} // namespace oneapi::dal::linear_regression::backend

#endif // ONEDAL_DATA_PARALLEL
