/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/primitives/blas/gemv.hpp"
#include "oneapi/dal/backend/primitives/element_wise.hpp"
#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/algo/linear_regression/common.hpp"
#include "oneapi/dal/algo/linear_regression/train_types.hpp"
#include "oneapi/dal/algo/linear_regression/backend/model_impl.hpp"
#include "oneapi/dal/algo/linear_regression/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/algo/linear_regression/backend/gpu/update_kernel.hpp"
#include "oneapi/dal/algo/linear_regression/backend/gpu/misc.hpp"

namespace oneapi::dal::linear_regression::backend {

using dal::backend::context_gpu;

namespace be = dal::backend;
namespace pr = be::primitives;

template <typename Float, typename Task>
static train_result<Task> make_train_result(sycl::queue& queue,
                                            const detail::descriptor_base<Task>& desc,
                                            const array<Float>& betas_arr,
                                            std::int64_t response_count,
                                            std::int64_t feature_count) {
    using dal::detail::check_mul_overflow;

    using model_t = model<Task>;
    using model_impl_t = detail::model_impl<Task>;

    constexpr auto alloc = sycl::usm::alloc::device;

    auto betas_table = homogen_table::wrap(betas_arr, response_count, feature_count + 1);

    const auto model_impl = std::make_shared<model_impl_t>(betas_table);
    const auto model = dal::detail::make_private<model_t>(model_impl);

    const auto options = desc.get_result_options();
    auto result = train_result<Task>().set_model(model).set_result_options(options);

    const auto betas =
        pr::ndview<Float, 2>::wrap(betas_arr.get_data(), { response_count, feature_count + 1 });

    if (options.test(result_options::intercept)) {
        auto arr = array<Float>::zeros(queue, response_count, alloc);
        auto dst = pr::ndview<Float, 2>::wrap_mutable(arr, { 1l, response_count });
        const auto src = betas.get_col_slice(0l, 1l).t();

        pr::copy(queue, dst, src).wait_and_throw();

        auto intercept = homogen_table::wrap(arr, 1l, response_count);
        result.set_intercept(intercept);
    }

    if (options.test(result_options::coefficients)) {
        const auto size = check_mul_overflow(response_count, feature_count);

        auto arr = array<Float>::zeros(queue, size, alloc);
        const auto src = betas.get_col_slice(1l, feature_count + 1);
        auto dst = pr::ndview<Float, 2>::wrap_mutable(arr, { response_count, feature_count });

        pr::copy(queue, dst, src).wait_and_throw();

        auto coefficients = homogen_table::wrap(arr, response_count, feature_count);
        result.set_coefficients(coefficients);
    }

    return result;
}

/// Solves the regression problem through the dual system: (Xc * Xc^t + alpha * I) * dualCoef = yc.
/// The resulting coefficients beta = Xc^t * dualCoef,
/// where Xc and yc are X and y centered when the intercept is computed
template <typename Float, typename Task>
static train_result<Task> call_dual_kernel(const context_gpu& ctx,
                                           const detail::descriptor_base<Task>& desc,
                                           const pr::ndview<Float, 2>& x,
                                           const pr::ndview<Float, 2>& y) {
    using dal::detail::check_mul_overflow;

    auto& queue = ctx.get_queue();

    ONEDAL_PROFILER_TASK(linreg_train_dual_kernel, queue);

    constexpr auto uplo = pr::mkl::uplo::upper;
    constexpr auto alloc = sycl::usm::alloc::device;
    constexpr pr::sum<Float> plus;
    constexpr pr::identity<Float> identity;
    constexpr std::minus<Float> minus;
    constexpr std::multiplies<Float> multiplies;

    const auto sample_count = x.get_dimension(0);
    const auto feature_count = x.get_dimension(1);
    const auto response_count = y.get_dimension(1);
    ONEDAL_ASSERT(sample_count == y.get_dimension(0));
    const bool compute_intercept = desc.get_compute_intercept();

    const auto betas_size = check_mul_overflow(response_count, feature_count + 1);
    auto betas_arr = array<Float>::zeros(queue, betas_size, alloc);
    auto betas =
        pr::ndview<Float, 2>::wrap_mutable(betas_arr, { response_count, feature_count + 1 });
    auto intercept = betas.get_col_slice(0l, 1l);
    auto coefs = betas.get_col_slice(1l, feature_count + 1);

    if (compute_intercept && sample_count == 1) {
        // The intercept fits the only observation exactly, the coefficients are zero
        pr::copy(queue, intercept, y.t()).wait_and_throw();
        return make_train_result<Float, Task>(queue,
                                              desc,
                                              betas_arr,
                                              response_count,
                                              feature_count);
    }

    array<Float> x_means_arr, y_means_arr;
    pr::ndview<Float, 1> x_means, y_means;
    pr::ndarray<Float, 2> xc_arr, yc_arr;
    pr::ndview<Float, 2> xc = x, yc = y;
    sycl::event center_x_event, center_y_event;
    if (compute_intercept) {
        ONEDAL_PROFILER_TASK(center_data, queue);
        const Float inv_sample_count = Float(1) / Float(sample_count);

        x_means_arr = array<Float>::empty(queue, feature_count, alloc);
        y_means_arr = array<Float>::empty(queue, response_count, alloc);
        x_means = pr::ndview<Float, 1>::wrap_mutable(x_means_arr);
        y_means = pr::ndview<Float, 1>::wrap_mutable(y_means_arr);
        auto x_sums_event = pr::reduce_by_columns(queue, x, x_means, plus, identity);
        auto y_sums_event = pr::reduce_by_columns(queue, y, y_means, plus, identity);
        auto x_means_event = pr::element_wise(queue,
                                              multiplies,
                                              x_means,
                                              inv_sample_count,
                                              x_means,
                                              { x_sums_event });
        auto y_means_event = pr::element_wise(queue,
                                              multiplies,
                                              y_means,
                                              inv_sample_count,
                                              y_means,
                                              { y_sums_event });

        xc_arr = pr::ndarray<Float, 2>::empty(queue, x.get_shape(), alloc);
        yc_arr = pr::ndarray<Float, 2>::empty(queue, y.get_shape(), alloc);
        center_x_event = pr::element_wise(queue, minus, x, x_means, xc_arr, { x_means_event });
        center_y_event = pr::element_wise(queue, minus, y, y_means, yc_arr, { y_means_event });
        xc = xc_arr;
        yc = yc_arr;
    }

    const pr::ndshape<2> xxt_shape{ sample_count, sample_count };
    auto xxt = pr::ndarray<Float, 2>::empty(queue, xxt_shape, alloc);
    sycl::event xxt_event = pr::gemm(queue, xc, xc.t(), xxt, { center_x_event });

    const double alpha = desc.get_alpha();
    if (alpha != 0.0) {
        xxt_event = add_ridge_penalty<Float>(queue, xxt, false, alpha, { xxt_event });
    }

    // solve_system<uplo, false> prepends a zero column for the intercept, which the dual system does not have
    auto dual_ext =
        pr::ndarray<Float, 2>::empty(queue, { response_count, sample_count + 1 }, alloc);
    auto dual = dual_ext.get_col_slice(1l, sample_count + 1);
    pr::ndview<Float, 2> unused_xxt;
    // The rounding noise in the eigenvalues of Xc * Xc^t grows with its size.
    // Unpenalized centered Xc * Xc^t is singular, so Cholesky is skipped.
    const bool skip_cholesky = compute_intercept && alpha == 0.0;
    auto solve_event = pr::solve_system<uplo, false>(queue,
                                                     xxt,
                                                     yc.t(),
                                                     unused_xxt,
                                                     dual_ext,
                                                     Float(sample_count),
                                                     skip_cholesky,
                                                     { xxt_event, center_y_event });

    auto coefs_event = pr::gemm(queue, dual, xc, coefs, { solve_event });

    if (compute_intercept) {
        // intercept = y_means - coefs * x_means
        auto intercept_event =
            pr::gemv(queue, coefs, x_means, y_means, Float(-1), Float(1), { coefs_event });
        const auto y_means_2d = y_means.template reshape<2>({ response_count, 1l });
        pr::copy(queue, intercept, y_means_2d, { intercept_event }).wait_and_throw();
    }
    else {
        coefs_event.wait_and_throw();
    }

    return make_train_result<Float, Task>(queue, desc, betas_arr, response_count, feature_count);
}

template <typename Float, typename Task>
static train_result<Task> call_dal_kernel(const context_gpu& ctx,
                                          const detail::descriptor_base<Task>& desc,
                                          const detail::train_parameters<Task>& params,
                                          const table& data,
                                          const table& resp) {
    using dal::detail::check_mul_overflow;

    auto& queue = ctx.get_queue();

    ONEDAL_PROFILER_TASK(linreg_train_kernel, queue);

    constexpr auto uplo = pr::mkl::uplo::upper;
    constexpr auto alloc = sycl::usm::alloc::device;

    row_accessor<const Float> x_accessor(data);
    row_accessor<const Float> y_accessor(resp);

    const auto sample_count = data.get_row_count();
    const auto feature_count = data.get_column_count();
    const auto response_count = resp.get_column_count();
    ONEDAL_ASSERT(sample_count == resp.get_row_count());
    const bool compute_intercept = desc.get_compute_intercept();
    const std::int64_t ext_feature_count = feature_count + compute_intercept;

    const auto betas_size = check_mul_overflow(response_count, feature_count + 1);
    auto betas_arr = array<Float>::zeros(queue, betas_size, alloc);

    const std::int64_t b_count = params.get_gpu_macro_block();
    const be::uniform_blocking blocking(sample_count, b_count);

    const pr::ndshape<2> xty_shape{ response_count, ext_feature_count };
    const pr::ndshape<2> betas_shape{ response_count, feature_count + 1 };
    const pr::ndshape<2> xtx_shape{ ext_feature_count, ext_feature_count };

    auto [xty, fill_xty_event] =
        pr::ndarray<Float, 2, pr::ndorder::f>::zeros(queue, xty_shape, alloc);
    auto [xtx, fill_xtx_event] =
        pr::ndarray<Float, 2, pr::ndorder::c>::zeros(queue, xtx_shape, alloc);
    sycl::event last_xty_event = fill_xty_event, last_xtx_event = fill_xtx_event;

    array<Float> old_x_arr{}, old_y_arr{};
    for (std::int64_t b = 0; b < blocking.get_block_count(); ++b) {
        const auto last = blocking.get_block_end_index(b);
        const auto first = blocking.get_block_start_index(b);

        ONEDAL_ASSERT(first < last);
        const std::int64_t length = last - first;

        auto x_arr = x_accessor.pull(queue, { first, last }, alloc);
        auto x = pr::ndview<Float, 2>::wrap(x_arr.get_data(), { length, feature_count });

        auto y_arr = y_accessor.pull(queue, { first, last }, alloc);
        auto y = pr::ndview<Float, 2>::wrap(y_arr.get_data(), { length, response_count });

        last_xty_event = update_xty(queue, compute_intercept, x, y, xty, { last_xty_event });
        last_xtx_event = update_xtx(queue, compute_intercept, x, xtx, { last_xtx_event });

        // We keep the latest slice of data up to date because of pimpl -
        // it virtually extend lifetime of pulled arrays
        old_x_arr = std::move(x_arr), old_y_arr = std::move(y_arr);
    }

    auto& comm = ctx.get_communicator();
    if (comm.get_rank_count() > 1) {
        sycl::event::wait_and_throw({ last_xty_event, last_xtx_event });
        {
            ONEDAL_PROFILER_TASK(xtx_allreduce);
            auto xtx_arr = dal::array<Float>::wrap(queue, xtx.get_mutable_data(), xtx.get_count());
            comm.allreduce(xtx_arr).wait();
        }
        {
            ONEDAL_PROFILER_TASK(xty_allreduce);
            auto xty_arr = dal::array<Float>::wrap(queue, xty.get_mutable_data(), xty.get_count());
            comm.allreduce(xty_arr).wait();
        }
    }

    double alpha = desc.get_alpha();
    if (alpha != 0.0) {
        last_xtx_event =
            add_ridge_penalty<Float>(queue, xtx, compute_intercept, alpha, { last_xtx_event });
    }
    const be::event_vector solve_deps{ last_xty_event, last_xtx_event };

    auto nxtx = pr::ndarray<Float, 2>::empty(queue, xtx_shape, alloc);
    auto nxty = pr::ndview<Float, 2>::wrap_mutable(betas_arr, betas_shape);
    auto solve_event =
        pr::solve_system<uplo>(queue, compute_intercept, xtx, xty, nxtx, nxty, solve_deps);
    sycl::event::wait_and_throw({ solve_event });

    return make_train_result<Float, Task>(queue, desc, betas_arr, response_count, feature_count);
}

template <typename Float, typename Task>
static train_result<Task> train(const context_gpu& ctx,
                                const detail::descriptor_base<Task>& desc,
                                const detail::train_parameters<Task>& params,
                                const train_input<Task>& input) {
    auto& queue = ctx.get_queue();
    constexpr auto alloc = sycl::usm::alloc::device;

    // table2ndarray() ensures the returned ndarray is row-major and accessible from the target device.
    // Depending on source alloc/layout, it may reuse the original storage or allocate/copy/transpose.
    const auto data_nd = pr::table2ndarray<Float, pr::ndorder::c>(queue, input.get_data(), alloc);
    const auto resp_nd =
        pr::table2ndarray<Float, pr::ndorder::c>(queue, input.get_responses(), alloc);

    const bool underdetermined = data_nd.get_dimension(0) < data_nd.get_dimension(1);
    if (underdetermined && ctx.get_communicator().get_rank_count() == 1) {
        return call_dual_kernel<Float, Task>(ctx, desc, data_nd, resp_nd);
    }

    auto data_table = homogen_table::wrap(queue,
                                          data_nd.get_data(),
                                          data_nd.get_dimension(0),
                                          data_nd.get_dimension(1));
    auto resp_table = homogen_table::wrap(queue,
                                          resp_nd.get_data(),
                                          resp_nd.get_dimension(0),
                                          resp_nd.get_dimension(1));

    return call_dal_kernel<Float, Task>(ctx, desc, params, data_table, resp_table);
}

template <typename Float, typename Task>
struct train_kernel_gpu<Float, method::norm_eq, Task> {
    train_result<Task> operator()(const context_gpu& ctx,
                                  const detail::descriptor_base<Task>& desc,
                                  const detail::train_parameters<Task>& params,
                                  const train_input<Task>& input) const {
        return train<Float, Task>(ctx, desc, params, input);
    }
};

template struct train_kernel_gpu<float, method::norm_eq, task::regression>;
template struct train_kernel_gpu<double, method::norm_eq, task::regression>;

} // namespace oneapi::dal::linear_regression::backend
