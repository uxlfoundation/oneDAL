/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"

#include "oneapi/dal/algo/linear_regression/common.hpp"
#include "oneapi/dal/algo/linear_regression/train_types.hpp"

#include "oneapi/dal/algo/linear_regression/backend/gpu/partial_train_kernel.hpp"
#include "oneapi/dal/algo/linear_regression/backend/gpu/update_kernel.hpp"

namespace oneapi::dal::linear_regression::backend {

using dal::backend::context_gpu;

namespace be = dal::backend;
namespace pr = be::primitives;

template <typename Float, typename Task>
static partial_train_result<Task> call_dal_kernel(const context_gpu& ctx,
                                                  const detail::descriptor_base<Task>& desc,
                                                  const detail::train_parameters<Task>& params,
                                                  const partial_train_input<Task>& input) {
    auto result = partial_train_result<Task>();
    using dal::detail::check_mul_overflow;

    auto& queue = ctx.get_queue();

    constexpr auto alloc = sycl::usm::alloc::device;

    const bool compute_intercept = desc.get_compute_intercept();

    const auto feature_count = input.get_data().get_column_count();
    const auto response_count = input.get_responses().get_column_count();
    const std::int64_t ext_feature_count = feature_count + compute_intercept;

    const pr::ndshape<2> xty_shape{ response_count, ext_feature_count };
    const pr::ndshape<2> xtx_shape{ ext_feature_count, ext_feature_count };

    const auto input_ = input.get_prev();

    const bool has_xtx_data = input_.get_partial_xtx().has_data();
    if (has_xtx_data) {
        const auto data_nd =
            pr::table2ndarray<Float>(queue, input.get_data(), sycl::usm::alloc::device);
        const auto res_nd =
            pr::table2ndarray<Float>(queue, input.get_responses(), sycl::usm::alloc::device);

        auto xtx_nd =
            pr::table2ndarray<Float>(queue, input_.get_partial_xtx(), sycl::usm::alloc::device);
        auto [xtx, fill_xtx_event] =
            pr::ndarray<Float, 2, pr::ndorder::c>::zeros(queue, xtx_shape, alloc);
        auto copy_xtx_event = copy(queue, xtx, xtx_nd, { fill_xtx_event });
        auto [xty, fill_xty_event] =
            pr::ndarray<Float, 2, pr::ndorder::f>::zeros(queue, xty_shape, alloc);
        auto xty_nd = pr::table2ndarray<Float, pr::ndorder::f>(queue,
                                                               input_.get_partial_xty(),
                                                               sycl::usm::alloc::device);
        auto copy_xty_event = copy(queue, xty, xty_nd, { fill_xty_event });
        auto last_xtx_event =
            update_xtx(queue, compute_intercept, data_nd, xtx, { copy_xtx_event });
        auto last_xty_event =
            update_xty(queue, compute_intercept, data_nd, res_nd, xty, { copy_xty_event });

        result.set_partial_xtx(homogen_table::wrap(xtx.flatten(queue, { last_xtx_event }),
                                                   ext_feature_count,
                                                   ext_feature_count));
        result.set_partial_xty(homogen_table::wrap(xty.flatten(queue, { last_xty_event }),
                                                   response_count,
                                                   ext_feature_count,
                                                   data_layout::column_major));

        return result;
    }
    else {
        const auto data_nd =
            pr::table2ndarray<Float>(queue, input.get_data(), sycl::usm::alloc::device);
        const auto res_nd =
            pr::table2ndarray<Float>(queue, input.get_responses(), sycl::usm::alloc::device);
        auto [xty, fill_xty_event] =
            pr::ndarray<Float, 2, pr::ndorder::f>::zeros(queue, xty_shape, alloc);
        auto [xtx, fill_xtx_event] =
            pr::ndarray<Float, 2, pr::ndorder::c>::zeros(queue, xtx_shape, alloc);

        auto last_xty_event =
            update_xty(queue, compute_intercept, data_nd, res_nd, xty, { fill_xty_event });
        auto last_xtx_event =
            update_xtx(queue, compute_intercept, data_nd, xtx, { fill_xtx_event });

        result.set_partial_xtx(homogen_table::wrap(xtx.flatten(queue, { last_xtx_event }),
                                                   ext_feature_count,
                                                   ext_feature_count));
        result.set_partial_xty(homogen_table::wrap(xty.flatten(queue, { last_xty_event }),
                                                   response_count,
                                                   ext_feature_count,
                                                   data_layout::column_major));

        return result;
    }
}

template <typename Float, typename Task>
static partial_train_result<Task> train(const context_gpu& ctx,
                                        const detail::descriptor_base<Task>& desc,
                                        const detail::train_parameters<Task>& params,
                                        const partial_train_input<Task>& input) {
    return call_dal_kernel<Float, Task>(ctx, desc, params, input);
}

template <typename Float, typename Task>
struct partial_train_kernel_gpu<Float, method::norm_eq, Task> {
    partial_train_result<Task> operator()(const context_gpu& ctx,
                                          const detail::descriptor_base<Task>& desc,
                                          const detail::train_parameters<Task>& params,
                                          const partial_train_input<Task>& input) const {
        return train<Float, Task>(ctx, desc, params, input);
    }
};

template struct partial_train_kernel_gpu<float, method::norm_eq, task::regression>;
template struct partial_train_kernel_gpu<double, method::norm_eq, task::regression>;

} // namespace oneapi::dal::linear_regression::backend
