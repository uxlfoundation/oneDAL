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

#include "oneapi/dal/algo/basic_statistics/backend/gpu/partial_compute_kernel.hpp"
#include "oneapi/dal/algo/basic_statistics/backend/gpu/partial_compute_kernel_misc.hpp"
#include "oneapi/dal/algo/basic_statistics/backend/gpu/compute_kernel.hpp"

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/util/common.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"

#include "oneapi/dal/algo/basic_statistics/backend/basic_statistics_interop.hpp"
namespace oneapi::dal::basic_statistics::backend {

namespace bk = dal::backend;
namespace pr = oneapi::dal::backend::primitives;

using alloc = sycl::usm::alloc;

using bk::context_gpu;
using task_t = task::compute;
using input_t = partial_compute_input<task_t>;
using result_t = partial_compute_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

template <typename Float, typename Task>
static partial_compute_result<Task> partial_compute(const context_gpu& ctx,
                                                    const descriptor_t& desc,
                                                    const partial_compute_input<Task>& input) {
    auto& q = ctx.get_queue();
    const auto data = input.get_data();
    const bool weights_enabling = input.get_weights().has_data();
    const auto weights = input.get_weights();

    auto kernel = compute_kernel_gpu<Float, method::dense, task::compute>{};
    auto compute_result_ = compute_result();
    // This local_desc marks the necessary results for the partial_compute step.
    auto local_desc = get_desc_to_compute<Float>(desc);
    const auto res_op = local_desc.get_result_options();

    auto result = partial_compute_result();
    const auto input_ = input.get_prev();

    const std::int64_t row_count = data.get_row_count();
    const std::int64_t column_count = data.get_column_count();
    const std::int64_t component_count = data.get_column_count();

    dal::detail::check_mul_overflow(row_count, column_count);
    dal::detail::check_mul_overflow(column_count, column_count);
    dal::detail::check_mul_overflow(component_count, column_count);

    const bool has_nobs_data = input_.get_partial_n_rows().has_data();
    if (has_nobs_data) {
        if (weights_enabling) {
            compute_result_ = kernel(ctx, local_desc, { data, weights });
        }
        else {
            compute_result_ = kernel(ctx, local_desc, { data });
        }
        const auto nobs_nd = pr::table2ndarray_1d<Float>(q, input_.get_partial_n_rows());
        auto [result_nobs, nobs_update_event] =
            update_partial_n_rows_results(q, row_count, nobs_nd);

        if (res_op.test(result_options::min) || res_op.test(result_options::max)) {
            const auto min_nd =
                pr::table2ndarray_1d<Float>(q, input_.get_partial_min(), sycl::usm::alloc::device);
            const auto max_nd =
                pr::table2ndarray_1d<Float>(q, input_.get_partial_max(), sycl::usm::alloc::device);
            const auto cur_min_nd =
                pr::table2ndarray_1d<Float>(q, compute_result_.get_min(), sycl::usm::alloc::device);
            const auto cur_max_nd =
                pr::table2ndarray_1d<Float>(q, compute_result_.get_max(), sycl::usm::alloc::device);
            auto [result_min, result_max, update_min_max_event] =
                update_min_max_results(q,
                                       min_nd,
                                       cur_min_nd,
                                       max_nd,
                                       cur_max_nd,
                                       column_count,
                                       { nobs_update_event });

            result.set_partial_min(
                (homogen_table::wrap(result_min.flatten(q, { update_min_max_event }),
                                     1,
                                     column_count)));

            result.set_partial_max(
                (homogen_table::wrap(result_max.flatten(q, { update_min_max_event }),
                                     1,
                                     column_count)));
        }

        if (res_op.test(result_options::sum)) {
            const auto sums_nd =
                pr::table2ndarray_1d<Float>(q, input_.get_partial_sum(), sycl::usm::alloc::device);
            const auto sums2_nd = pr::table2ndarray_1d<Float>(q,
                                                              input_.get_partial_sum_squares(),
                                                              sycl::usm::alloc::device);
            const auto cur_sums_nd =
                pr::table2ndarray_1d<Float>(q, compute_result_.get_sum(), sycl::usm::alloc::device);
            const auto cur_sums2_nd = pr::table2ndarray_1d<Float>(q,
                                                                  compute_result_.get_sum_squares(),
                                                                  sycl::usm::alloc::device);
            auto [result_sums, result_sums2, result_sums2cent, merge_sums_event] =
                update_partial_sums(q,
                                    sums_nd,
                                    cur_sums_nd,
                                    sums2_nd,
                                    cur_sums2_nd,
                                    column_count,
                                    result_nobs,
                                    { nobs_update_event });

            result.set_partial_sum(
                (homogen_table::wrap(result_sums.flatten(q, { merge_sums_event }),
                                     1,
                                     column_count)));

            result.set_partial_sum_squares(
                (homogen_table::wrap(result_sums2.flatten(q, { merge_sums_event }),
                                     1,
                                     column_count)));

            result.set_partial_sum_squares_centered(
                (homogen_table::wrap(result_sums2cent.flatten(q, { merge_sums_event }),
                                     1,
                                     column_count)));
        }

        result.set_partial_n_rows(
            (homogen_table::wrap(result_nobs.flatten(q, { nobs_update_event }), 1, 1)));
    }
    else {
        auto [init_nobs, init_event] =
            pr::ndarray<Float, 1>::full(q, { 1 }, row_count, sycl::usm::alloc::device);
        init_event.wait_and_throw();

        if (weights_enabling) {
            compute_result_ = kernel(ctx, local_desc, { data, weights });
        }
        else {
            compute_result_ = kernel(ctx, local_desc, { data });
        }

        if (res_op.test(result_options::min)) {
            result.set_partial_min(compute_result_.get_min());
        }

        if (res_op.test(result_options::max)) {
            result.set_partial_max(compute_result_.get_max());
        }

        if (res_op.test(result_options::sum)) {
            result.set_partial_sum(compute_result_.get_sum());
        }

        if (res_op.test(result_options::sum_squares)) {
            result.set_partial_sum_squares(compute_result_.get_sum_squares());
        }

        if (res_op.test(result_options::sum_squares_centered))
            result.set_partial_sum_squares_centered(compute_result_.get_sum_squares_centered());

        result.set_partial_n_rows((homogen_table::wrap(init_nobs.flatten(q, {}), 1, 1)));
    }

    return result;
}

template <typename Float>
struct partial_compute_kernel_gpu<Float, method::dense, task::compute> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return partial_compute<Float, task::compute>(ctx, desc, input);
    }
};

template struct partial_compute_kernel_gpu<float, method::dense, task::compute>;
template struct partial_compute_kernel_gpu<double, method::dense, task::compute>;

} // namespace oneapi::dal::basic_statistics::backend
