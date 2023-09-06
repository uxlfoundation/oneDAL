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

#include "oneapi/dal/algo/basic_statistics/backend/gpu/finalize_compute_kernel.hpp"

// #include "oneapi/dal/backend/primitives/lapack.hpp"
// #include "oneapi/dal/backend/primitives/reduction.hpp"
// #include "oneapi/dal/backend/primitives/stat.hpp"
// #include "oneapi/dal/backend/primitives/utils.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::basic_statistics::backend {

namespace bk = dal::backend;
//namespace pr = oneapi::dal::backend::primitives;
using alloc = sycl::usm::alloc;

using bk::context_gpu;
using task_t = task::compute;
using input_t = partial_compute_result<task_t>;
using result_t = compute_result<task_t>;
using descriptor_t = detail::descriptor_base<task::compute>;


// template <typename Float>
// auto compute_all_metrics(sycl::queue& q,
//                       const pr::ndview<Float, 2>& data,
//                       std::int64_t column_count,
//                       std::int64_t row_count,
//                       const dal::backend::event_vector& deps = {}) {
//     ONEDAL_PROFILER_TASK(init_partial_results, q);

//         const Float inv_n = Float(1.0 / double(rows_count_global));

//         last_event = q_.submit([&](sycl::handler& cgh) {
//             cgh.depends_on(deps);
//             cgh.parallel_for(range, [=](sycl::id<1> id) {
//                 Float mrgsum2cent = bsum2cent_ptr[id];
//                 Float mrgmean = bsum_ptr[id] * inv_n;

//                 Float local_mean = bmean_ptr[id];

//                 Float delta = mrgmean - local_mean;

//                 mrgsum2cent += delta * delta * row_count;
//                 if constexpr (check_mask_flag(bs_list::mean, List)) {
//                     rmean_ptr[id] = mrgmean;
//                 }
//                 if constexpr (check_mask_flag(bs_list::sum2cent, List)) {
//                     rsum2cent_ptr[id] = mrgsum2cent;
//                 }
//                 if constexpr (check_mask_flag(bs_list::varc, List)) {
//                     rvarc_ptr[id] = mrgsum2cent / (rows_count_global - 1);
//                 }
//                 if constexpr (check_mask_flag(bs_list::sorm, List)) {
//                     rsorm_ptr[id] = bsum2_ptr[id] * inv_n;
//                 }
//             });
//         });


//     return std::make_tuple(result_min,
//                            result_max,
//                            result_sums,
//                            result_sums2,
//                            result_sums2cent,
//                            result_nobs,
//                            update_event);
// }

template <typename Float, typename Task>
static compute_result<Task> finalize_compute(const context_gpu& ctx,
                                             const descriptor_t& desc,
                                             const partial_compute_result<Task>& input) {
    //auto& q_ = ctx.get_queue();
    result_t res;

    // auto column_count = input.get_partial_sums_squares().get_column_count();
    // ONEDAL_ASSERT(column_count > 0);
    sycl::event event;
    const auto res_op = desc.get_result_options();
    res.set_result_options(desc.get_result_options());

    if (res_op.test(result_options::min)) {
        // ONEDAL_ASSERT(input.get_min().get_count() == column_count);
        res.set_min(input.get_partial_min());
    }
    if (res_op.test(result_options::max)) {
        // ONEDAL_ASSERT(input.get_max().get_count() == column_count);
        res.set_max(input.get_partial_max());
    }
    if (res_op.test(result_options::sum)) {
        // ONEDAL_ASSERT(input.get_sum().get_count() == column_count);
        res.set_sum(input.get_partial_sums());
    }
    if (res_op.test(result_options::sum_squares)) {
        // ONEDAL_ASSERT(input.get_sum2().get_count() == column_count);
        res.set_sum_squares(input.get_partial_sums_squares());
    }

    if (res_op.test(result_options::sum_squares_centered)) {
        // ONEDAL_ASSERT(ndres.get_sum2cent().get_count() == column_count);
        res.set_sum_squares_centered(
            input.get_partial_sums_squares_centered());
    }
    // if (res_op.test(result_options::mean)) {
    //     ONEDAL_ASSERT(ndres.get_mean().get_count() == column_count);
    //     res.set_mean(homogen_table::wrap(ndres.get_mean().flatten(q_, {event}), 1, column_count));
    // }
    // if (res_op.test(result_options::second_order_raw_moment)) {
    //     ONEDAL_ASSERT(ndres.get_sorm().get_count() == column_count);
    //     res.set_second_order_raw_moment(
    //         homogen_table::wrap(ndres.get_sorm().flatten(q_, {event}), 1, column_count));
    // }
    // if (res_op.test(result_options::variance)) {
    //     ONEDAL_ASSERT(ndres.get_varc().get_count() == column_count);
    //     res.set_variance(homogen_table::wrap(ndres.get_varc().flatten(q_, {event}), 1, column_count));
    // }
    // if (res_op.test(result_options::standard_deviation)) {
    //     ONEDAL_ASSERT(ndres.get_stdev().get_count() == column_count);
    //     res.set_standard_deviation(
    //         homogen_table::wrap(ndres.get_stdev().flatten(q_, {event}), 1, column_count));
    // }
    // if (res_op.test(result_options::variation)) {
    //     ONEDAL_ASSERT(ndres.get_vart().get_count() == column_count);
    //     res.set_variation(homogen_table::wrap(ndres.get_vart().flatten(q_, {event}), 1, column_count));
    // }
    return res;
}

template <typename Float>
struct finalize_compute_kernel_gpu<Float, method::dense, task::compute> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return finalize_compute<Float, task::compute>(ctx, desc, input);
    }
};

template struct finalize_compute_kernel_gpu<float, method::dense, task::compute>;
template struct finalize_compute_kernel_gpu<double, method::dense, task::compute>;

} // namespace oneapi::dal::basic_statistics::backend
