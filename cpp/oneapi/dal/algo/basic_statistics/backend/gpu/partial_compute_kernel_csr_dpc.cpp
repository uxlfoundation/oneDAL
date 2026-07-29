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

#include "oneapi/dal/algo/basic_statistics/backend/gpu/partial_compute_kernel.hpp"
#include "oneapi/dal/algo/basic_statistics/backend/gpu/compute_kernel.hpp"

#include "oneapi/dal/algo/basic_statistics/backend/basic_statistics_interop.hpp"

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/util/common.hpp"

#include "oneapi/dal/table/csr.hpp"

namespace oneapi::dal::basic_statistics::backend {

namespace bk = dal::backend;
namespace pr = oneapi::dal::backend::primitives;

using alloc = sycl::usm::alloc;

using bk::context_gpu;
using method_t = method::sparse;
using task_t = task::compute;
using input_t = partial_compute_input<task_t>;
using result_t = partial_compute_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

/// Build a per-batch sparse descriptor that requests the minimal set of statistics
/// (min, max, sum, sum_squares) needed to accumulate across partial_compute calls.
/// mean / variance / stddev / etc. are derived only at finalize time.
template <typename Float>
inline auto get_csr_desc_to_compute(const descriptor_t& desc) {
    const auto res_op = desc.get_result_options();
    const bool has_min_max = res_op.test(result_options::min) || res_op.test(result_options::max);
    const bool has_moments =
        res_op.test(result_options::sum) || res_op.test(result_options::sum_squares) ||
        res_op.test(result_options::sum_squares_centered) || res_op.test(result_options::mean) ||
        res_op.test(result_options::variance) ||
        res_op.test(result_options::second_order_raw_moment) ||
        res_op.test(result_options::standard_deviation) || res_op.test(result_options::variation);
    auto local_desc =
        basic_statistics::descriptor<Float, method::sparse, basic_statistics::task::compute>();
    result_option_id options;
    if (has_min_max) {
        options = options | result_options::min | result_options::max;
    }
    if (has_moments) {
        options = options | result_options::sum | result_options::sum_squares;
    }
    if (!has_min_max && !has_moments) {
        options = options | result_options::sum | result_options::sum_squares;
    }
    local_desc.set_result_options(options);
    return local_desc;
}

/// Bump nobs by `row_count` and return the updated total (1-element device array).
template <typename Float>
auto update_partial_n_rows_results(sycl::queue& q,
                                   const std::int64_t row_count,
                                   const pr::ndview<Float, 1>& nobs,
                                   const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(update_partial_n_rows_results, q);

    auto result_nobs = pr::ndarray<Float, 1>::empty(q, 1, alloc::device);
    auto result_nobs_ptr = result_nobs.get_mutable_data();
    auto nobs_ptr = nobs.get_data();

    auto ev = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(sycl::range<1>(1), [=](sycl::item<1>) {
            result_nobs_ptr[0] = nobs_ptr[0] + row_count;
        });
    });
    return std::make_tuple(result_nobs, ev);
}

/// Element-wise min/max merge of the previous partial min/max with the current-batch
/// min/max returned by the GPU CSR batch kernel.
template <typename Float>
auto update_min_max_results(sycl::queue& q,
                            const pr::ndview<Float, 1>& prev_min,
                            const table& current_min,
                            const pr::ndview<Float, 1>& prev_max,
                            const table& current_max,
                            const std::int64_t column_count,
                            const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(update_min_max_results, q);

    auto result_min = pr::ndarray<Float, 1>::empty(q, column_count, alloc::device);
    auto result_max = pr::ndarray<Float, 1>::empty(q, column_count, alloc::device);

    auto result_min_ptr = result_min.get_mutable_data();
    auto result_max_ptr = result_max.get_mutable_data();

    auto current_min_ptr =
        pr::table2ndarray_1d<Float>(q, current_min, sycl::usm::alloc::device).get_data();
    auto current_max_ptr =
        pr::table2ndarray_1d<Float>(q, current_max, sycl::usm::alloc::device).get_data();

    auto prev_min_data = prev_min.get_data();
    auto prev_max_data = prev_max.get_data();

    auto ev = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(sycl::range<1>(column_count), [=](sycl::item<1> id) {
            result_min_ptr[id] = sycl::fmin(current_min_ptr[id], prev_min_data[id]);
            result_max_ptr[id] = sycl::fmax(current_max_ptr[id], prev_max_data[id]);
        });
    });
    return std::make_tuple(result_min, result_max, ev);
}

/// Merge previous partial sums / sum_squares with current-batch sums / sum_squares.
/// sums2cent stored on the partial is the "single-node" running estimate
/// `sum^2 - sum*sum/n`; finalize will recompute the exact value.
template <typename Float>
auto update_partial_sums(sycl::queue& q,
                         const pr::ndview<Float, 1>& prev_sum,
                         const table& current_sum,
                         const pr::ndview<Float, 1>& prev_sum2,
                         const table& current_sum2,
                         const std::int64_t column_count,
                         const pr::ndview<Float, 1>& nobs,
                         const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(update_partial_sums_csr, q);

    auto result_sums = pr::ndarray<Float, 1>::empty(q, column_count, alloc::device);
    auto result_sums2 = pr::ndarray<Float, 1>::empty(q, column_count, alloc::device);
    auto result_sums2cent = pr::ndarray<Float, 1>::empty(q, column_count, alloc::device);

    auto result_sums_ptr = result_sums.get_mutable_data();
    auto result_sums2_ptr = result_sums2.get_mutable_data();
    auto result_sums2cent_ptr = result_sums2cent.get_mutable_data();

    auto current_sums_ptr =
        pr::table2ndarray_1d<Float>(q, current_sum, sycl::usm::alloc::device).get_data();
    auto current_sums2_ptr =
        pr::table2ndarray_1d<Float>(q, current_sum2, sycl::usm::alloc::device).get_data();

    auto nobs_ptr = nobs.get_data();
    auto prev_sums_data = prev_sum.get_data();
    auto prev_sums2_data = prev_sum2.get_data();

    auto ev = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(sycl::range<1>(column_count), [=](sycl::item<1> id) {
            result_sums_ptr[id] = prev_sums_data[id] + current_sums_ptr[id];
            result_sums2_ptr[id] = prev_sums2_data[id] + current_sums2_ptr[id];
            result_sums2cent_ptr[id] =
                result_sums2_ptr[id] - result_sums_ptr[id] * result_sums_ptr[id] / nobs_ptr[0];
        });
    });
    return std::make_tuple(result_sums, result_sums2, result_sums2cent, ev);
}

template <typename Float, typename Task>
static partial_compute_result<Task> partial_compute(const context_gpu& ctx,
                                                    const descriptor_t& desc,
                                                    const partial_compute_input<Task>& input) {
    auto& q = ctx.get_queue();
    const auto data = input.get_data();
    ONEDAL_ASSERT(data.get_kind() == csr_table::kind());

    // The GPU CSR batch kernel does not accept per-row weights.
    if (input.get_weights().has_data()) {
        throw unimplemented(dal::detail::error_messages::method_not_implemented());
    }

    const auto local_desc = get_csr_desc_to_compute<Float>(desc);
    const auto res_op = local_desc.get_result_options();

    auto batch_kernel = compute_kernel_gpu<Float, method::sparse, task::compute>{};
    auto batch_result = batch_kernel(ctx, local_desc, { data });

    const std::int64_t row_count = data.get_row_count();
    const std::int64_t column_count = data.get_column_count();

    auto result = partial_compute_result();
    const auto& prev = input.get_prev();
    const bool has_prev = prev.get_partial_n_rows().has_data();

    if (has_prev) {
        const auto nobs_nd = pr::table2ndarray_1d<Float>(q, prev.get_partial_n_rows());
        auto [result_nobs, nobs_ev] = update_partial_n_rows_results(q, row_count, nobs_nd);

        if (res_op.test(result_options::min) || res_op.test(result_options::max)) {
            const auto prev_min_nd =
                pr::table2ndarray_1d<Float>(q, prev.get_partial_min(), sycl::usm::alloc::device);
            const auto prev_max_nd =
                pr::table2ndarray_1d<Float>(q, prev.get_partial_max(), sycl::usm::alloc::device);
            auto [res_min, res_max, mm_ev] = update_min_max_results(q,
                                                                    prev_min_nd,
                                                                    batch_result.get_min(),
                                                                    prev_max_nd,
                                                                    batch_result.get_max(),
                                                                    column_count,
                                                                    { nobs_ev });
            result.set_partial_min(
                homogen_table::wrap(res_min.flatten(q, { mm_ev }), 1, column_count));
            result.set_partial_max(
                homogen_table::wrap(res_max.flatten(q, { mm_ev }), 1, column_count));
        }

        if (res_op.test(result_options::sum)) {
            const auto prev_sum_nd =
                pr::table2ndarray_1d<Float>(q, prev.get_partial_sum(), sycl::usm::alloc::device);
            const auto prev_sum2_nd = pr::table2ndarray_1d<Float>(q,
                                                                  prev.get_partial_sum_squares(),
                                                                  sycl::usm::alloc::device);
            auto [res_sum, res_sum2, res_sum2cent, sum_ev] =
                update_partial_sums(q,
                                    prev_sum_nd,
                                    batch_result.get_sum(),
                                    prev_sum2_nd,
                                    batch_result.get_sum_squares(),
                                    column_count,
                                    result_nobs,
                                    { nobs_ev });

            result.set_partial_sum(
                homogen_table::wrap(res_sum.flatten(q, { sum_ev }), 1, column_count));
            result.set_partial_sum_squares(
                homogen_table::wrap(res_sum2.flatten(q, { sum_ev }), 1, column_count));
            result.set_partial_sum_squares_centered(
                homogen_table::wrap(res_sum2cent.flatten(q, { sum_ev }), 1, column_count));
        }

        result.set_partial_n_rows(homogen_table::wrap(result_nobs.flatten(q, { nobs_ev }), 1, 1));
    }
    else {
        auto [init_nobs, init_ev] =
            pr::ndarray<Float, 1>::full(q, { 1 }, Float(row_count), sycl::usm::alloc::device);
        init_ev.wait_and_throw();

        if (res_op.test(result_options::min)) {
            result.set_partial_min(batch_result.get_min());
        }
        if (res_op.test(result_options::max)) {
            result.set_partial_max(batch_result.get_max());
        }
        if (res_op.test(result_options::sum)) {
            result.set_partial_sum(batch_result.get_sum());
            result.set_partial_sum_squares(batch_result.get_sum_squares());
            // Initial sum_squares_centered is the batch's own centered sum of squares
            // for a single-batch case where n == row_count and mean == sum / n. That
            // is exactly the identity `sum^2 - sum*sum/n`; compute it here so that
            // downstream partial_compute merges have a consistent baseline.
            const auto sum_nd =
                pr::table2ndarray_1d<Float>(q, batch_result.get_sum(), sycl::usm::alloc::device);
            const auto sum2_nd = pr::table2ndarray_1d<Float>(q,
                                                             batch_result.get_sum_squares(),
                                                             sycl::usm::alloc::device);
            auto sums2cent =
                pr::ndarray<Float, 1>::empty(q, column_count, sycl::usm::alloc::device);
            auto sums2cent_ptr = sums2cent.get_mutable_data();
            auto sum_ptr = sum_nd.get_data();
            auto sum2_ptr = sum2_nd.get_data();
            const Float n = Float(row_count);
            auto ev = q.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>(column_count), [=](sycl::item<1> id) {
                    sums2cent_ptr[id] = sum2_ptr[id] - sum_ptr[id] * sum_ptr[id] / n;
                });
            });
            result.set_partial_sum_squares_centered(
                homogen_table::wrap(sums2cent.flatten(q, { ev }), 1, column_count));
        }

        result.set_partial_n_rows(homogen_table::wrap(init_nobs.flatten(q, {}), 1, 1));
    }

    return result;
}

template <typename Float>
struct partial_compute_kernel_gpu<Float, method_t, task_t> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return partial_compute<Float, task_t>(ctx, desc, input);
    }
};

template struct partial_compute_kernel_gpu<float, method_t, task_t>;
template struct partial_compute_kernel_gpu<double, method_t, task_t>;

} // namespace oneapi::dal::basic_statistics::backend
