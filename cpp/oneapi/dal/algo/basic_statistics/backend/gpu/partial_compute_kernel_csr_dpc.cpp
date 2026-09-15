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
#include "oneapi/dal/algo/basic_statistics/backend/gpu/compute_kernel_csr_impl.hpp"

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
#include "oneapi/dal/table/csr_accessor.hpp"

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

/// Which groups of partial statistics the requested `result_options` need carried across
/// `partial_compute` calls. mean / variance / stddev / etc. are derived only at finalize
/// time, so they only require the moment partials to be accumulated.
struct required_partials {
    bool min_max;
    bool moments;
};

inline required_partials get_required_partials(const descriptor_t& desc) {
    const auto res_op = desc.get_result_options();
    const bool min_max = res_op.test(result_options::min) || res_op.test(result_options::max);
    const bool moments =
        res_op.test(result_options::sum) || res_op.test(result_options::sum_squares) ||
        res_op.test(result_options::sum_squares_centered) || res_op.test(result_options::mean) ||
        res_op.test(result_options::variance) ||
        res_op.test(result_options::second_order_raw_moment) ||
        res_op.test(result_options::standard_deviation) || res_op.test(result_options::variation);
    // With nothing requested at all, still carry the moment partials so that the partial
    // result stays a valid input for a later `partial_compute` or `finalize_compute` with
    // a wider set of result options.
    return { min_max, moments || !min_max };
}

/// A CSR table whose values have been scaled by the per-row weights, together with the
/// arrays that own the device memory it points at. Keep the whole aggregate alive for
/// as long as `table` is in use.
template <typename Float>
struct scaled_csr {
    csr_table table;
    dal::array<Float> values;
    dal::array<std::int64_t> column_indices;
    dal::array<std::int64_t> row_offsets;
};

/// Apply per-row weights to a CSR table.
///
/// Weighted statistics in this algorithm are plain per-row scaling of the data: the
/// dense path multiplies every element of row `i` by `weights[i]` and keeps the plain
/// row count as the observation count. Scaling the stored values reproduces that for
/// every statistic, not only for the sums, because the batch CSR kernel this feeds
/// computes statistics of the *dense* interpretation of the table: it counts the stored
/// entries per column and, where a column has fewer of them than there are rows, folds
/// the missing entries in as literal zeros (`fmin(cur_min, 0)` / `fmax(cur_max, 0)` and
/// the `(row_count - cur_row_count) * mean^2` term for `sum2_cent`, see
/// `compute_kernel_csr_impl_dpc.cpp`). Each such zero stands for a `0` in some row `i`,
/// and `weights[i] * 0 == 0`, so it is exactly what weighting that row would have
/// produced; each stored value `v` in row `i` becomes `weights[i] * v`, which is again
/// exactly what weighting that row would have produced. The kernel therefore reduces
/// over the same multiset of numbers as the dense weighted path, so extrema, sums and
/// centered sums all agree, negative weights included, and the merge and finalize steps
/// need no change.
///
/// This is one `O(nnz)` pass on top of the kernel's own, which is what makes it a
/// two-pass formulation on this backend. Folding the weights into the reduction instead
/// is left to the follow-up PR agreed in review, since it also needs the DAAL-side
/// weighted `fastCSR` kernel that the CPU backend lacks.
template <typename Float>
inline scaled_csr<Float> scale_csr_by_weights(sycl::queue& q,
                                              const csr_table& csr,
                                              const table& weights) {
    const std::int64_t row_count = csr.get_row_count();
    const std::int64_t column_count = csr.get_column_count();
    const std::int64_t nonzero_count = csr.get_non_zero_count();

    auto [values, column_indices, row_offsets] =
        csr_accessor<const Float>(csr).pull(q,
                                            { 0, -1 },
                                            sparse_indexing::zero_based,
                                            alloc::device);
    const auto weights_nd = pr::table2ndarray_1d<Float>(q, weights, alloc::device);

    auto scaled = dal::array<Float>::empty(q, nonzero_count, alloc::device);
    auto* const scaled_ptr = scaled.get_mutable_data();
    const auto* const values_ptr = values.get_data();
    const auto* const offsets_ptr = row_offsets.get_data();
    const auto* const weights_ptr = weights_nd.get_data();

    auto ev = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(row_count), [=](sycl::item<1> id) {
            const std::int64_t row = id.get_id(0);
            const Float weight = weights_ptr[row];
            for (std::int64_t i = offsets_ptr[row]; i < offsets_ptr[row + 1]; ++i) {
                scaled_ptr[i] = values_ptr[i] * weight;
            }
        });
    });
    ev.wait_and_throw();

    auto scaled_table = csr_table::wrap(q,
                                        scaled.get_data(),
                                        column_indices.get_data(),
                                        row_offsets.get_data(),
                                        row_count,
                                        column_count,
                                        sparse_indexing::zero_based);
    return { std::move(scaled_table),
             std::move(scaled),
             std::move(column_indices),
             std::move(row_offsets) };
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
/// min/max computed by the GPU CSR batch kernel.
template <typename Float>
auto update_min_max_results(sycl::queue& q,
                            const pr::ndview<Float, 1>& prev_min,
                            const Float* current_min_ptr,
                            const pr::ndview<Float, 1>& prev_max,
                            const Float* current_max_ptr,
                            const std::int64_t column_count,
                            const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(update_min_max_results, q);

    auto result_min = pr::ndarray<Float, 1>::empty(q, column_count, alloc::device);
    auto result_max = pr::ndarray<Float, 1>::empty(q, column_count, alloc::device);

    auto result_min_ptr = result_min.get_mutable_data();
    auto result_max_ptr = result_max.get_mutable_data();

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
/// sum_squares_centered stored on the partial is computed via the identity
/// `sum_squares_centered = sum_squares - sum^2 / n`; finalize will recompute it exactly.
template <typename Float>
auto update_partial_sums(sycl::queue& q,
                         const pr::ndview<Float, 1>& prev_sum,
                         const Float* current_sums_ptr,
                         const pr::ndview<Float, 1>& prev_sum2,
                         const Float* current_sums2_ptr,
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

/// Copy one statistic out of the batch kernel's device result array into a partial result
/// table of its own, without leaving the device. The copy is what keeps the partial result
/// from pinning the whole `res_opt_count_ x num_data_blocks x column_count` scratch array
/// alive.
template <typename Float>
inline table extract_stat(sycl::queue& q,
                          const Float* stats,
                          stat which,
                          const std::int64_t column_count,
                          const dal::backend::event_vector& deps = {}) {
    auto extracted = pr::ndarray<Float, 1>::empty(q, column_count, alloc::device);
    auto ev = dal::backend::copy(q,
                                 extracted.get_mutable_data(),
                                 stats + which * column_count,
                                 column_count,
                                 deps);
    return homogen_table::wrap(extracted.flatten(q, { ev }), 1, column_count);
}

template <typename Float, typename Task>
static partial_compute_result<Task> partial_compute(const context_gpu& ctx,
                                                    const descriptor_t& desc,
                                                    const partial_compute_input<Task>& input) {
    auto& q = ctx.get_queue();
    const auto data = input.get_data();
    ONEDAL_ASSERT(data.get_kind() == csr_table::kind());

    const auto needed = get_required_partials(desc);

    // The GPU CSR batch kernel takes no weights of its own, so fold them into the data
    // by scaling the stored values per row. `weighted` must outlive the kernel call: it
    // owns the device memory that the scaled table points at.
    const auto weights = input.get_weights();
    scaled_csr<Float> weighted;
    if (weights.has_data()) {
        ONEDAL_ASSERT(weights.get_row_count() == data.get_row_count());
        ONEDAL_ASSERT(weights.get_column_count() == std::int64_t(1));
        weighted = scale_csr_by_weights<Float>(q, static_cast<const csr_table&>(data), weights);
    }
    const table& batch_data = weights.has_data() ? static_cast<const table&>(weighted.table) : data;

    // Take the batch statistics through `compute_stats` rather than through the batch
    // kernel's `operator()`: the latter returns a `compute_result` whose tables are host
    // copies, so merging them here would move every statistic off the device and straight
    // back for `O(column_count)` of arithmetic, once per `partial_compute` call. Reading
    // the kernel's own device array keeps the whole online accumulation on the device --
    // the partial result tables are device-backed too, so no statistic touches the host
    // between the first `partial_compute` and `finalize_compute`.
    auto batch = compute_kernel_csr_impl<Float>{}.compute_stats(ctx, { batch_data });
    const auto batch_stats = std::get<0>(batch);
    const sycl::event batch_ev = std::get<1>(batch);
    const Float* const batch_stats_ptr = batch_stats.get_data();
    const auto batch_stat = [=](stat which) {
        return batch_stats_ptr + which * data.get_column_count();
    };

    const std::int64_t row_count = data.get_row_count();
    const std::int64_t column_count = data.get_column_count();

    auto result = partial_compute_result();
    const auto& prev = input.get_prev();
    const bool has_prev = prev.get_partial_n_rows().has_data();

    if (has_prev) {
        // The count is only read by the merge kernel below, so ask for it on the device:
        // the partial result carries it in device memory, and the default `shared`
        // allocation would force a copy out of it.
        const auto nobs_nd =
            pr::table2ndarray_1d<Float>(q, prev.get_partial_n_rows(), sycl::usm::alloc::device);
        auto [result_nobs, nobs_ev] = update_partial_n_rows_results(q, row_count, nobs_nd);

        if (needed.min_max) {
            const auto prev_min_nd =
                pr::table2ndarray_1d<Float>(q, prev.get_partial_min(), sycl::usm::alloc::device);
            const auto prev_max_nd =
                pr::table2ndarray_1d<Float>(q, prev.get_partial_max(), sycl::usm::alloc::device);
            auto [res_min, res_max, mm_ev] = update_min_max_results(q,
                                                                    prev_min_nd,
                                                                    batch_stat(stat::min),
                                                                    prev_max_nd,
                                                                    batch_stat(stat::max),
                                                                    column_count,
                                                                    { nobs_ev, batch_ev });
            result.set_partial_min(
                homogen_table::wrap(res_min.flatten(q, { mm_ev }), 1, column_count));
            result.set_partial_max(
                homogen_table::wrap(res_max.flatten(q, { mm_ev }), 1, column_count));
        }

        if (needed.moments) {
            const auto prev_sum_nd =
                pr::table2ndarray_1d<Float>(q, prev.get_partial_sum(), sycl::usm::alloc::device);
            const auto prev_sum2_nd = pr::table2ndarray_1d<Float>(q,
                                                                  prev.get_partial_sum_squares(),
                                                                  sycl::usm::alloc::device);
            auto [res_sum, res_sum2, res_sum2cent, sum_ev] =
                update_partial_sums(q,
                                    prev_sum_nd,
                                    batch_stat(stat::sum),
                                    prev_sum2_nd,
                                    batch_stat(stat::sum2),
                                    column_count,
                                    result_nobs,
                                    { nobs_ev, batch_ev });

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

        if (needed.min_max) {
            result.set_partial_min(
                extract_stat<Float>(q, batch_stats_ptr, stat::min, column_count, { batch_ev }));
            result.set_partial_max(
                extract_stat<Float>(q, batch_stats_ptr, stat::max, column_count, { batch_ev }));
        }
        if (needed.moments) {
            result.set_partial_sum(
                extract_stat<Float>(q, batch_stats_ptr, stat::sum, column_count, { batch_ev }));
            result.set_partial_sum_squares(
                extract_stat<Float>(q, batch_stats_ptr, stat::sum2, column_count, { batch_ev }));
            // Initial sum_squares_centered is the batch's own centered sum of squares
            // for a single-batch case where n == row_count and mean == sum / n, i.e.
            // `sum_squares - sum^2 / n`. Compute it here so that downstream
            // partial_compute merges have a consistent baseline.
            auto sums2cent =
                pr::ndarray<Float, 1>::empty(q, column_count, sycl::usm::alloc::device);
            auto sums2cent_ptr = sums2cent.get_mutable_data();
            const auto* const sum_ptr = batch_stat(stat::sum);
            const auto* const sum2_ptr = batch_stat(stat::sum2);
            const Float n = Float(row_count);
            auto ev = q.submit([&](sycl::handler& cgh) {
                cgh.depends_on(batch_ev);
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
