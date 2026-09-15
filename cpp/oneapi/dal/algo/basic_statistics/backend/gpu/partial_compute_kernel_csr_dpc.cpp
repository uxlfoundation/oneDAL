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
#include "oneapi/dal/algo/basic_statistics/backend/gpu/partial_compute_kernel_misc.hpp"
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
/// Scaling the stored values weights every statistic and not only the sums, because the
/// batch CSR kernel folds the entries a column does not store in as literal zeros
/// (`fmin(cur_min, 0)`, `fmax(cur_max, 0)` and the `(row_count - cur_row_count) * mean^2`
/// term of `sum2_cent`, see `compute_kernel_csr_impl_dpc.cpp`), and `weights[i] * 0 == 0`.
/// So the kernel reduces over exactly the numbers the dense weighted path reduces over,
/// negative weights included.
///
/// Scaling is a separate `O(nnz)` pass. Folding the weights into the reduction instead
/// needs the weighted `fastCSR` kernel the DAAL side lacks, left to the follow-up PR
/// agreed in review.
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
    const std::int64_t row_count = data.get_row_count();
    const std::int64_t column_count = data.get_column_count();

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

    // `compute_stats` hands back the statistics where the kernels wrote them, in device
    // memory. The batch kernel's `operator()` would instead return a `compute_result` of
    // host copies, and merging those would move every statistic off the device and
    // straight back for `O(column_count)` of arithmetic, once per `partial_compute` call.
    auto batch = compute_kernel_csr_impl<Float>{}.compute_stats(ctx, { batch_data });
    const auto batch_stats = std::get<0>(batch);
    const sycl::event batch_ev = std::get<1>(batch);
    const Float* const batch_stats_ptr = batch_stats.get_data();
    const auto batch_stat = [=](stat which) {
        return pr::ndview<Float, 1>::wrap(batch_stats_ptr + which * column_count, column_count);
    };

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

        if (needed.sums) {
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
        if (needed.sums) {
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
            const auto* const sum_ptr = batch_stat(stat::sum).get_data();
            const auto* const sum2_ptr = batch_stat(stat::sum2).get_data();
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
