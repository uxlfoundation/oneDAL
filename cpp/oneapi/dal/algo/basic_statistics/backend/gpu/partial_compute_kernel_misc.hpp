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

#pragma once

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/detail/profiler.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::basic_statistics::backend {

namespace pr = oneapi::dal::backend::primitives;

/// Merge steps of `partial_compute`, shared by the dense and the sparse GPU kernels: both
/// of them accumulate the same partial statistics, and only the way the current block's
/// statistics are produced differs. Every argument and result stays in device memory.

template <typename Float>
inline auto update_partial_n_rows_results(sycl::queue& q,
                                          const std::int64_t row_count,
                                          const pr::ndview<Float, 1>& nobs,
                                          const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(update_partial_n_rows_results, q);

    auto result_nobs = pr::ndarray<Float, 1>::empty(q, 1, sycl::usm::alloc::device);
    auto result_nobs_ptr = result_nobs.get_mutable_data();
    auto nobs_ptr = nobs.get_data();

    auto nobs_update_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(sycl::range<1>(1), [=](sycl::item<1>) {
            result_nobs_ptr[0] = nobs_ptr[0] + row_count;
        });
    });

    return std::make_tuple(result_nobs, nobs_update_event);
}

template <typename Float>
inline auto update_min_max_results(sycl::queue& q,
                                   const pr::ndview<Float, 1>& min,
                                   const pr::ndview<Float, 1>& current_min,
                                   const pr::ndview<Float, 1>& max,
                                   const pr::ndview<Float, 1>& current_max,
                                   const std::int64_t column_count,
                                   const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(update_min_max_results, q);

    auto result_min = pr::ndarray<Float, 1>::empty(q, column_count, sycl::usm::alloc::device);
    auto result_max = pr::ndarray<Float, 1>::empty(q, column_count, sycl::usm::alloc::device);

    auto result_min_ptr = result_min.get_mutable_data();
    auto result_max_ptr = result_max.get_mutable_data();

    auto current_min_ptr = current_min.get_data();
    auto current_max_ptr = current_max.get_data();

    auto min_data = min.get_data();
    auto max_data = max.get_data();

    auto merge_min_max_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(sycl::range<1>(column_count), [=](sycl::item<1> id) {
            result_min_ptr[id] = sycl::fmin(current_min_ptr[id], min_data[id]);
            result_max_ptr[id] = sycl::fmax(current_max_ptr[id], max_data[id]);
        });
    });

    return std::make_tuple(result_min, result_max, merge_min_max_event);
}

/// Merges a block's sums into the partial ones. The centered sums of squares merge with
/// Chan et al.'s update, `ssc1 + ssc2 + n1 * n2 / n * (mean2 - mean1)^2`, which does not
/// cancel the way `sum_squares - sum^2 / n` does when the mean is large.
///
/// @param[in]  q                The SYCL queue
/// @param[in]  sums             The partial sums
/// @param[in]  current_sums     The block's sums
/// @param[in]  sums2            The partial sums of squares
/// @param[in]  current_sums2    The block's sums of squares
/// @param[in]  sums2cent        The partial centered sums of squares
/// @param[in]  current_sums2cent The block's centered sums of squares
/// @param[in]  column_count     The number of columns
/// @param[in]  nobs             The number of rows after the merge, a one-element array
/// @param[in]  block_row_count  The number of rows in the block
/// @param[in]  deps             Events indicating availability of the inputs
///
/// @return The merged sums, sums of squares and centered sums of squares, and the event of the merge
template <typename Float>
inline auto update_partial_sums(sycl::queue& q,
                                const pr::ndview<Float, 1>& sums,
                                const pr::ndview<Float, 1>& current_sums,
                                const pr::ndview<Float, 1>& sums2,
                                const pr::ndview<Float, 1>& current_sums2,
                                const pr::ndview<Float, 1>& sums2cent,
                                const pr::ndview<Float, 1>& current_sums2cent,
                                const std::int64_t column_count,
                                const pr::ndview<Float, 1>& nobs,
                                const std::int64_t block_row_count,
                                const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(update_partial_sums, q);

    auto result_sums = pr::ndarray<Float, 1>::empty(q, column_count, sycl::usm::alloc::device);
    auto result_sums2 = pr::ndarray<Float, 1>::empty(q, column_count, sycl::usm::alloc::device);
    auto result_sums2cent = pr::ndarray<Float, 1>::empty(q, column_count, sycl::usm::alloc::device);

    auto result_sums_ptr = result_sums.get_mutable_data();
    auto result_sums2_ptr = result_sums2.get_mutable_data();
    auto result_sums2cent_ptr = result_sums2cent.get_mutable_data();

    const auto current_sums_ptr = current_sums.get_data();
    const auto current_sums2_ptr = current_sums2.get_data();
    const auto current_sums2cent_ptr = current_sums2cent.get_data();
    const auto sums_ptr = sums.get_data();
    const auto sums2_ptr = sums2.get_data();
    const auto sums2cent_ptr = sums2cent.get_data();
    const auto nobs_ptr = nobs.get_data();
    const Float n2 = Float(block_row_count);
    // An empty side has zero sums, so a zero inverse count drops its mean shift.
    const Float inv_n2 = block_row_count > 0 ? Float(1) / n2 : Float(0);

    auto update_sums_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(sycl::range<1>(column_count), [=](sycl::item<1> id) {
            const Float n = nobs_ptr[0];
            const Float n1 = n - n2;
            const Float inv_n1 = n1 > Float(0) ? Float(1) / n1 : Float(0);
            const Float weight = n > Float(0) ? n1 * n2 / n : Float(0);
            const Float mean_shift = current_sums_ptr[id] * inv_n2 - sums_ptr[id] * inv_n1;
            result_sums_ptr[id] = sums_ptr[id] + current_sums_ptr[id];
            result_sums2_ptr[id] = sums2_ptr[id] + current_sums2_ptr[id];
            result_sums2cent_ptr[id] =
                sums2cent_ptr[id] + current_sums2cent_ptr[id] + weight * mean_shift * mean_shift;
        });
    });

    return std::make_tuple(result_sums, result_sums2, result_sums2cent, update_sums_event);
}

} // namespace oneapi::dal::basic_statistics::backend

#endif // ONEDAL_DATA_PARALLEL
