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

#pragma once

#include "oneapi/dal/algo/basic_statistics/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/util/common.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/backend/communicator.hpp"

#include <tuple>

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::basic_statistics::backend {

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

enum stat { min, max, sum, sum2, sum2_cent, mean, moment2, variance, stddev, variation };

/// The per-column statistics of a CSR table in device memory, together with the device
/// arrays the kernels that produced them read from.
///
/// Keep the whole aggregate alive until `event` has completed. Freeing a device USM
/// allocation is not a synchronizing operation -- `dal::array`'s deleter calls
/// `sycl::free` straight away (`backend/memory.hpp`) -- so dropping `values` /
/// `column_indices` / `row_offsets` while the statistics kernels are still enqueued is a
/// device-side use-after-free. It is the CSR input the kernels read, not the output, that
/// makes this necessary: `stats` is what the caller wanted, so it stays alive on its own.
/// Same reasoning, and same shape, as `scaled_csr` in `partial_compute_kernel_csr_dpc.cpp`.
template <typename Float>
struct csr_stats {
    pr::ndarray<Float, 2> stats;
    sycl::event event;
    dal::array<Float> values;
    dal::array<std::int64_t> column_indices;
    dal::array<std::int64_t> row_offsets;
};

template <typename Float>
class compute_kernel_csr_impl {
    using method_t = method::sparse;
    using task_t = task::compute;
    using comm_t = bk::communicator<spmd::device_memory_access::usm>;
    using input_t = compute_input<task_t>;
    using result_t = compute_result<task_t>;
    using descriptor_t = detail::descriptor_base<task_t>;

public:
    result_t operator()(const bk::context_gpu& ctx, const descriptor_t& desc, const input_t& input);

    /// Computes per-column statistics for a CSR table directly in GPU memory, returning
    /// the data block along with a sync event.
    ///
    /// @note Use this instead of `operator()` when consuming statistics on-device (e.g., in
    ///       `partial_compute`) to avoid expensive host-device roundtrips.
    ///
    /// The kernels are only enqueued, not awaited, so the returned aggregate has to own the
    /// pulled CSR arrays they read -- see `csr_stats`. Hold it until `csr_stats::event` has
    /// completed.
    ///
    /// @param ctx   GPU execution context
    /// @param input Input dataset in CSR format
    /// @return      The statistics as a 2D array of size
    ///              `(num_data_blocks * res_opt_count_) x column_count`, where the first
    ///              `res_opt_count_` rows store the merged statistics, row `stat::<name>`
    ///              holding that statistic, and the remaining rows act as per-block
    ///              scratchpad memory; the event tracking the final writing kernel; and the
    ///              device arrays that kernel reads from.
    csr_stats<Float> compute_stats(const bk::context_gpu& ctx, const input_t& input);

private:
    // Number of different basic statistics
    static constexpr std::int32_t res_opt_count_ = 10;
    // An array of basic statistics
    const result_option_id res_options_[res_opt_count_] = { result_options::min,
                                                            result_options::max,
                                                            result_options::sum,
                                                            result_options::sum_squares,
                                                            result_options::sum_squares_centered,
                                                            result_options::mean,
                                                            result_options::second_order_raw_moment,
                                                            result_options::variance,
                                                            result_options::standard_deviation,
                                                            result_options::variation };

    result_t get_result(sycl::queue q,
                        const pr::ndarray<Float, 2> computed_result,
                        result_option_id requested_results,
                        const std::vector<sycl::event>& deps = {}) {
        result_t res;
        std::vector<sycl::event> res_events;
        res.set_result_options(requested_results);
        if (requested_results.test(result_options::min)) {
            auto index = get_result_option_index(result_options::min);
            auto [res_table, event] = get_result_table(q, computed_result, index, deps);
            res.set_min(res_table);
            res_events.push_back(event);
        }
        if (requested_results.test(result_options::max)) {
            auto index = get_result_option_index(result_options::max);
            auto [res_table, event] = get_result_table(q, computed_result, index, deps);
            res.set_max(res_table);
            res_events.push_back(event);
        }
        if (requested_results.test(result_options::sum)) {
            auto index = get_result_option_index(result_options::sum);
            auto [res_table, event] = get_result_table(q, computed_result, index, deps);
            res.set_sum(res_table);
            res_events.push_back(event);
        }
        if (requested_results.test(result_options::sum_squares)) {
            auto index = get_result_option_index(result_options::sum_squares);
            auto [res_table, event] = get_result_table(q, computed_result, index, deps);
            res.set_sum_squares(res_table);
            res_events.push_back(event);
        }
        if (requested_results.test(result_options::sum_squares_centered)) {
            auto index = get_result_option_index(result_options::sum_squares_centered);
            auto [res_table, event] = get_result_table(q, computed_result, index, deps);
            res.set_sum_squares_centered(res_table);
            res_events.push_back(event);
        }
        if (requested_results.test(result_options::mean)) {
            auto index = get_result_option_index(result_options::mean);
            auto [res_table, event] = get_result_table(q, computed_result, index, deps);
            res.set_mean(res_table);
            res_events.push_back(event);
        }
        if (requested_results.test(result_options::second_order_raw_moment)) {
            auto index = get_result_option_index(result_options::second_order_raw_moment);
            auto [res_table, event] = get_result_table(q, computed_result, index, deps);
            res.set_second_order_raw_moment(res_table);
            res_events.push_back(event);
        }
        if (requested_results.test(result_options::variance)) {
            auto index = get_result_option_index(result_options::variance);
            auto [res_table, event] = get_result_table(q, computed_result, index, deps);
            res.set_variance(res_table);
            res_events.push_back(event);
        }
        if (requested_results.test(result_options::standard_deviation)) {
            auto index = get_result_option_index(result_options::standard_deviation);
            auto [res_table, event] = get_result_table(q, computed_result, index, deps);
            res.set_standard_deviation(res_table);
            res_events.push_back(event);
        }
        if (requested_results.test(result_options::variation)) {
            auto index = get_result_option_index(result_options::variation);
            auto [res_table, event] = get_result_table(q, computed_result, index, deps);
            res.set_variation(res_table);
            res_events.push_back(event);
        }
        sycl::event::wait_and_throw(res_events);
        return res;
    }

    std::tuple<table, sycl::event> get_result_table(sycl::queue q,
                                                    const pr::ndarray<Float, 2> computed_result,
                                                    std::int32_t index,
                                                    const std::vector<sycl::event>& deps = {}) {
        ONEDAL_ASSERT(computed_result.has_data());
        auto column_count = computed_result.get_dimension(1);
        const auto arr = dal::array<Float>::empty(column_count);
        const auto res_arr_ptr = arr.get_mutable_data();
        const auto computed_res_ptr = computed_result.get_data() + index * column_count;
        auto event =
            dal::backend::copy_usm2host(q, res_arr_ptr, computed_res_ptr, column_count, deps);
        return std::make_tuple(homogen_table::wrap(arr, 1, column_count), event);
    }

    std::int32_t get_result_option_index(result_option_id opt) {
        std::int32_t index = 0;
        while (!opt.test(res_options_[index]))
            ++index;
        return index;
    }

    sycl::event finalize_for_distr(sycl::queue& q,
                                   comm_t& communicator,
                                   pr::ndarray<Float, 2>& results,
                                   const input_t& input,
                                   const std::vector<sycl::event>& deps);
};

} // namespace oneapi::dal::basic_statistics::backend
#endif // ONEDAL_DATA_PARALLEL
