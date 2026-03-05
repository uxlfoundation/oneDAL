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

#include "oneapi/dal/algo/basic_statistics/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/algo/basic_statistics/backend/gpu/compute_kernel_csr_impl.hpp"
#include "oneapi/dal/table/csr_accessor.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/util/common.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/backend/communicator.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::basic_statistics::backend {

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

using method_t = method::sparse;
using task_t = task::compute;
using comm_t = bk::communicator<spmd::device_memory_access::usm>;
using input_t = compute_input<task_t>;
using result_t = compute_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

template <typename Float>
sycl::event compute_kernel_csr_impl<Float>::finalize_for_distr(
    sycl::queue& q,
    comm_t& communicator,
    pr::ndarray<Float, 2>& results,
    const input_t& input,
    const std::vector<sycl::event>& deps) {
    auto result_ptr = results.get_mutable_data();
    const csr_table csr_tdata = static_cast<const csr_table&>(input.get_data());
    auto [csr_data, column_indices, row_offsets] =
        csr_accessor<const Float>(csr_tdata).pull(q, { 0, -1 }, sparse_indexing::zero_based);
    const auto column_count = csr_tdata.get_column_count();
    const auto row_count = csr_tdata.get_row_count();
    const auto nonzero_count = csr_tdata.get_non_zero_count();

    auto host_results = results.flatten(q, deps);
    communicator
        .allreduce(host_results.get_slice(stat::min * column_count, column_count),
                   spmd::reduce_op::min)
        .wait();
    communicator
        .allreduce(host_results.get_slice(stat::max * column_count, column_count),
                   spmd::reduce_op::max)
        .wait();
    communicator
        .allreduce(host_results.get_slice(stat::sum * column_count, column_count),
                   spmd::reduce_op::sum)
        .wait();
    communicator
        .allreduce(host_results.get_slice(stat::sum2 * column_count, column_count),
                   spmd::reduce_op::sum)
        .wait();
    communicator
        .allreduce(host_results.get_slice(stat::moment2 * column_count, column_count),
                   spmd::reduce_op::sum)
        .wait();

    results.assign_from_host(q, host_results.get_data(), res_opt_count_ * column_count)
        .wait_and_throw();

    auto csr_data_ptr = csr_data.get_data();
    auto column_indices_ptr = column_indices.get_data();
    auto distr_range = sycl::range<1>(column_count);
    auto calc_s2c_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(distr_range, [=](std::int64_t col_idx) {
            auto mean_val = result_ptr[stat::sum * column_count + col_idx] / row_count;
            result_ptr[stat::mean * column_count + col_idx] = mean_val;
            Float sum2_cent = Float(0);
            std::int32_t nnz_row_count = 0;
            for (std::int32_t data_idx = 0; data_idx < nonzero_count; ++data_idx) {
                if (col_idx == column_indices_ptr[data_idx]) {
                    auto val = csr_data_ptr[data_idx];
                    sum2_cent += (val - mean_val) * (val - mean_val);
                    nnz_row_count += 1;
                }
            }
            // For zero values sum2_cent is just square of mean value
            sum2_cent += (row_count - nnz_row_count) * mean_val * mean_val;

            result_ptr[stat::sum2_cent * column_count + col_idx] = sum2_cent;
        });
    });

    host_results = results.flatten(q, { calc_s2c_event });
    communicator
        .allreduce(host_results.get_slice(stat::sum2_cent * column_count, column_count),
                   spmd::reduce_op::sum)
        .wait();
    auto allreduce_event =
        results.assign_from_host(q, host_results.get_data(), res_opt_count_ * column_count);

    auto final_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ allreduce_event });
        cgh.parallel_for(distr_range, [=](std::int64_t col_idx) {
            auto mean_val = result_ptr[stat::mean * column_count + col_idx];
            result_ptr[stat::variance * column_count + col_idx] =
                result_ptr[stat::sum2_cent * column_count + col_idx] / (row_count - 1);
            result_ptr[stat::stddev * column_count + col_idx] =
                sycl::sqrt(result_ptr[stat::variance * column_count + col_idx]);
            result_ptr[stat::variation * column_count + col_idx] =
                result_ptr[stat::stddev * column_count + col_idx] / mean_val;
        });
    });

    return final_event;
}

template <typename Float>
result_t compute_kernel_csr_impl<Float>::operator()(const bk::context_gpu& ctx,
                                                    const descriptor_t& desc,
                                                    const input_t& input) {
    auto queue = ctx.get_queue();
    const auto table = input.get_data();
    ONEDAL_ASSERT(table.get_kind() == csr_table::kind());
    const csr_table csr_tdata = static_cast<const csr_table&>(table);
    comm_t comm = ctx.get_communicator();
    const bool distr_mode = comm.get_rank_count() > 1;
    const auto column_count = csr_tdata.get_column_count();
    const auto row_count = csr_tdata.get_row_count();
    auto result_options = desc.get_result_options();
    const auto nonzero_count = csr_tdata.get_non_zero_count();
    auto [csr_data, column_indices, row_offsets] =
        csr_accessor<const Float>(csr_tdata).pull(queue,
                                                  { 0, -1 },
                                                  sparse_indexing::zero_based,
                                                  sycl::usm::alloc::device);
    auto csr_data_ptr = csr_data.get_data();
    auto column_indices_ptr = column_indices.get_data();

    using limits_t = std::numeric_limits<Float>;
    constexpr Float maximum = limits_t::max();

    // number of columns processed in one group
    const auto local_size = bk::device_max_wg_size(queue);
    // number of data elements processed by one working item
    constexpr std::int64_t n_items_per_work = 512;
    const auto num_data_blocks = nonzero_count / (local_size * n_items_per_work) +
                                 bool(nonzero_count % (local_size * n_items_per_work));
    const auto num_col_blocks = column_count / local_size + bool(column_count % local_size);

    auto result_data =
        pr::ndarray<Float, 2>::empty(queue,
                                     { num_data_blocks * res_opt_count_, column_count },
                                     sycl::usm::alloc::device);
    auto result_data_ptr = result_data.get_mutable_data();

    const auto nd_range =
        bk::make_multiple_nd_range_3d({ num_data_blocks, num_col_blocks, local_size },
                                      { 1, 1, local_size });
    const auto merge_range = bk::make_multiple_nd_range_1d(column_count, 1);
    // First order kernel calculates basic statistics for min, max, sum, sum squares.
    // The computation is splitted by blocks for columns and data to achieve best performance
    // on GPU.
    auto first_order_event = queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<Float, 1> local_res_buf(local_size * res_opt_count_, cgh);
        cgh.parallel_for(nd_range, [=](auto item) {
            std::int64_t block_id = item.get_global_id(0);
            std::int64_t col_ofs = item.get_global_id(1) * local_size;
            std::int64_t local_id = item.get_local_id(2);
            std::int64_t data_ofs = block_id * local_size * n_items_per_work;
            if (col_ofs >= column_count) {
                return;
            }
            Float* work_group_buf =
                local_res_buf.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
            auto local_buf = work_group_buf + local_id * res_opt_count_;
            local_buf[stat::min] = maximum;
            local_buf[stat::max] = -maximum;
            local_buf[stat::sum] = Float(0);
            local_buf[stat::sum2] = Float(0);
            local_buf[stat::sum2_cent] = Float(0);
            item.barrier(sycl::access::fence_space::local_space);
            for (std::int64_t idx = 0; idx < n_items_per_work; ++idx) {
                auto data_idx = data_ofs + local_id * n_items_per_work + idx;
                if (data_idx >= nonzero_count) {
                    break;
                }
                auto col_idx = column_indices_ptr[data_idx] - col_ofs;
                auto val = csr_data_ptr[data_idx];
                if (col_idx >= 0 && col_idx < local_size) {
                    sycl::atomic_ref<Float,
                                     sycl::memory_order_relaxed,
                                     sycl::memory_scope_device,
                                     sycl::access::address_space::local_space>
                        col_min(local_res_buf[col_idx * res_opt_count_ + stat::min]);
                    col_min.fetch_min(val);
                    sycl::atomic_ref<Float,
                                     sycl::memory_order_relaxed,
                                     sycl::memory_scope_device,
                                     sycl::access::address_space::local_space>
                        col_max(local_res_buf[col_idx * res_opt_count_ + stat::max]);
                    col_max.fetch_max(val);
                    sycl::atomic_ref<Float,
                                     sycl::memory_order_relaxed,
                                     sycl::memory_scope_device,
                                     sycl::access::address_space::local_space>
                        col_sum(local_res_buf[col_idx * res_opt_count_ + stat::sum]);
                    col_sum.fetch_add(val);
                    sycl::atomic_ref<Float,
                                     sycl::memory_order_relaxed,
                                     sycl::memory_scope_device,
                                     sycl::access::address_space::local_space>
                        col_sum2(local_res_buf[col_idx * res_opt_count_ + stat::sum2]);
                    col_sum2.fetch_add(val * val);
                }
            }
            item.barrier(sycl::access::fence_space::local_space);
            if ((local_id + col_ofs) >= column_count) {
                return;
            }
            const auto col_idx = col_ofs + local_id;
            const auto block_idx = block_id * res_opt_count_ * column_count;
            result_data_ptr[stat::min * column_count + block_idx + col_idx] = local_buf[stat::min];
            result_data_ptr[stat::max * column_count + block_idx + col_idx] = local_buf[stat::max];
            result_data_ptr[stat::sum * column_count + block_idx + col_idx] = local_buf[stat::sum];
            result_data_ptr[stat::sum2 * column_count + block_idx + col_idx] =
                local_buf[stat::sum2];
        });
    });

    // First order merge kernel merges results for data blocks computed on previous step.
    auto merge_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ first_order_event });
        cgh.parallel_for(merge_range, [=](auto item) {
            const auto col_idx = item.get_global_id();
            auto cur_min = result_data_ptr[stat::min * column_count + col_idx];
            auto cur_max = result_data_ptr[stat::max * column_count + col_idx];
            auto cur_sum = result_data_ptr[stat::sum * column_count + col_idx];
            auto cur_sum2 = result_data_ptr[stat::sum2 * column_count + col_idx];
            for (std::int64_t block_id = 1; block_id < num_data_blocks; ++block_id) {
                const auto block_idx = block_id * res_opt_count_ * column_count;
                cur_min =
                    sycl::min(cur_min,
                              result_data_ptr[stat::min * column_count + block_idx + col_idx]);
                cur_max =
                    sycl::max(cur_max,
                              result_data_ptr[stat::max * column_count + block_idx + col_idx]);
                cur_sum += result_data_ptr[stat::sum * column_count + block_idx + col_idx];
                cur_sum2 += result_data_ptr[stat::sum2 * column_count + block_idx + col_idx];
            }
            result_data_ptr[stat::min * column_count + col_idx] = cur_min;
            result_data_ptr[stat::max * column_count + col_idx] = cur_max;
            result_data_ptr[stat::sum * column_count + col_idx] = cur_sum;
            result_data_ptr[stat::sum2 * column_count + col_idx] = cur_sum2;
        });
    });

    // Second order kernel computes sum squares centered.
    // And additionally computes the number of non-zero rows
    // in order to proper finalize min, max, sum squares centered statistics,
    // since zero values are invovled to the results of them.
    auto second_order_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ merge_event });
        sycl::local_accessor<Float, 1> local_res_buf(local_size, cgh);
        sycl::local_accessor<Float, 1> mean_vals_buf(local_size, cgh);
        sycl::local_accessor<std::int32_t, 1> row_counter(local_size, cgh);
        cgh.parallel_for(nd_range, [=](auto item) {
            std::int64_t block_id = item.get_global_id(0);
            std::int64_t col_ofs = item.get_global_id(1) * local_size;
            std::int64_t local_id = item.get_local_id(2);
            std::int64_t data_ofs = block_id * local_size * n_items_per_work;
            if (col_ofs >= column_count) {
                return;
            }
            Float* work_group_buf =
                local_res_buf.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
            Float* mean_vals =
                mean_vals_buf.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
            std::int32_t* row_counter_buf =
                row_counter.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
            mean_vals[local_id] =
                result_data_ptr[stat::sum * column_count + col_ofs + local_id] / row_count;
            row_counter_buf[local_id] = 0;
            work_group_buf[local_id] = Float(0);
            item.barrier(sycl::access::fence_space::local_space);
            // Merge results of first order moments
            for (std::int64_t idx = 0; idx < n_items_per_work; ++idx) {
                auto data_idx = data_ofs + local_id * n_items_per_work + idx;
                if (data_idx >= nonzero_count) {
                    break;
                }
                auto col_idx = column_indices_ptr[data_idx] - col_ofs;
                auto val = csr_data_ptr[data_idx];
                if (col_idx >= 0 && col_idx < local_size) {
                    sycl::atomic_ref<Float,
                                     sycl::memory_order_relaxed,
                                     sycl::memory_scope_device,
                                     sycl::access::address_space::local_space>
                        col_sum2_cent(local_res_buf[col_idx]);
                    auto mean_val = mean_vals[col_idx];
                    col_sum2_cent.fetch_add((val - mean_val) * (val - mean_val));
                    sycl::atomic_ref<std::int32_t,
                                     sycl::memory_order_relaxed,
                                     sycl::memory_scope_device,
                                     sycl::access::address_space::local_space>
                        row_counter_at(row_counter[col_idx]);
                    row_counter_at.fetch_add(1);
                }
            }
            item.barrier(sycl::access::fence_space::local_space);
            if ((local_id + col_ofs) >= column_count) {
                return;
            }
            const auto col_idx = col_ofs + local_id;
            const auto block_idx = block_id * res_opt_count_ * column_count;
            result_data_ptr[stat::sum2_cent * column_count + block_idx + col_idx] =
                work_group_buf[local_id];
            // Mean is no need to merge it is the same for all blocks
            result_data_ptr[stat::mean * column_count + block_idx + col_idx] = mean_vals[local_id];
            // Temporary save row_counts into varitaion placeholder in order to merge it
            result_data_ptr[stat::variation * column_count + block_idx + col_idx] =
                row_counter_buf[local_id];
        });
    });

    // Second order merge kernel finalizes computations on basic statistics and
    // merges sum squares centered statistic among data blocks.
    auto second_merge_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ second_order_event });
        cgh.parallel_for(merge_range, [=](auto item) {
            const auto col_idx = item.get_global_id();
            auto cur_sum2_cent = result_data_ptr[stat::sum2_cent * column_count + col_idx];
            auto mean_val = result_data_ptr[stat::mean * column_count + col_idx];
            auto cur_row_count = result_data_ptr[stat::variation * column_count + col_idx];
            for (std::int64_t block_id = 1; block_id < num_data_blocks; ++block_id) {
                const auto block_idx = block_id * res_opt_count_ * column_count;
                cur_sum2_cent +=
                    result_data_ptr[stat::sum2_cent * column_count + block_idx + col_idx];
                cur_row_count +=
                    result_data_ptr[stat::variation * column_count + block_idx + col_idx];
            }
            // In case when there are zeros in column it must be compared with min and max
            // And added to sum2_cent
            if (row_count != cur_row_count) {
                auto cur_min = result_data_ptr[stat::min * column_count + col_idx];
                auto cur_max = result_data_ptr[stat::max * column_count + col_idx];
                result_data_ptr[stat::min * column_count + col_idx] = Float(sycl::fmin(cur_min, 0));
                result_data_ptr[stat::max * column_count + col_idx] = Float(sycl::fmax(cur_max, 0));
                cur_sum2_cent += Float(row_count - cur_row_count) * mean_val * mean_val;
            }
            result_data_ptr[stat::sum2_cent * column_count + col_idx] = cur_sum2_cent;
            result_data_ptr[stat::moment2 * column_count + col_idx] =
                result_data_ptr[stat::sum2 * column_count + col_idx] / row_count;
            result_data_ptr[stat::variance * column_count + col_idx] =
                cur_sum2_cent / (row_count - 1);
            result_data_ptr[stat::stddev * column_count + col_idx] =
                sycl::sqrt(result_data_ptr[stat::variance * column_count + col_idx]);
            result_data_ptr[stat::variation * column_count + col_idx] =
                result_data_ptr[stat::stddev * column_count + col_idx] / mean_val;
        });
    });

    if (distr_mode) {
        second_merge_event = finalize_for_distr(queue, comm, result_data, input, { merge_event });
    }
    return get_result(queue, result_data, result_options, { second_merge_event });
}

template class compute_kernel_csr_impl<float>;
template class compute_kernel_csr_impl<double>;

} // namespace oneapi::dal::basic_statistics::backend
#endif // ONEDAL_DATA_PARALLEL
