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

#include "oneapi/dal/algo/dbscan/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/algo/dbscan/backend/gpu/results.hpp"

#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/backend/communicator.hpp"

namespace oneapi::dal::dbscan::backend {

namespace bk = oneapi::dal::backend;
namespace pr = oneapi::dal::backend::primitives;
namespace spmd = oneapi::dal::preview::spmd;
namespace de = oneapi::dal::detail;

using dal::backend::context_gpu;

using descriptor_t = detail::descriptor_base<task::clustering>;
using result_t = compute_result<task::clustering>;
using input_t = compute_input<task::clustering>;

template <typename Float>
static result_t compute_kernel_kd_tree_impl(const context_gpu& ctx,
                                            const descriptor_t& desc,
                                            const table& local_data,
                                            const table& local_weights) {
    auto& comm = ctx.get_communicator();
    auto& queue = ctx.get_queue();

    std::int64_t rank_count = comm.get_rank_count();

    auto current_rank = comm.get_rank();

    auto prev_node = (current_rank - 1 + rank_count) % rank_count;
    auto next_node = (current_rank + 1) % rank_count;

    const std::int64_t local_row_count = local_data.get_row_count();
    const std::int64_t column_count = local_data.get_column_count();

    std::int64_t global_row_count = local_row_count;

    std::int64_t max_local_block_size = local_row_count;
    {
        ONEDAL_PROFILER_TASK(allreduce_rows_count_global, queue);
        comm.allreduce(max_local_block_size, spmd::reduce_op::max).wait();
    }

    auto send_recv_replace_local_size = array<std::int64_t>::zeros(1);
    send_recv_replace_local_size.get_mutable_data()[0] = local_row_count;

    auto global_rank_offsets = array<std::int64_t>::zeros(rank_count);
    global_rank_offsets.get_mutable_data()[current_rank] = local_row_count;
    {
        ONEDAL_PROFILER_TASK(allreduce_recv_counts, queue);
        comm.allreduce(global_rank_offsets, spmd::reduce_op::sum).wait();
    }
    {
        ONEDAL_PROFILER_TASK(allreduce_rows_count_global, queue);
        comm.allreduce(global_row_count, spmd::reduce_op::sum).wait();
    }

    std::int64_t local_offset = 0;

    for (std::int64_t i = 0; i < current_rank; i++) {
        ONEDAL_ASSERT(global_rank_offsets.get_data()[i] >= 0);
        local_offset += global_rank_offsets.get_data()[i];
    }

    const auto data_nd = pr::table2ndarray<Float>(queue, local_data, sycl::usm::alloc::device);

    pr::ndarray<Float, 2> data_nd_replace;
    if (rank_count > 1) {
        auto data_copy_count = column_count * local_row_count;
        data_nd_replace = pr::ndarray<Float, 2>::empty(queue,
                                                       { max_local_block_size, column_count },
                                                       sycl::usm::alloc::device);
        bk::copy(queue, data_nd_replace.get_mutable_data(), data_nd.get_data(), data_copy_count, {})
            .wait_and_throw();
    }

    bool use_weights = false;
    if (local_weights.get_row_count() == data_nd.get_dimension(0)) {
        use_weights = true;
    }

    pr::ndarray<Float, 2> weights_nd;
    if (use_weights) {
        weights_nd = pr::table2ndarray<Float>(queue, local_weights, sycl::usm::alloc::device);
    }

    pr::ndarray<Float, 2> weights_nd_replace;
    if (rank_count > 1 && use_weights) {
        auto weights_copy_count = 1 * local_row_count;
        weights_nd_replace = pr::ndarray<Float, 2>::empty(queue,
                                                          { max_local_block_size, 1 },
                                                          sycl::usm::alloc::device);
        bk::copy(queue,
                 weights_nd_replace.get_mutable_data(),
                 weights_nd.get_data(),
                 weights_copy_count,
                 {})
            .wait_and_throw();
    }
    const Float epsilon = desc.get_epsilon() * desc.get_epsilon();
    const std::int64_t min_observations = desc.get_min_observations();

    auto [arr_cores, cores_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, local_row_count, 0, sycl::usm::alloc::device);

    auto [arr_neighbours, neighbours_event] =
        pr::ndarray<Float, 1>::full(queue, local_row_count, 0, sycl::usm::alloc::device);

    auto [arr_responses, responses_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, local_row_count, -1, sycl::usm::alloc::device);

    auto [observation_indices, observation_indices_event] =
        pr::ndarray<bool, 1>::full(queue, local_row_count, false, sycl::usm::alloc::device);

    auto [total_points_queue_size_arr, total_points_queue_size_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, 1, 0, sycl::usm::alloc::device);

    auto [local_points_queue_size_arr, local_points_queue_size_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, 1, 0, sycl::usm::alloc::device);
    sycl::event::wait({ cores_event,
                        neighbours_event,
                        responses_event,
                        observation_indices_event,
                        total_points_queue_size_event,
                        local_points_queue_size_event });

    auto get_cores_event = kernels_fp<Float>::get_cores(queue,
                                                        data_nd,
                                                        weights_nd,
                                                        arr_cores,
                                                        arr_neighbours,
                                                        epsilon,
                                                        min_observations);

    for (std::int64_t j = 0; j < rank_count - 1; j++) {
        comm.sendrecv_replace(queue,
                              data_nd_replace.get_mutable_data(),
                              max_local_block_size * column_count,
                              prev_node,
                              next_node)
            .wait();
        if (use_weights) {
            comm.sendrecv_replace(queue,
                                  weights_nd_replace.get_mutable_data(),
                                  max_local_block_size,
                                  prev_node,
                                  next_node)
                .wait();
        }
        comm.sendrecv_replace(send_recv_replace_local_size.get_mutable_data(),
                              1,
                              prev_node,
                              next_node)
            .wait();
        auto local_row_block_count = send_recv_replace_local_size.get_data()[0];
        auto actual_current_block = data_nd_replace.get_row_slice(0, local_row_block_count);
        if (use_weights) {
            auto actual_weights = weights_nd_replace.get_row_slice(0, local_row_block_count);
            kernels_fp<Float>::get_cores_send_recv_replace(queue,
                                                           data_nd,
                                                           actual_current_block,
                                                           actual_weights,
                                                           arr_cores,
                                                           arr_neighbours,
                                                           epsilon,
                                                           min_observations,
                                                           { get_cores_event })
                .wait_and_throw();
        }
        else {
            kernels_fp<Float>::get_cores_send_recv_replace(queue,
                                                           data_nd,
                                                           actual_current_block,
                                                           weights_nd,
                                                           arr_cores,
                                                           arr_neighbours,
                                                           epsilon,
                                                           min_observations,
                                                           { get_cores_event })
                .wait_and_throw();
        }
    }

    std::int64_t cluster_count = 0;

    std::int32_t cluster_index =
        kernels_fp<Float>::start_next_cluster(queue, arr_cores, arr_responses, { get_cores_event });
    cluster_index =
        cluster_index < local_row_count ? cluster_index + local_offset : global_row_count;
    {
        ONEDAL_PROFILER_TASK(allreduce_cluster_index, queue);
        comm.allreduce(cluster_index, spmd::reduce_op::min).wait();
    }

    if (cluster_index < 0) {
        return make_results(queue, desc, data_nd, arr_responses, arr_cores, 0, 0);
    }

    while (cluster_index < de::integral_cast<std::int32_t>(global_row_count)) {
        cluster_count++;
        bool in_range =
            cluster_index >= local_offset && cluster_index < local_offset + local_row_count;

        std::int32_t local_points_queue_size = 0;

        if (in_range) {
            set_arr_value(queue, arr_responses, cluster_index - local_offset, cluster_count - 1)
                .wait_and_throw();
            set_init_index(queue, observation_indices, cluster_index - local_offset, true)
                .wait_and_throw();
            local_points_queue_size++;
        }

        std::int32_t total_points_queue_size = local_points_queue_size;

        {
            ONEDAL_PROFILER_TASK(allreduce_total_points_queue_size_outer, queue);
            comm.allreduce(total_points_queue_size, spmd::reduce_op::sum).wait();
        }

        while (total_points_queue_size != 0) {
            auto recv_counts = array<std::int64_t>::zeros(rank_count);
            recv_counts.get_mutable_data()[current_rank] = local_points_queue_size;
            {
                ONEDAL_PROFILER_TASK(allreduce_recv_counts, queue);
                comm.allreduce(recv_counts, spmd::reduce_op::sum).wait();
            }

            auto displs = array<std::int64_t>::zeros(rank_count);
            auto displs_ptr = displs.get_mutable_data();
            std::int64_t total_count = 0;
            for (std::int64_t i = 0; i < rank_count; i++) {
                displs_ptr[i] = total_count;
                total_count += recv_counts.get_data()[i];
            }

            auto [current_points_queue, current_points_queue_event] =
                pr::ndarray<Float, 2>::full(queue,
                                            { total_points_queue_size, column_count },
                                            0,
                                            sycl::usm::alloc::device);

            set_arr_value(queue, total_points_queue_size_arr, 0, total_points_queue_size)
                .wait_and_throw();

            sycl::event fill_queue_event;

            if (local_points_queue_size != 0) {
                fill_queue_event =
                    kernels_fp<Float>::fill_current_points_queue(queue,
                                                                 data_nd,
                                                                 observation_indices,
                                                                 current_points_queue,
                                                                 local_points_queue_size_arr,
                                                                 displs_ptr[current_rank],
                                                                 { current_points_queue_event });
                set_arr_value(queue, local_points_queue_size_arr, 0, 0, { fill_queue_event })
                    .wait_and_throw();
            }
            {
                ONEDAL_PROFILER_TASK(allreduce_xtx, queue);
                comm.allreduce(current_points_queue.flatten(queue, { fill_queue_event }),
                               spmd::reduce_op::sum)
                    .wait();
            }

            kernels_fp<Float>::update_points_queue(queue,
                                                   data_nd,
                                                   arr_cores,
                                                   current_points_queue,
                                                   arr_responses,
                                                   total_points_queue_size_arr,
                                                   observation_indices,
                                                   epsilon,
                                                   cluster_count - 1,
                                                   { fill_queue_event })
                .wait_and_throw();

            local_points_queue_size =
                kernels_fp<Float>::get_points_queue_size(queue, total_points_queue_size_arr);

            total_points_queue_size = local_points_queue_size;

            {
                ONEDAL_PROFILER_TASK(allreduce_total_points_queue_size_inner, queue);
                comm.allreduce(total_points_queue_size, spmd::reduce_op::sum).wait();
            }
        }

        cluster_index = kernels_fp<Float>::start_next_cluster(queue, arr_cores, arr_responses);
        cluster_index =
            cluster_index < local_row_count ? cluster_index + local_offset : global_row_count;
        {
            ONEDAL_PROFILER_TASK(cluster_index, queue);
            comm.allreduce(cluster_index, spmd::reduce_op::min).wait();
        }
    }

    return make_results(queue, desc, data_nd, arr_responses, arr_cores, cluster_count);
}

template <typename Float>
static result_t compute(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    return compute_kernel_kd_tree_impl<Float>(ctx, desc, input.get_data(), input.get_weights());
}

template <typename Float>
struct compute_kernel_gpu<Float, method::kd_tree, task::clustering> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute<Float>(ctx, desc, input);
    }
};

template struct compute_kernel_gpu<float, method::kd_tree, task::clustering>;
template struct compute_kernel_gpu<double, method::kd_tree, task::clustering>;

} // namespace oneapi::dal::dbscan::backend
