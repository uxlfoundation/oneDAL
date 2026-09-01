/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <tuple>
#include <daal/src/algorithms/kmeans/kmeans_init_kernel.h>

#include "oneapi/dal/algo/kmeans/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/kernels_integral.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/cluster_updater.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/kernels_fp.hpp"
#include "oneapi/dal/algo/kmeans/detail/train_init_centroids.hpp"
#include "oneapi/dal/exceptions.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/backend/transfer.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/detail/profiler.hpp"

namespace oneapi::dal::kmeans::backend {

using dal::backend::context_gpu;
using descriptor_t = detail::descriptor_base<task::clustering>;

namespace interop = dal::backend::interop;
namespace pr = dal::backend::primitives;
namespace de = dal::detail;
namespace bk = dal::backend;

template <typename Float>
static pr::ndarray<Float, 2> get_initial_centroids(const dal::backend::context_gpu& ctx,
                                                   const descriptor_t& params,
                                                   const train_input<task::clustering>& input) {
    auto& queue = ctx.get_queue();

    const auto data = input.get_data();

    const std::int64_t column_count = data.get_column_count();
    const std::int64_t cluster_count = params.get_cluster_count();

    daal::data_management::NumericTablePtr daal_initial_centroids;

    if (!input.get_initial_centroids().has_data()) {
        daal_initial_centroids =
            oneapi::dal::kmeans::detail::daal_generate_centroids<Float, method::lloyd_dense>(params,
                                                                                             data);
        daal::data_management::BlockDescriptor<Float> block;
        daal_initial_centroids->getBlockOfRows(0,
                                               cluster_count,
                                               daal::data_management::readOnly,
                                               block);
        Float* initial_centroids_ptr = block.getBlockPtr();
        auto arr_host_initial =
            pr::ndarray<Float, 2>::wrap(initial_centroids_ptr, { cluster_count, column_count });
        return arr_host_initial.to_device(queue);
    }
    auto initial_centroids_ptr = row_accessor<const Float>(input.get_initial_centroids())
                                     .pull(queue, { 0, -1 }, sycl::usm::alloc::device);
    return pr::ndarray<Float, 2>::wrap(initial_centroids_ptr, { cluster_count, column_count });
}

template <typename Float>
struct train_kernel_gpu<Float, method::lloyd_dense, task::clustering> {
    train_result<task::clustering> operator()(const dal::backend::context_gpu& ctx,
                                              const descriptor_t& desc,
                                              const train_input<task::clustering>& input) const {
        auto& queue = ctx.get_queue();
        auto& comm = ctx.get_communicator();
        const auto data = input.get_data();
        ONEDAL_ASSERT(data.get_kind() != dal::csr_table::kind());
        const std::int64_t row_count = data.get_row_count();
        const std::int64_t column_count = data.get_column_count();
        const std::int64_t cluster_count = desc.get_cluster_count();
        const std::int64_t max_iteration_count = desc.get_max_iteration_count();
        const double accuracy_threshold = desc.get_accuracy_threshold();
        dal::detail::check_mul_overflow(cluster_count, column_count);

        auto result = train_result<task::clustering>{};

        auto data_ptr =
            row_accessor<const Float>(data).pull(queue, { 0, -1 }, sycl::usm::alloc::device);
        auto arr_data = pr::ndarray<Float, 2>::wrap(data_ptr, { row_count, column_count });

        // TODO: Use truly-distributed algorithm for computing initial centroids.
        // The current implementation of distributed algorithm initializes centroids
        // independently on each rank using the data available. This may result in
        // inconsistent results between single-rank and distributed runs. To fix
        // this issue the correct distributed implementation of K-Means++ should be
        // called underneath.
        auto arr_initial = get_initial_centroids<Float>(ctx, desc, input);

        std::int64_t block_size_in_rows =
            std::min(row_count,
                     kernels_fp<Float>::get_block_size_in_rows(queue, column_count, cluster_count));

        dal::detail::check_mul_overflow(block_size_in_rows, cluster_count);
        std::int64_t part_count =
            kernels_fp<Float>::get_part_count_for_partial_centroids(queue,
                                                                    column_count,
                                                                    cluster_count);

        auto arr_centroid_squares =
            pr::ndarray<Float, 1>::empty(queue, cluster_count, sycl::usm::alloc::device);
        auto arr_data_squares =
            pr::ndarray<Float, 1>::empty(queue, row_count, sycl::usm::alloc::device);
        auto data_squares_event =
            kernels_fp<Float>::compute_squares(queue, arr_data, arr_data_squares);
        auto arr_distance_block =
            pr::ndarray<Float, 2>::empty(queue,
                                         { block_size_in_rows, cluster_count },
                                         sycl::usm::alloc::device);
        auto arr_closest_distances =
            pr::ndarray<Float, 2>::empty(queue, { row_count, 1 }, sycl::usm::alloc::device);
        auto arr_centroids = pr::ndarray<Float, 2>::empty(queue,
                                                          { cluster_count, column_count },
                                                          sycl::usm::alloc::device);
        auto arr_responses =
            pr::ndarray<std::int32_t, 2>::empty(queue, { row_count, 1 }, sycl::usm::alloc::device);
        auto arr_objective_function =
            pr::ndarray<Float, 1>::empty(queue, 1, sycl::usm::alloc::device);

        Float prev_objective_function = de::limits<Float>::max();
        std::int64_t iter;
        // Seed `arr_centroids` with the initial centroids. Every centroid an iteration recomputes
        // is overwritten, but a cluster can keep its previous centroid instead: an empty cluster
        // whose only available candidate row already sits on its assigned centroid is left alone
        // (see `fill_empty_clusters`). On iteration 0 "previous" means the initial centroids, so
        // without this copy such a cluster would read back uninitialized device memory.
        sycl::event centroids_event = arr_centroids.assign(queue, arr_initial);

        auto updater = cluster_updater<Float>{ queue, comm }
                           .set_cluster_count(cluster_count)
                           .set_part_count(part_count)
                           .set_data(arr_data)
                           .set_data_squares(arr_data_squares);
        updater.allocate_buffers();

        for (iter = 0; iter < max_iteration_count; iter++) {
            auto centroid_squares_event =
                kernels_fp<Float>::compute_squares(queue,
                                                   iter == 0 ? arr_initial : arr_centroids,
                                                   arr_centroid_squares);
            updater.set_initial_centroids(iter == 0 ? arr_initial : arr_centroids);
            updater.set_centroid_squares(arr_centroid_squares);
            auto [objective_function, update_clusters_event] =
                updater.update(arr_centroids,
                               arr_distance_block,
                               arr_closest_distances,
                               arr_objective_function,
                               arr_responses,
                               { centroids_event, data_squares_event, centroid_squares_event });
            centroids_event = update_clusters_event;
            // Both comparisons are inclusive, matching the CPU kernel
            // (`par->accuracyThreshold >= 0` and `l2Norm <= par->accuracyThreshold` in
            // kmeans_lloyd_batch_impl.i). A threshold of exactly zero is a valid request
            // for "stop as soon as the objective function stops improving"; with `> 0` and
            // a strict `>` it instead meant "run all `max_iteration_count` iterations",
            // because a converged iteration leaves the objective function unchanged
            // rather than larger.
            if (accuracy_threshold >= 0 &&
                objective_function + accuracy_threshold >= prev_objective_function) {
                iter++;
                break;
            }
            prev_objective_function = objective_function;
        }
        auto centroid_squares_event = kernels_fp<Float>::compute_squares(queue,
                                                                         arr_centroids,
                                                                         arr_centroid_squares,
                                                                         { centroids_event });
        auto assign_event = kernels_fp<Float>::assign_clusters(queue,
                                                               arr_data,
                                                               arr_centroids,
                                                               arr_data_squares,
                                                               arr_centroid_squares,
                                                               block_size_in_rows,
                                                               arr_responses,
                                                               arr_distance_block,
                                                               arr_closest_distances,
                                                               { centroid_squares_event });

        auto objective_event = kernels_fp<Float>::compute_objective_function( //
            queue,
            arr_closest_distances,
            arr_objective_function,
            { assign_event });

        Float final_objective_function =
            arr_objective_function.to_host(queue, { objective_event }).get_data()[0];

        {
            ONEDAL_PROFILER_TASK(allreduce_final_objective);
            comm.allreduce(final_objective_function).wait();
        }
        result.set_objective_function_value(final_objective_function);

        result.set_responses(dal::homogen_table::wrap(arr_responses.flatten(queue), row_count, 1));

        result.set_iteration_count(iter);

        result.set_model(model<task::clustering>().set_centroids(
            dal::homogen_table::wrap(arr_centroids.flatten(queue), cluster_count, column_count)));

        return result;
    }
};

template struct train_kernel_gpu<float, method::lloyd_dense, task::clustering>;
template struct train_kernel_gpu<double, method::lloyd_dense, task::clustering>;

} // namespace oneapi::dal::kmeans::backend
