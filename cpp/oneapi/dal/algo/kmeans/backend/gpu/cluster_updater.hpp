/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/algo/kmeans/backend/gpu/kernels_integral.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/kernels_fp.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/empty_cluster_handling.hpp"

namespace oneapi::dal::kmeans::backend {

namespace pr = dal::backend::primitives;
namespace bk = dal::backend;
namespace spmd = oneapi::dal::preview::spmd;

template <typename Float>
class cluster_updater {
public:
    using kernels_fp_t = kernels_fp<Float>;

    cluster_updater(const sycl::queue& q,
                    const bk::communicator<spmd::device_memory_access::usm>& comm)
            : queue_(q),
              comm_(comm) {}

    auto& set_cluster_count(std::int64_t cluster_count) {
        ONEDAL_ASSERT(cluster_count > 0);
        cluster_count_ = cluster_count;
        return *this;
    }

    auto& set_part_count(std::int64_t part_count) {
        part_count_ = part_count;
        return *this;
    }

    auto& set_data(const pr::ndarray<Float, 2> data) {
        ONEDAL_ASSERT(data.has_data());
        data_ = data;
        row_count_ = data.get_dimension(0);
        column_count_ = data.get_dimension(1);
        return *this;
    }

    auto& set_initial_centroids(const pr::ndarray<Float, 2> initial_centroids) {
        initial_centroids_ = initial_centroids;
        return *this;
    }
    auto& set_centroid_squares(const pr::ndarray<Float, 1> centroid_squares) {
        centroid_squares_ = centroid_squares;
        return *this;
    }
    auto& set_data_squares(const pr::ndarray<Float, 1> data_squares) {
        data_squares_ = data_squares;
        return *this;
    }
    void allocate_buffers() {
        partial_centroids_ = pr::ndarray<Float, 2>::empty( //
            queue_,
            { part_count_ * cluster_count_, column_count_ },
            sycl::usm::alloc::device);

        counters_ = pr::ndarray<std::int32_t, 1>::empty( //
            queue_,
            cluster_count_,
            sycl::usm::alloc::device);
    }

    auto update(pr::ndarray<Float, 2>& centroids,
                pr::ndarray<Float, 2>& distance_block,
                pr::ndarray<Float, 2>& closest_distances,
                pr::ndarray<Float, 1>& objective_function,
                pr::ndarray<std::int32_t, 2>& responses,
                const bk::event_vector& deps = {}) -> std::tuple<Float, sycl::event> {
        ONEDAL_ASSERT(data_.get_dimension(0) == row_count_);
        ONEDAL_ASSERT(data_.get_dimension(1) == column_count_);
        ONEDAL_ASSERT(closest_distances.get_dimension(0) == row_count_);
        ONEDAL_ASSERT(partial_centroids_.get_dimension(0) == part_count_ * cluster_count_);
        ONEDAL_ASSERT(partial_centroids_.get_dimension(1) == column_count_);
        ONEDAL_ASSERT(counters_.get_dimension(0) == cluster_count_);
        ONEDAL_ASSERT(distance_block.get_dimension(1) == cluster_count_);
        ONEDAL_ASSERT(responses.get_dimension(0) == row_count_);
        ONEDAL_ASSERT(responses.get_dimension(1) == 1);
        ONEDAL_ASSERT(objective_function.get_dimension(0) == 1);
        ONEDAL_ASSERT(centroids.get_dimension(0) == cluster_count_);
        ONEDAL_ASSERT(centroids.get_dimension(1) == column_count_);

        const auto block_size_in_rows = distance_block.get_dimension(0);
        auto assign_event = kernels_fp_t::assign_clusters( //
            queue_,
            data_,
            initial_centroids_,
            data_squares_,
            centroid_squares_,
            block_size_in_rows,
            responses,
            distance_block,
            closest_distances,
            deps);

        auto count_event =
            count_clusters(queue_, responses, cluster_count_, counters_, { assign_event });

        auto count_reduce_event = comm_.allreduce(counters_.flatten(queue_, { count_event }));

        auto objective_function_event = kernels_fp_t::compute_objective_function( //
            queue_,
            closest_distances,
            objective_function,
            { assign_event });

        auto reset_event = partial_centroids_.fill(queue_, 0.0);
        auto centroids_event = kernels_fp_t::partial_reduce_centroids( //
            queue_,
            data_,
            responses,
            cluster_count_,
            part_count_,
            partial_centroids_,
            { assign_event, reset_event });

        objective_function_event.wait_and_throw();
        Float objective_function_value = objective_function.to_host(queue_).get_data()[0];

        auto objective_reduce_event = comm_.allreduce(objective_function_value);

        // Counters are needed in the `merge_reduce_centroids` function,
        // we wait until cross-rank reduction is finished
        count_reduce_event.wait();

        centroids_event = kernels_fp_t::merge_reduce_centroids( //
            queue_,
            counters_,
            partial_centroids_,
            part_count_,
            centroids,
            { count_event, centroids_event });

        auto centroids_reduce_event =
            comm_.allreduce(centroids.flatten(queue_, { centroids_event }));

        const std::int64_t empty_cluster_count =
            count_empty_clusters(queue_, cluster_count_, counters_, { count_event });

        // Centroids and objective function are needed in the `handle_empty_clusters`,
        // we wait until cross-rank reduction is finished
        centroids_reduce_event.wait();
        objective_reduce_event.wait();

        if (empty_cluster_count > 0) {
            auto [correction, event] = handle_empty_clusters( //
                queue_,
                comm_,
                empty_cluster_count,
                data_,
                responses,
                closest_distances,
                counters_,
                centroids,
                { assign_event, centroids_event });

            event.wait_and_throw();
            objective_function_value += correction;
        }

        return { objective_function_value, centroids_event };
    }

private:
    sycl::queue queue_;
    bk::communicator<spmd::device_memory_access::usm> comm_;

    std::int64_t row_count_ = 0;
    std::int64_t column_count_ = 0;
    std::int64_t cluster_count_ = 0;
    std::int64_t part_count_ = 0;

    pr::ndarray<Float, 2> data_;
    pr::ndarray<Float, 2> initial_centroids_;
    pr::ndarray<Float, 2> partial_centroids_;
    pr::ndarray<Float, 1> centroid_squares_;
    pr::ndarray<Float, 1> data_squares_;
    pr::ndarray<std::int32_t, 1> counters_;
};

} // namespace oneapi::dal::kmeans::backend
