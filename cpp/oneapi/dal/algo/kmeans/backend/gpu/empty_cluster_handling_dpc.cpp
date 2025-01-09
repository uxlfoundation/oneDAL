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

#include "oneapi/dal/algo/kmeans/backend/gpu/empty_cluster_handling.hpp"
#include "oneapi/dal/backend/primitives/selection/select_indexed_rows.hpp"
#include "oneapi/dal/backend/primitives/sort/sort.hpp"
#include "oneapi/dal/detail/profiler.hpp"

namespace spmd = oneapi::dal::preview::spmd;

namespace oneapi::dal::kmeans::backend {

template <typename Float>
static auto fill_candidate_indices_and_distances(sycl::queue& queue,
                                                 std::int64_t candidate_count,
                                                 const pr::ndview<Float, 2>& closest_distances,
                                                 pr::ndview<std::int32_t, 1>& candidate_indices,
                                                 pr::ndview<Float, 1>& candidate_distances,
                                                 const bk::event_vector& deps = {}) -> sycl::event {
    ONEDAL_PROFILER_TASK(fill_candidates, queue);
    ONEDAL_ASSERT(candidate_count > 0);
    ONEDAL_ASSERT(closest_distances.get_dimension(0) >= candidate_count);
    ONEDAL_ASSERT(closest_distances.get_dimension(1) == 1);
    ONEDAL_ASSERT(candidate_indices.get_dimension(0) == candidate_indices.get_dimension(0));
    ONEDAL_ASSERT(candidate_indices.get_dimension(0) >= candidate_count);

    constexpr auto alloc = sycl::usm::alloc::device;
    const std::int64_t elem_count = closest_distances.get_dimension(0);
    auto indices = pr::ndarray<std::int32_t, 1>::empty(queue, { elem_count }, alloc);
    auto values = pr::ndarray<Float, 1>::empty(queue, { elem_count }, alloc);

    const Float* closest_distances_ptr = closest_distances.get_data();
    std::int32_t* indices_ptr = indices.get_mutable_data();
    std::int32_t* candidate_indices_ptr = candidate_indices.get_mutable_data();
    Float* values_ptr = values.get_mutable_data();
    Float* candidate_distances_ptr = candidate_distances.get_mutable_data();

    auto fill_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(bk::make_range_1d(elem_count), [=](sycl::id<1> idx) {
            indices_ptr[idx] = idx;
            values_ptr[idx] = -closest_distances_ptr[idx];
        });
    });

    pr::radix_sort_indices_inplace<Float, std::int32_t> radix_sort{ queue };
    auto sort_event = radix_sort(values, indices, { fill_event });

    auto copy_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(sort_event);
        cgh.parallel_for(bk::make_range_1d(candidate_count), [=](sycl::id<1> idx) {
            candidate_distances_ptr[idx] = -values_ptr[idx];
            candidate_indices_ptr[idx] = indices_ptr[idx];
        });
    });

    // We need to wait as `indices` and `values` will be deallocated
    // as we leave scope of the function
    copy_event.wait_and_throw();

    return copy_event;
}

static auto fill_empty_cluster_indices(sycl::queue& queue,
                                       std::int64_t candidate_count,
                                       const pr::ndarray<std::int32_t, 1>& counters,
                                       pr::ndarray<std::int32_t, 1>& empty_cluster_indices,
                                       const bk::event_vector& deps = {}) -> sycl::event {
    ONEDAL_PROFILER_TASK(fill_indices, queue);
    const std::int64_t cluster_count = counters.get_dimension(0);

    ONEDAL_ASSERT(cluster_count > 0);
    ONEDAL_ASSERT(candidate_count > 0);
    ONEDAL_ASSERT(candidate_count <= cluster_count);
    ONEDAL_ASSERT(empty_cluster_indices.get_dimension(0) >= candidate_count);

    const auto host_counters = counters.to_host(queue);
    const auto host_empty_cluster_indices = bk::make_unique_host<std::int32_t>(candidate_count);

    const std::int32_t* host_counters_ptr = host_counters.get_data();
    std::int32_t* host_empty_cluster_indices_ptr = host_empty_cluster_indices.get();

    std::int64_t counter = 0;
    for (std::int64_t i = 0; i < cluster_count; i++) {
        if (host_counters_ptr[i] > 0) {
            continue;
        }

        host_empty_cluster_indices_ptr[counter] = i;
        counter++;
    }

    // We have to wait as `host_counters` will be deleted once we leave scope of the function
    dal::backend::copy_host2usm(queue,
                                empty_cluster_indices.get_mutable_data(),
                                host_empty_cluster_indices_ptr,
                                candidate_count)
        .wait_and_throw();

    return sycl::event{};
}

template <typename Float>
static auto copy_candidates_from_data(sycl::queue& queue,
                                      const pr::ndview<Float, 2>& data,
                                      const centroid_candidates<Float>& candidates,
                                      pr::ndview<Float, 2>& centroids,
                                      const bk::event_vector& deps) -> sycl::event {
    ONEDAL_PROFILER_TASK(copy_candidates, queue);
    ONEDAL_ASSERT(data.get_dimension(0) >= candidates.get_candidate_count());
    ONEDAL_ASSERT(data.get_dimension(1) == centroids.get_dimension(1));
    ONEDAL_ASSERT(centroids.get_dimension(0) >= candidates.get_candidate_count());

    const std::int64_t candidate_count = candidates.get_candidate_count();
    const std::int64_t column_count = centroids.get_dimension(1);

    const Float* data_ptr = data.get_data();
    Float* centroids_ptr = centroids.get_mutable_data();
    const std::int32_t* candidate_indices_ptr = candidates.get_indices().get_data();
    const std::int32_t* empty_cluster_indices_ptr =
        candidates.get_empty_cluster_indices().get_data();

    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);

        const auto range = bk::make_range_2d(candidate_count, column_count);
        cgh.parallel_for(range, [=](sycl::id<2> id) {
            const std::int64_t i = id[0];
            const std::int64_t j = id[1];
            const std::int64_t dst_i = empty_cluster_indices_ptr[i];
            const std::int64_t src_i = candidate_indices_ptr[i];
            centroids_ptr[dst_i * column_count + j] = data_ptr[src_i * column_count + j];
        });
    });

    return event;
}

template <typename Float>
auto copy_candidates_from_data(sycl::queue& queue,
                               const pr::ndview<Float, 1>& values,
                               const pr::ndview<std::int64_t, 1>& column_indices,
                               const pr::ndview<std::int64_t, 1>& row_offsets,
                               const centroid_candidates<Float>& candidates,
                               pr::ndview<Float, 2>& centroids,
                               const bk::event_vector& deps) -> sycl::event {
    ONEDAL_PROFILER_TASK(copy_candidates, queue);
    ONEDAL_ASSERT(centroids.get_dimension(0) >= candidates.get_candidate_count());

    const std::int64_t column_count = centroids.get_dimension(1);
    const std::int64_t candidate_count = candidates.get_candidate_count();

    const auto* values_ptr = values.get_data();
    const auto* column_indices_ptr = column_indices.get_data();
    const auto* row_offsets_ptr = row_offsets.get_data();
    Float* centroids_ptr = centroids.get_mutable_data();
    const std::int32_t* candidate_indices_ptr = candidates.get_indices().get_data();
    const std::int32_t* empty_cluster_indices_ptr =
        candidates.get_empty_cluster_indices().get_data();

    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(sycl::range(candidate_count), [=](sycl::id<1> id) {
            const std::int64_t dst_i = empty_cluster_indices_ptr[id];
            const std::int64_t src_i = candidate_indices_ptr[id];
            const std::int64_t begin_idx = row_offsets_ptr[src_i];
            const std::int64_t end_idx = row_offsets_ptr[src_i + 1];
            for (std::int64_t j = 0; j < column_count; ++j) {
                centroids_ptr[dst_i * column_count + j] = Float(0.0);
            }
            for (std::int64_t j = begin_idx; j < end_idx; ++j) {
                centroids_ptr[dst_i * column_count + column_indices_ptr[j]] = values_ptr[j];
            }
        });
    });

    return event;
}

template <typename Float>
static auto gather_candidates(sycl::queue& queue,
                              const pr::ndview<Float, 2>& data,
                              const centroid_candidates<Float>& candidates,
                              pr::ndview<Float, 2>& gathered_candidates,
                              const bk::event_vector& deps) -> sycl::event {
    ONEDAL_PROFILER_TASK(gather_candidates, queue);
    ONEDAL_ASSERT(gathered_candidates.get_dimension(0) == candidates.get_candidate_count());
    ONEDAL_ASSERT(gathered_candidates.get_dimension(1) == data.get_dimension(1));

    const std::int64_t column_count = data.get_dimension(1);
    const std::int64_t candidate_count = candidates.get_candidate_count();

    const Float* data_ptr = data.get_data();
    const std::int32_t* candidate_indices_ptr = candidates.get_indices().get_data();
    Float* gathered_candidates_ptr = gathered_candidates.get_mutable_data();

    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);

        const auto range = bk::make_range_2d(candidate_count, column_count);
        cgh.parallel_for(range, [=](sycl::id<2> id) {
            const std::int64_t i = id[0];
            const std::int64_t j = id[1];
            const std::int64_t src_i = candidate_indices_ptr[i];
            gathered_candidates_ptr[i * column_count + j] = data_ptr[src_i * column_count + j];
        });
    });

    return event;
}

template <typename Float>
static auto scatter_candidates(sycl::queue& queue,
                               const pr::ndview<std::int32_t, 1>& empty_cluster_indices,
                               const pr::ndview<Float, 2>& candidates,
                               pr::ndview<Float, 2>& centroids,
                               const bk::event_vector& deps) -> sycl::event {
    ONEDAL_PROFILER_TASK(scatter_candidates, queue);
    ONEDAL_ASSERT(empty_cluster_indices.get_dimension(0) == candidates.get_dimension(0));
    ONEDAL_ASSERT(candidates.get_dimension(1) == centroids.get_dimension(1));

    const std::int64_t candidate_count = candidates.get_dimension(0);
    const std::int64_t column_count = candidates.get_dimension(1);

    const std::int32_t* empty_cluster_indices_ptr = empty_cluster_indices.get_data();
    const Float* candidates_ptr = candidates.get_data();
    Float* centroids_ptr = centroids.get_mutable_data();

    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);

        const auto range = bk::make_range_2d(candidate_count, column_count);
        cgh.parallel_for(range, [=](sycl::id<2> id) {
            const std::int64_t i = id[0];
            const std::int64_t j = id[1];
            const std::int64_t dst_i = empty_cluster_indices_ptr[i];
            centroids_ptr[dst_i * column_count + j] = candidates_ptr[i * column_count + j];
        });
    });

    return event;
}

template <typename Float>
static auto reduce_candidates(sycl::queue& queue,
                              const bk::communicator<spmd::device_memory_access::usm>& comm,
                              std::int64_t candidate_count,
                              pr::ndarray<Float, 1>& distances,
                              pr::ndarray<Float, 2>& candidates,
                              const bk::event_vector& deps = {}) -> sycl::event {
    ONEDAL_PROFILER_TASK(reduce_candidates, queue);
    const std::int64_t column_count = candidates.get_dimension(1);

    ONEDAL_ASSERT(candidate_count > 0);
    ONEDAL_ASSERT(column_count > 0);
    ONEDAL_ASSERT(candidates.get_dimension(0) == candidate_count);
    ONEDAL_ASSERT(distances.get_dimension(0) == candidate_count);

    const std::int64_t all_candidate_count =
        dal::detail::check_mul_overflow(comm.get_rank_count(), candidate_count);

    // Allgather candidates
    const auto host_candidates = candidates.to_host(queue, deps);
    auto host_all_candidates = pr::ndarray<Float, 3>::empty({ comm.get_rank_count(), //
                                                              candidate_count,
                                                              column_count });

    dal::backend::communicator_event candidates_reduce_event;
    {
        ONEDAL_PROFILER_TASK(allgather_host_candidates);
        candidates_reduce_event = comm.allgather(host_candidates.flatten(), //
                                                 host_all_candidates.flatten());
    }

    // Allgather distances
    const auto host_distances = distances.to_host(queue);
    auto host_all_distances = pr::ndarray<Float, 2>::empty({ comm.get_rank_count(), //
                                                             candidate_count });

    dal::backend::communicator_event distances_reduce_event;
    {
        ONEDAL_PROFILER_TASK(algather_host_distances);
        distances_reduce_event = comm.allgather(host_distances.flatten(), //
                                                host_all_distances.flatten());
    }

    auto host_all_indices = bk::make_unique_host<std::int32_t>(all_candidate_count);
    {
        std::int32_t* host_all_indices_ptr = host_all_indices.get();
        for (std::int64_t i = 0; i < all_candidate_count; i++) {
            host_all_indices_ptr[i] = i;
        }
    }

    candidates_reduce_event.wait();
    distances_reduce_event.wait();

    {
        ONEDAL_ASSERT(candidate_count <= all_candidate_count);
        std::int32_t* host_all_indices_ptr = host_all_indices.get();
        const Float* host_all_distances_ptr = host_all_distances.get_data();

        std::partial_sort(host_all_indices_ptr,
                          host_all_indices_ptr + candidate_count,
                          host_all_indices_ptr + all_candidate_count,
                          [=](std::int32_t i, std::int32_t j) {
                              return host_all_distances_ptr[i] > host_all_distances_ptr[j];
                          });

        if (candidate_count >= 2) {
            ONEDAL_ASSERT(host_all_distances_ptr[host_all_indices_ptr[0]] >=
                          host_all_distances_ptr[host_all_indices_ptr[1]]);
        }
    }

    {
        const Float* host_all_candidates_ptr = host_all_candidates.get_data();
        const Float* host_all_distances_ptr = host_all_distances.get_data();
        const std::int32_t* host_all_indices_ptr = host_all_indices.get();
        Float* host_distances_ptr = host_distances.get_mutable_data();
        Float* host_candidates_ptr = host_candidates.get_mutable_data();

        for (std::int64_t i = 0; i < candidate_count; i++) {
            const std::int64_t src_i = host_all_indices_ptr[i];
            host_distances_ptr[i] = host_all_distances_ptr[src_i];
            bk::copy(host_candidates_ptr + i * column_count,
                     host_all_candidates_ptr + src_i * column_count,
                     column_count);
        }
    }

    {
        auto distances_copy_event = dal::backend::copy_host2usm( //
            queue,
            distances.get_mutable_data(),
            host_distances.get_data(),
            candidate_count);

        auto candidates_copy_event = dal::backend::copy_host2usm( //
            queue,
            candidates.get_mutable_data(),
            host_candidates.get_data(),
            candidate_count * column_count);

        sycl::event::wait({ distances_copy_event, candidates_copy_event });
    }

    return sycl::event{};
}

template <typename Float>
static auto gather_scatter_candidates(sycl::queue& queue,
                                      bk::communicator<spmd::device_memory_access::usm>& comm,
                                      const pr::ndview<Float, 2>& data,
                                      const centroid_candidates<Float>& candidates,
                                      pr::ndview<Float, 2>& centroids,
                                      const bk::event_vector& deps) -> sycl::event {
    ONEDAL_PROFILER_TASK(gather_scatter_candidates, queue);
    const std::int64_t column_count = data.get_dimension(1);
    const std::int64_t candidate_count = candidates.get_candidate_count();

    auto gathered_candidates = pr::ndarray<Float, 2>::empty( //
        queue,
        { candidate_count, column_count },
        sycl::usm::alloc::device);

    auto gather_event = gather_candidates(queue, data, candidates, gathered_candidates, deps);
    auto candidate_distances = candidates.get_distances();
    auto reduce_event = reduce_candidates(queue,
                                          comm,
                                          candidate_count,
                                          candidate_distances,
                                          gathered_candidates,
                                          { gather_event });

    auto scatter_event = scatter_candidates(queue,
                                            candidates.get_empty_cluster_indices(),
                                            gathered_candidates,
                                            centroids,
                                            { reduce_event });

    // We need to wait as `gathered_candidates` will be deallocated
    // as we leave scope of the function
    scatter_event.wait_and_throw();

    return scatter_event;
}

template <typename Float>
auto find_candidates(sycl::queue& queue,
                     std::int64_t candidate_count,
                     const pr::ndarray<Float, 2>& closest_distances,
                     const pr::ndarray<std::int32_t, 1>& counters,
                     const bk::event_vector& deps)
    -> std::tuple<centroid_candidates<Float>, sycl::event> {
    ONEDAL_PROFILER_TASK(find_candidates, queue);
    ONEDAL_ASSERT(closest_distances.get_dimension(0) >= candidate_count);
    ONEDAL_ASSERT(closest_distances.get_dimension(1) == 1);
    ONEDAL_ASSERT(counters.get_dimension(0) >= candidate_count);

    constexpr auto alloc = sycl::usm::alloc::device;
    const auto shape = pr::ndshape<1>{ candidate_count };
    auto candidate_indices = pr::ndarray<std::int32_t, 1>::empty(queue, shape, alloc);
    auto candidate_distances = pr::ndarray<Float, 1>::empty(queue, shape, alloc);
    auto empty_cluster_indices = pr::ndarray<std::int32_t, 1>::empty(queue, shape, alloc);

    auto fill_indices_and_distances_event =
        fill_candidate_indices_and_distances(queue,
                                             candidate_count,
                                             closest_distances,
                                             candidate_indices,
                                             candidate_distances,
                                             deps);

    auto fill_empty_cluster_indices_event =
        fill_empty_cluster_indices(queue,
                                   candidate_count,
                                   counters,
                                   empty_cluster_indices,
                                   { fill_indices_and_distances_event });

    centroid_candidates<Float> candidates{ candidate_indices,
                                           candidate_distances,
                                           empty_cluster_indices };

    return { candidates, fill_empty_cluster_indices_event };
}

template <typename Float>
auto fill_empty_clusters(sycl::queue& queue,
                         bk::communicator<spmd::device_memory_access::usm>& comm,
                         const pr::ndview<Float, 2>& data,
                         const centroid_candidates<Float>& candidates,
                         pr::ndview<Float, 2>& centroids,
                         const bk::event_vector& deps) -> sycl::event {
    if (comm.is_distributed()) {
        return gather_scatter_candidates(queue, comm, data, candidates, centroids, deps);
    }
    else {
        return copy_candidates_from_data(queue, data, candidates, centroids, deps);
    }
}

#define INSTANTIATE(Float)                                                                     \
    template auto find_candidates(sycl::queue& queue,                                          \
                                  std::int64_t candidate_count,                                \
                                  const pr::ndarray<Float, 2>& closest_distances,              \
                                  const pr::ndarray<std::int32_t, 1>& counters,                \
                                  const bk::event_vector& deps)                                \
        ->std::tuple<centroid_candidates<Float>, sycl::event>;                                 \
                                                                                               \
    template auto fill_empty_clusters(sycl::queue& queue,                                      \
                                      bk::communicator<spmd::device_memory_access::usm>& comm, \
                                      const pr::ndview<Float, 2>& data,                        \
                                      const centroid_candidates<Float>& candidates,            \
                                      pr::ndview<Float, 2>& centroids,                         \
                                      const bk::event_vector& deps)                            \
        ->sycl::event;                                                                         \
    template auto copy_candidates_from_data(sycl::queue& queue,                                \
                                            const pr::ndview<Float, 1>& values,                \
                                            const pr::ndview<std::int64_t, 1>& column_indices, \
                                            const pr::ndview<std::int64_t, 1>& row_offsets,    \
                                            const centroid_candidates<Float>& candidates,      \
                                            pr::ndview<Float, 2>& centroids,                   \
                                            const bk::event_vector& deps)                      \
        ->sycl::event;

INSTANTIATE(float)
INSTANTIATE(double)

} // namespace oneapi::dal::kmeans::backend
