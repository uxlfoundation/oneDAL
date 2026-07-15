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

#include "oneapi/dal/algo/kmeans/backend/gpu/kernels_fp.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/sort.hpp"
#include "oneapi/dal/detail/profiler.hpp"
namespace oneapi::dal::kmeans::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

std::int64_t get_max_block_size_in_bytes(const sycl::queue& queue) {
    constexpr std::int64_t mem_block_size_limit_ratio = 4; // To ensure all blocks fit in memory
    const std::int64_t max_block_size_in_bytes =
        std::min(bk::device_max_mem_alloc_size(queue),
                 bk::device_global_mem_size(queue) / mem_block_size_limit_ratio);
    return max_block_size_in_bytes;
}

bool can_use_cache_for_distance_matrix(const sycl::queue& queue,
                                       std::int64_t cache_size_in_bytes,
                                       std::int64_t column_count) {
    // TODO optimization/dispatching
    constexpr std::int64_t effective_cache_column_count_limit = 256;
    bool use_cache = column_count < effective_cache_column_count_limit;
    return use_cache;
}

template <typename Float>
std::int64_t propose_block_size(const sycl::queue& q, const std::int64_t r) {
    constexpr std::int64_t fsize = sizeof(Float);
    return 0x10000l * (8 / fsize);
}

inline std::int64_t get_recommended_sg_size(const sycl::queue& queue) {
    // TODO optimization/dispatching
    return 16;
}

inline std::int64_t get_recommended_wg_count(sycl::queue& queue) {
    // TODO optimization/dispatching
    return 128;
}

inline std::int64_t get_gpu_sg_size(sycl::queue& queue) {
    // TODO optimization/dispatching
    return 16;
}

template <typename T>
struct select_min_distance {};

template <typename T>
struct set_indices {};

template <typename T>
struct centroid_reduction {};

template <typename T>
struct reset_partial_centroids {};

template <typename T>
struct centroid_merge {};

template <typename T>
struct compute_obj_function {};

template <typename T>
struct complete_distances {};

template <typename Float>
std::int64_t kernels_fp<Float>::get_block_size_in_rows(sycl::queue& queue,
                                                       std::int64_t column_count,
                                                       std::int64_t cluster_count) {
    std::int64_t block_size_in_bytes = bk::device_global_mem_size(queue);
    bool use_cache = can_use_cache_for_distance_matrix(queue, block_size_in_bytes, column_count);
    if (!use_cache) {
        const auto max_block_size_in_bytes = get_max_block_size_in_bytes(queue);
        const std::int64_t max_width = std::max(column_count, cluster_count);
        std::int64_t block_size_in_rows = max_block_size_in_bytes / max_width / sizeof(Float);
        ONEDAL_ASSERT(block_size_in_rows > 0);
        return block_size_in_rows;
    }
    const std::int64_t block_size_in_rows = block_size_in_bytes / column_count / sizeof(Float);
    ONEDAL_ASSERT(block_size_in_rows > 0);
    return block_size_in_rows;
}

template <typename Float>
std::int64_t kernels_fp<Float>::get_part_count_for_partial_centroids(sycl::queue& queue,
                                                                     std::int64_t column_count,
                                                                     std::int64_t cluster_count) {
    // TODO optimization
    const std::int64_t block_size_in_bytes = get_max_block_size_in_bytes(queue);
    std::int64_t part_count = 128; // Number of partial centroids. Reasonable initial guess.
    dal::detail::check_mul_overflow(cluster_count, column_count);
    dal::detail::check_mul_overflow(cluster_count * column_count, part_count);
    std::int64_t fp_size = dal::detail::get_data_type_size(dal::detail::make_data_type<Float>());
    dal::detail::check_mul_overflow(cluster_count * column_count * part_count, fp_size);
    const std::int64_t part_size = cluster_count * column_count * fp_size;
    while (part_count * part_size > block_size_in_bytes / 2) {
        part_count /= 2;
    }
    ONEDAL_ASSERT(part_count > 0);
    return part_count;
}

template <typename Float>
sycl::event kernels_fp<Float>::select(sycl::queue& queue,
                                      const pr::ndview<Float, 2>& distances,
                                      const pr::ndview<Float, 1>& centroid_squares,
                                      pr::ndview<Float, 2>& selection,
                                      pr::ndview<std::int32_t, 2>& indices,
                                      const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(select_min_distance, queue);
    ONEDAL_ASSERT(indices.get_dimension(0) == distances.get_dimension(0));
    ONEDAL_ASSERT(indices.get_dimension(1) == 1);
    ONEDAL_ASSERT(selection.get_dimension(0) == distances.get_dimension(0));
    ONEDAL_ASSERT(selection.get_dimension(1) == 1);
    ONEDAL_ASSERT(centroid_squares.get_dimension(0) == distances.get_dimension(1));

    const std::int64_t cluster_count = distances.get_dimension(1);
    const std::int64_t row_count = distances.get_dimension(0);
    const std::int64_t stride = distances.get_dimension(1);

    const std::int64_t cluster_count_as_int32 =
        dal::detail::integral_cast<std::int32_t>(cluster_count);

    const std::int64_t preffered_wg_size = bk::device_max_wg_size(queue);
    const std::int64_t wg_size =
        bk::get_scaled_wg_size_per_row(queue, cluster_count, preffered_wg_size);
    dal::detail::check_mul_overflow(wg_size, stride);

    const Float* distances_ptr = distances.get_data();
    const Float* centroid_squares_ptr = centroid_squares.get_data();
    Float* selection_ptr = selection.get_mutable_data();
    std::int32_t* indices_ptr = indices.get_mutable_data();

    const auto block_size = propose_block_size<Float>(queue, row_count);
    const bk::uniform_blocking blocking(row_count, block_size);
    const auto sg_size = bk::device_max_sg_size(queue);

    const auto sg_count = wg_size / sg_size;

    std::vector<sycl::event> events(blocking.get_block_count());
    for (std::int64_t block_index = 0; block_index < blocking.get_block_count(); ++block_index) {
        const auto first_row = blocking.get_block_start_index(block_index);
        const auto last_row = blocking.get_block_end_index(block_index);
        const auto curr_block = last_row - first_row;
        ONEDAL_ASSERT(curr_block > 0);

        auto event = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.parallel_for<select_min_distance<Float>>(
                bk::make_multiple_nd_range_2d({ curr_block, wg_size }, { sg_count, sg_size }),
                [=](sycl::nd_item<2> item) {
                    constexpr sycl::minimum<Float> minimum_val;
                    constexpr sycl::minimum<std::int32_t> minimum_idx;

                    const std::int64_t row = item.get_global_id(0) + first_row;

                    if (last_row <= row)
                        return;
                    auto sg = item.get_sub_group();
                    std::int32_t local_id = item.get_local_id(1);
                    auto min_val = std::numeric_limits<Float>::max();
                    auto min_idx = std::numeric_limits<std::int32_t>::max();

                    const auto* const row_ptr = distances_ptr + row * stride;
                    for (std::int32_t col = local_id; col < cluster_count_as_int32; ++col) {
                        const Float cur_val = row_ptr[col] + centroid_squares_ptr[col];
                        const bool handle = cur_val < min_val;
                        min_val = handle ? cur_val : min_val;
                        min_idx = handle ? col : min_idx;
                    }

                    sycl::group_barrier(sg);

                    const auto final_min_val = sycl::reduce_over_group(sg, min_val, minimum_val);
                    const auto handle = (min_val == final_min_val)
                                            ? min_idx
                                            : std::numeric_limits<std::int32_t>::max();
                    const auto final_min_idx = sycl::reduce_over_group(sg, handle, minimum_idx);

                    if (local_id == 0) {
                        indices_ptr[row] = final_min_idx;
                        selection_ptr[row] = final_min_val;
                    }
                });
        });

        events.push_back(event);
    }
    return bk::wait_or_pass(events);
}

template <typename Float>
sycl::event kernels_fp<Float>::assign_clusters(sycl::queue& queue,
                                               const pr::ndview<Float, 2>& data,
                                               const pr::ndview<Float, 2>& centroids,
                                               const pr::ndview<Float, 1>& data_squares,
                                               const pr::ndview<Float, 1>& centroid_squares,
                                               std::int64_t block_size_in_rows,
                                               pr::ndview<std::int32_t, 2>& responses,
                                               pr::ndview<Float, 2>& distances,
                                               pr::ndview<Float, 2>& closest_distances,
                                               const bk::event_vector& deps) {
    ONEDAL_ASSERT(data.get_dimension(1) == centroids.get_dimension(1));
    ONEDAL_ASSERT(responses.get_dimension(0) >= data.get_dimension(0));
    ONEDAL_ASSERT(responses.get_dimension(1) == 1);
    ONEDAL_ASSERT(closest_distances.get_dimension(0) >= data.get_dimension(0));
    ONEDAL_ASSERT(closest_distances.get_dimension(1) == 1);
    ONEDAL_ASSERT(distances.get_dimension(0) >= block_size_in_rows);
    ONEDAL_ASSERT(distances.get_dimension(1) >= centroids.get_dimension(0));
    ONEDAL_ASSERT(centroid_squares.get_dimension(0) == centroids.get_dimension(0));
    ONEDAL_ASSERT(data_squares.get_dimension(0) == data.get_dimension(0));
    sycl::event selection_event;
    const auto row_count = data.get_dimension(0);
    const auto column_count = data.get_dimension(1);
    const auto centroid_count = centroids.get_dimension(0);
    auto block_count = row_count / block_size_in_rows + bool(row_count % block_size_in_rows);
    for (std::int64_t iblock = 0; iblock < block_count; iblock++) {
        const auto row_offset = block_size_in_rows * iblock;
        auto cur_rows = std::min(block_size_in_rows, row_count - row_offset);
        auto distance_block =
            pr::ndview<Float, 2>::wrap(distances.get_mutable_data(), { cur_rows, centroid_count });
        auto data_block = pr::ndview<Float, 2>::wrap(data.get_data() + row_offset * column_count,
                                                     { cur_rows, column_count });
        sycl::event distance_event;
        {
            ONEDAL_PROFILER_TASK(gemm, queue);
            distance_event = pr::gemm(queue,
                                      data_block,
                                      centroids.t(),
                                      distance_block,
                                      Float(-2.0),
                                      Float(0.0),
                                      { selection_event });
        }
        auto response_block =
            pr::ndview<std::int32_t, 2>::wrap(responses.get_mutable_data() + row_offset,
                                              { cur_rows, 1 });
        auto closest_distance_block =
            pr::ndview<Float, 2>::wrap(closest_distances.get_mutable_data() + row_offset,
                                       { cur_rows, 1 });
        selection_event = select(queue,
                                 distance_block,
                                 centroid_squares,
                                 closest_distance_block,
                                 response_block,
                                 { distance_event });
    }
    auto completion_event =
        complete_closest_distances(queue, data_squares, closest_distances, { selection_event });
    return completion_event;
}

template <typename Float>
sycl::event kernels_fp<Float>::merge_reduce_centroids(sycl::queue& queue,
                                                      const pr::ndview<std::int32_t, 1>& counters,
                                                      const pr::ndview<Float, 2>& partial_centroids,
                                                      std::int64_t part_count,
                                                      pr::ndview<Float, 2>& centroids,
                                                      const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(merge_reduce_centroids, queue);
    ONEDAL_ASSERT(partial_centroids.get_dimension(0) == centroids.get_dimension(0) * part_count);
    ONEDAL_ASSERT(partial_centroids.get_dimension(1) == centroids.get_dimension(1));
    ONEDAL_ASSERT(counters.get_dimension(0) == centroids.get_dimension(0));
    const Float* partial_centroids_ptr = partial_centroids.get_data();
    Float* centroids_ptr = centroids.get_mutable_data();
    const std::int32_t* counters_ptr = counters.get_data();
    const auto column_count = centroids.get_dimension(1);
    const auto cluster_count = centroids.get_dimension(0);
    const auto sg_size_to_set = get_gpu_sg_size(queue);

    return queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for<centroid_merge<Float>>(
            bk::make_multiple_nd_range_2d({ sg_size_to_set, column_count * cluster_count },
                                          { sg_size_to_set, 1 }),
            [=](sycl::nd_item<2> item) {
                auto sg = item.get_sub_group();
                const std::int64_t sg_id = sg.get_group_id()[0];
                const std::int64_t wg_id = item.get_global_id(1);
                const std::int64_t sg_count = sg.get_group_range()[0];
                const std::int64_t sg_global_id = wg_id * sg_count + sg_id;
                if (sg_global_id >= column_count * cluster_count)
                    return;
                const std::int64_t sg_cluster_id = sg_global_id / column_count;
                const std::int64_t local_id = sg.get_local_id()[0];
                const std::int64_t local_range = sg.get_local_range()[0];
                Float sum = 0.0;
                for (std::int64_t i = local_id; i < part_count; i += local_range) {
                    sum += partial_centroids_ptr[i * cluster_count * column_count + sg_global_id];
                }
                sum = sycl::reduce_over_group(sg, sum, sycl::plus<Float>());

                if (local_id == 0) {
                    auto count = counters_ptr[sg_cluster_id];
                    if (count > 0) {
                        centroids_ptr[sg_global_id] = sum / count;
                    }
                }
            });
    });
}

template <typename Float>
sycl::event kernels_fp<Float>::partial_reduce_centroids(
    sycl::queue& queue,
    const pr::ndview<Float, 2>& data,
    const pr::ndview<std::int32_t, 2>& responses,
    std::int64_t cluster_count,
    std::int64_t part_count,
    pr::ndview<Float, 2>& partial_centroids,
    const bk::event_vector& deps) {
    //  TODO: Enable the task below
    //  ONEDAL_PROFILER_TASK(partial_reduce_centroids, queue);
    ONEDAL_ASSERT(data.get_dimension(1) == partial_centroids.get_dimension(1));
    ONEDAL_ASSERT(partial_centroids.get_dimension(0) == cluster_count * part_count);
    ONEDAL_ASSERT(responses.get_dimension(0) == data.get_dimension(0));
    ONEDAL_ASSERT(responses.get_dimension(1) == 1);
    const Float* data_ptr = data.get_data();
    const std::int32_t* response_ptr = responses.get_data();
    Float* partial_centroids_ptr = partial_centroids.get_mutable_data();
    const auto row_count = data.get_dimension(0);
    const auto column_count = data.get_dimension(1);
    const auto sg_size_to_set = get_recommended_sg_size(queue);
    return queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for<centroid_reduction<Float>>(
            bk::make_multiple_nd_range_2d({ sg_size_to_set, part_count }, { sg_size_to_set, 1 }),
            [=](sycl::nd_item<2> item) {
                auto sg = item.get_sub_group();
                const std::int64_t sg_id = sg.get_group_id()[0];
                const std::int64_t wg_id = item.get_global_id(1);
                const std::int64_t sg_count = sg.get_group_range()[0];
                const std::int64_t sg_global_id = wg_id * sg_count + sg_id;
                if (sg_global_id >= part_count)
                    return;
                const std::int64_t local_id = sg.get_local_id()[0];
                const std::int64_t local_range = sg.get_local_range()[0];
                for (std::int64_t i = sg_global_id; i < row_count; i += part_count) {
                    std::int32_t cl = -1;
                    if (local_id == 0) {
                        cl = response_ptr[i];
                    }
                    cl =
                        sycl::reduce_over_group(sg, cl, sycl::maximum<std::int32_t>());
                    for (std::int64_t j = local_id; j < column_count; j += local_range) {
                        partial_centroids_ptr[sg_global_id * cluster_count * column_count +
                                              cl * column_count + j] +=
                            data_ptr[i * column_count + j];
                    }
                }
            });
    });
}

template <typename Float>
sycl::event kernels_fp<Float>::compute_objective_function(
    sycl::queue& queue,
    const pr::ndview<Float, 2>& closest_distances,
    pr::ndview<Float, 1>& objective_function,
    const bk::event_vector& deps) {
    constexpr pr::sum<Float> binary;
    constexpr pr::identity<Float> unary;

    ONEDAL_PROFILER_TASK(compute_objective_function, queue);
    ONEDAL_ASSERT(closest_distances.get_dimension(1) == 1);
    ONEDAL_ASSERT(objective_function.get_dimension(0) == 1);

    const auto element_count = closest_distances.get_count();
    const auto closest_distances_1d = closest_distances.template reshape<1>({ element_count });
    auto fill_event = pr::fill(queue, objective_function, Float(0), deps);
    return pr::reduce_1d(queue,
                         closest_distances_1d,
                         objective_function,
                         binary,
                         unary,
                         { fill_event });
}

template <typename Float>
sycl::event kernels_fp<Float>::compute_squares(sycl::queue& queue,
                                               const pr::ndview<Float, 2>& data,
                                               pr::ndview<Float, 1>& squares,
                                               const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(compute_squares, queue);
    ONEDAL_ASSERT(data.get_dimension(0) == squares.get_dimension(0));

    constexpr pr::sum<Float> binary;
    constexpr pr::square<Float> unary;

    return pr::reduce_by_rows(queue, data, squares, binary, unary, deps);
}

template <typename Float>
sycl::event kernels_fp<Float>::complete_closest_distances(sycl::queue& queue,
                                                          const pr::ndview<Float, 1>& data_squares,
                                                          pr::ndview<Float, 2>& closest_distances,
                                                          const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(complete_closest_distances, queue);
    ONEDAL_ASSERT(data_squares.get_dimension(0) == closest_distances.get_dimension(0));
    ONEDAL_ASSERT(closest_distances.get_dimension(1) == 1);

    const auto elem_count = closest_distances.get_dimension(0);
    auto values_ptr = closest_distances.get_mutable_data();
    const auto squares_ptr = data_squares.get_data();

    auto complete_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for<complete_distances<Float>>(
            sycl::range<1>(elem_count),
            [=](sycl::id<1> idx) {
                Float val = values_ptr[idx] + squares_ptr[idx];
                values_ptr[idx] = val < Float(0) ? Float(0) : val;
            });
    });
    return complete_event;
}
#endif

} // namespace oneapi::dal::kmeans::backend
