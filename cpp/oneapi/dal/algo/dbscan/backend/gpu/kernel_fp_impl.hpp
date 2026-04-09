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

#include "oneapi/dal/algo/dbscan/backend/gpu/kernels_fp.hpp"
#include "oneapi/dal/detail/profiler.hpp"

namespace oneapi::dal::dbscan::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

inline std::int64_t get_recommended_wg_size(const sycl::queue& queue,
                                            std::int64_t column_count = 0) {
    // TODO optimization/dispatching
    return column_count > 32 ? 32 : 16;
}

inline std::int64_t get_recommended_check_block_size(const sycl::queue& queue,
                                                     std::int64_t column_count = 0,
                                                     std::int64_t wg_size = 0) {
    // TODO optimiztion/dispatching for cases with column_count > 2 * wg_size
    return 1;
}

inline std::int64_t get_recommended_min_width(const sycl::queue& queue,
                                              std::int64_t column_count = 0,
                                              std::int64_t wg_size = 0) {
    // This min_width values has been computed via experiments,
    // it shows the number of columns where the subgroups usage is effective
    return 4;
}

///  A struct that finds the core points via subgroups
///
/// @tparam Float        Floating-point type used to perform computations
/// @tparam use_weights  Bool type used to check that weights are enabled
///
/// @param[in]  queue            The SYCL queue
/// @param[in]  data             The input data of size `row_count` x `column_count`
/// @param[in]  weights          The input weights of size `row_count` x `1`
/// @param[in]  cores            The current cores of size `row_count` x `1`
/// @param[in]  neighbours       The current neighbours of size `row_count` x `1`
///                              it contains the counter of neighbours for each point
/// @param[in]  epsilon          The input parameter epsilon
/// @param[in]  min_observations The input parameter min_observation
/// @param[in]  deps             Events indicating availability of the `data` for reading or writing
///
/// @return A SYCL event indicating the availability
/// of the updated arrays(cores and neighbours) for reading and writing
template <typename Float, bool use_weights>
struct get_core_wide_kernel {
    static auto run(sycl::queue& queue,
                    const pr::ndview<Float, 2>& data,
                    const pr::ndview<Float, 2>& weights,
                    pr::ndview<std::int32_t, 1>& cores,
                    pr::ndview<Float, 1>& neighbours,
                    Float epsilon,
                    std::int64_t min_observations,
                    const bk::event_vector& deps) {
        const std::int64_t local_row_count = data.get_dimension(0);
        const std::int64_t column_count = data.get_dimension(1);

        ONEDAL_ASSERT(local_row_count > 0);
        ONEDAL_ASSERT(!use_weights || weights.get_dimension(0) == local_row_count);
        ONEDAL_ASSERT(!use_weights || weights.get_dimension(1) == 1);
        ONEDAL_ASSERT(cores.get_dimension(0) == local_row_count);
        ONEDAL_ASSERT(neighbours.get_dimension(0) == local_row_count);

        const Float* data_ptr = data.get_data();
        const Float* weights_ptr = weights.get_data();
        std::int32_t* cores_ptr = cores.get_mutable_data();
        Float* neighbours_ptr = neighbours.get_mutable_data();

        auto event = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            const std::int64_t wg_size = get_recommended_wg_size(queue, column_count);
            const std::int64_t block_split_size =
                get_recommended_check_block_size(queue, column_count, wg_size);
            cgh.parallel_for(
                bk::make_multiple_nd_range_2d({ wg_size, local_row_count }, { wg_size, 1 }),
                [=](sycl::nd_item<2> item) {
                    auto sg = item.get_sub_group();
                    const std::uint32_t sg_id = sg.get_group_id()[0];
                    if (sg_id > 0)
                        return;

                    const std::uint32_t wg_id = item.get_global_id(1);
                    if (wg_id >= local_row_count)
                        return;

                    const std::uint32_t local_id = sg.get_local_id();
                    const std::uint32_t local_size = sg.get_local_range()[0];

                    Float count = neighbours_ptr[wg_id];
                    for (std::int64_t j = 0; j < local_row_count; j++) {
                        Float sum = Float(0);
                        std::int64_t count_iter = 0;
                        for (std::int64_t i = local_id; i < column_count; i += local_size) {
                            count_iter++;
                            Float val =
                                data_ptr[wg_id * column_count + i] - data_ptr[j * column_count + i];
                            sum += val * val;

                            if (count_iter % block_split_size == 0 &&
                                local_size * count_iter <= column_count) {
                                Float distance_check =
                                    sycl::reduce_over_group(sg,
                                                            sum,
                                                            sycl::ext::oneapi::plus<Float>());
                                if (distance_check > epsilon) {
                                    break;
                                }
                            }
                        }
                        Float distance =
                            sycl::reduce_over_group(sg, sum, sycl::ext::oneapi::plus<Float>());
                        if (distance <= epsilon) {
                            count += use_weights ? weights_ptr[j] : Float(1);
                            if (local_id == 0) {
                                neighbours_ptr[wg_id] = count;
                            }
                            if (count >= min_observations && !use_weights) {
                                if (local_id == 0) {
                                    cores_ptr[wg_id] = Float(1);
                                }
                                break;
                            }
                        }
                    }
                    if (neighbours_ptr[wg_id] >= min_observations) {
                        cores_ptr[wg_id] = Float(1);
                    }
                });
        });
        return event;
    }
};

///  A struct that finds the core points without subgroups
///  it is effective only on narrow cases. The column count of narrow cases < 4.
///
/// @tparam Float        Floating-point type used to perform computations
/// @tparam use_weights  Bool type used to check that weights are enabled
///
/// @param[in]  queue             The SYCL queue
/// @param[in]  data              The input data of size `row_count` x `column_count`
/// @param[in]  weights           The input weights of size `row_count` x `1`
/// @param[in]  cores             The current cores of size `row_count` x `1`
/// @param[in]  neighbours        The current neighbours of size `row_count` x `1`
///                               it contains the counter of neighbours for each point
/// @param[in]  epsilon           The input parameter epsilon
/// @param[in]  min_observations  The input parameter min_observation
/// @param[in]  deps              Events indicating availability of the `data` for reading or writing
///
/// @return A SYCL event indicating the availability
/// of the updated arrays(cores and neighbours) for reading and writing
template <typename Float, bool use_weights>
struct get_core_narrow_kernel {
    static auto run(sycl::queue& queue,
                    const pr::ndview<Float, 2>& data,
                    const pr::ndview<Float, 2>& weights,
                    pr::ndview<std::int32_t, 1>& cores,
                    pr::ndview<Float, 1>& neighbours,
                    Float epsilon,
                    std::int64_t min_observations,
                    const bk::event_vector& deps) {
        const std::int64_t local_row_count = data.get_dimension(0);
        const std::int64_t column_count = data.get_dimension(1);

        ONEDAL_ASSERT(local_row_count > 0);
        ONEDAL_ASSERT(!use_weights || weights.get_dimension(0) == local_row_count);
        ONEDAL_ASSERT(!use_weights || weights.get_dimension(1) == 1);

        ONEDAL_ASSERT(cores.get_dimension(0) == local_row_count);
        ONEDAL_ASSERT(neighbours.get_dimension(0) == local_row_count);

        const Float* data_ptr = data.get_data();
        const Float* weights_ptr = weights.get_data();
        std::int32_t* cores_ptr = cores.get_mutable_data();
        Float* neighbours_ptr = neighbours.get_mutable_data();

        auto event = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.parallel_for(sycl::range<1>{ std::size_t(local_row_count) }, [=](sycl::id<1> idx) {
                for (std::int64_t j = 0; j < local_row_count; j++) {
                    Float sum = 0.0;
                    for (std::int64_t i = 0; i < column_count; i++) {
                        Float val =
                            data_ptr[idx * column_count + i] - data_ptr[j * column_count + i];
                        sum += val * val;
                    }
                    if (sum > epsilon) {
                        continue;
                    }
                    neighbours_ptr[idx] += use_weights ? weights_ptr[j] : Float(1);

                    if (neighbours_ptr[idx] >= min_observations && !use_weights) {
                        cores_ptr[idx] = Float(1);
                        break;
                    }
                }
                //It is necesasry to check in cases with weights, due to weights could be negative
                if (neighbours_ptr[idx] >= min_observations) {
                    cores_ptr[idx] = Float(1);
                }
            });
        });
        return event;
    }
};

///  A struct that finds the core points via subgroups
///  on sendrecv_replaced data. It means that this function tries to
///  update current rank cores and neighbours arrays with another rank data
///
/// @tparam Float        Floating-point type used to perform computations
/// @tparam use_weights  Bool type used to check that weights are enabled
///
/// @param[in]  queue             The SYCL queue
/// @param[in]  data              The input data of size `row_count` x `column_count`
/// @param[in]  data_replace      The input data from another rank of size `row_count` x `column_count`
/// @param[in]  weights           The input weights of size `row_count` x `1`
/// @param[in]  cores             The current cores of size `row_count` x `1`
/// @param[in]  neighbours        The current neighbours of size `row_count` x `1`
///                               it contains the counter of neighbours for each point
/// @param[in]  epsilon           The input parameter epsilon
/// @param[in]  min_observations  The input parameter min_observation
/// @param[in]  deps              Events indicating availability of the `data` for reading or writing
///
/// @return A SYCL event indicating the availability
/// of the updated arrays(cores and neighbours) for reading and writing
template <typename Float, bool use_weights>
struct get_core_send_recv_replace_wide_kernel {
    static auto run(sycl::queue& queue,
                    const pr::ndview<Float, 2>& data,
                    const pr::ndview<Float, 2>& data_replace,
                    const pr::ndview<Float, 2>& weights,
                    pr::ndview<std::int32_t, 1>& cores,
                    pr::ndview<Float, 1>& neighbours,
                    Float epsilon,
                    std::int64_t min_observations,
                    const bk::event_vector& deps) {
        const std::int64_t local_row_count = data.get_dimension(0);
        const std::int64_t row_count_replace = data_replace.get_dimension(0);
        const std::int64_t column_count = data.get_dimension(1);

        ONEDAL_ASSERT(local_row_count > 0);
        ONEDAL_ASSERT(row_count_replace > 0);
        ONEDAL_ASSERT(!use_weights || weights.get_dimension(0) == local_row_count);
        ONEDAL_ASSERT(!use_weights || weights.get_dimension(1) == 1);

        ONEDAL_ASSERT(cores.get_dimension(0) == local_row_count);

        const Float* data_ptr = data.get_data();
        const Float* data_replace_ptr = data_replace.get_data();
        const Float* weights_ptr = weights.get_data();
        std::int32_t* cores_ptr = cores.get_mutable_data();
        Float* neighbours_ptr = neighbours.get_mutable_data();

        auto event = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            const std::int64_t wg_size = get_recommended_wg_size(queue, column_count);
            const std::int64_t block_split_size =
                get_recommended_check_block_size(queue, column_count, wg_size);
            cgh.parallel_for(
                bk::make_multiple_nd_range_2d({ wg_size, local_row_count }, { wg_size, 1 }),
                [=](sycl::nd_item<2> item) {
                    auto sg = item.get_sub_group();
                    const std::uint32_t sg_id = sg.get_group_id()[0];
                    if (sg_id > 0)
                        return;
                    const std::uint32_t wg_id = item.get_global_id(1);
                    if (wg_id >= local_row_count)
                        return;
                    const std::uint32_t local_id = sg.get_local_id();
                    const std::uint32_t local_size = sg.get_local_range()[0];

                    Float count = neighbours_ptr[wg_id];
                    for (std::int64_t j = 0; j < row_count_replace; j++) {
                        Float sum = Float(0);
                        std::int64_t count_iter = 0;
                        for (std::int64_t i = local_id; i < column_count; i += local_size) {
                            count_iter++;
                            Float val = data_ptr[wg_id * column_count + i] -
                                        data_replace_ptr[j * column_count + i];
                            sum += val * val;
                            if (count_iter % block_split_size == 0 &&
                                local_size * count_iter <= column_count) {
                                Float distance_check =
                                    sycl::reduce_over_group(sg,
                                                            sum,
                                                            sycl::ext::oneapi::plus<Float>());
                                if (distance_check > epsilon) {
                                    break;
                                }
                            }
                        }
                        Float distance =
                            sycl::reduce_over_group(sg, sum, sycl::ext::oneapi::plus<Float>());
                        if (distance <= epsilon) {
                            count += use_weights ? weights_ptr[j] : Float(1);
                            if (local_id == 0) {
                                neighbours_ptr[wg_id] = count;
                            }
                            if (count >= min_observations && !use_weights) {
                                if (local_id == 0) {
                                    cores_ptr[wg_id] = Float(1);
                                }
                                break;
                            }
                        }
                    }
                    if (neighbours_ptr[wg_id] >= min_observations) {
                        cores_ptr[wg_id] = Float(1);
                    }
                });
        });
        return event;
    }
};

///  A struct that finds the core points without subgroups
///  on sendrecv_replaced data. It means that this function tries to
///  update current rank cores and neighbours arrays with another rank data.
///  It is effective only on narrow cases. The column count of narrow cases < 4.
///
/// @tparam Float        Floating-point type used to perform computations
/// @tparam use_weights  Bool type used to check that weights are enabled
///
/// @param[in]  queue             The SYCL queue
/// @param[in]  data              The input data of size `row_count` x `column_count`
/// @param[in]  data_replace      The input data from another rank of size `row_count` x `column_count`
/// @param[in]  weights           The input weights of size `row_count` x `1`
/// @param[in]  cores             The current cores of size `row_count` x `1`
/// @param[in]  neighbours        The current neighbours of size `row_count` x `1`
///                               it contains the counter of neighbours for each point
/// @param[in]  epsilon           The input parameter epsilon
/// @param[in]  min_observations  The input parameter min_observation
/// @param[in]  deps              Events indicating availability of the `data` for reading or writing
///
/// @return A SYCL event indicating the availability
/// of the updated arrays(cores and neighbours) for reading and writing
template <typename Float, bool use_weights>
struct get_core_send_recv_replace_narrow_kernel {
    static auto run(sycl::queue& queue,
                    const pr::ndview<Float, 2>& data,
                    const pr::ndview<Float, 2>& data_replace,
                    const pr::ndview<Float, 2>& weights,
                    pr::ndview<std::int32_t, 1>& cores,
                    pr::ndview<Float, 1>& neighbours,
                    Float epsilon,
                    std::int64_t min_observations,
                    const bk::event_vector& deps) {
        const auto local_row_count = data.get_dimension(0);
        const auto row_count_replace = data_replace.get_dimension(0);
        ONEDAL_ASSERT(local_row_count > 0);
        ONEDAL_ASSERT(!use_weights || weights.get_dimension(0) == local_row_count);
        ONEDAL_ASSERT(!use_weights || weights.get_dimension(1) == 1);

        ONEDAL_ASSERT(cores.get_dimension(0) >= local_row_count);
        const std::int64_t column_count = data.get_dimension(1);

        const Float* data_ptr = data.get_data();
        const Float* data_replace_ptr = data_replace.get_data();
        const Float* weights_ptr = weights.get_data();
        std::int32_t* cores_ptr = cores.get_mutable_data();
        Float* neighbours_ptr = neighbours.get_mutable_data();

        auto event = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.parallel_for(sycl::range<1>{ std::size_t(local_row_count) }, [=](sycl::id<1> idx) {
                for (std::int64_t j = 0; j < row_count_replace; j++) {
                    Float sum = 0.0;
                    for (std::int64_t i = 0; i < column_count; i++) {
                        Float val = data_ptr[idx * column_count + i] -
                                    data_replace_ptr[j * column_count + i];
                        sum += val * val;
                    }
                    if (sum > epsilon) {
                        continue;
                    }
                    neighbours_ptr[idx] += use_weights ? weights_ptr[j] : Float(1);

                    if (neighbours_ptr[idx] >= min_observations && !use_weights) {
                        cores_ptr[idx] = Float(1);
                        break;
                    }
                }
                //It is necesasry to check in cases with weights, due to weights could be negative
                if (neighbours_ptr[idx] >= min_observations) {
                    cores_ptr[idx] = Float(1);
                }
            });
        });
        return event;
    }
};

///  A function that dispatches the local core points search function
///  based on the column count
///
/// @tparam Float  Floating-point type used to perform computations
///
/// @param[in]  queue             The SYCL queue
/// @param[in]  data              The input data of size `row_count` x `column_count`
/// @param[in]  weights           The input weights of size `row_count` x `1`
/// @param[in]  cores             The current cores of size `row_count` x `1`
/// @param[in]  neighbours        The current neighbours of size `row_count` x `1`
///                               it contains the counter of neighbours for each point
/// @param[in]  epsilon           The input parameter epsilon
/// @param[in]  min_observations  The input parameter min_observation
/// @param[in]  deps              Events indicating availability of the `data` for reading or writing
///
/// @return A SYCL event indicating the availability
/// of the updated arrays(cores and neighbours) for reading and writing
template <typename Float>
template <bool use_weights>
sycl::event kernels_fp<Float>::get_cores_impl(sycl::queue& queue,
                                              const pr::ndview<Float, 2>& data,
                                              const pr::ndview<Float, 2>& weights,
                                              pr::ndview<std::int32_t, 1>& cores,
                                              pr::ndview<Float, 1>& neighbours,
                                              Float epsilon,
                                              std::int64_t min_observations,
                                              const bk::event_vector& deps) {
    const std::int64_t column_count = data.get_dimension(1);
    if (column_count > get_recommended_min_width(queue)) {
        return get_core_wide_kernel<Float, use_weights>::run(queue,
                                                             data,
                                                             weights,
                                                             cores,
                                                             neighbours,
                                                             epsilon,
                                                             min_observations,
                                                             deps);
    }
    else {
        return get_core_narrow_kernel<Float, use_weights>::run(queue,
                                                               data,
                                                               weights,
                                                               cores,
                                                               neighbours,
                                                               epsilon,
                                                               min_observations,
                                                               deps);
    }
}

///  A function that dispatches the sendrecv_replaced core points search function
///  based on the column count
///
/// @tparam Float  Floating-point type used to perform computations
///
/// @param[in]  queue             The SYCL queue
/// @param[in]  data              The input data of size `row_count` x `column_count`
/// @param[in]  data_replace      The input data from another rank of size `row_count` x `column_count`
/// @param[in]  weights           The input weights of size `row_count` x `1`
/// @param[in]  cores             The current cores of size `row_count` x `1`
/// @param[in]  neighbours        The current neighbours of size `row_count` x `1`
///                               it contains the counter of neighbours for each point
/// @param[in]  epsilon           The input parameter epsilon
/// @param[in]  min_observations  The input parameter min_observation
/// @param[in]  deps              Events indicating availability of the `data` for reading or writing
///
/// @return A SYCL event indicating the availability
/// of the updated arrays(cores and neighbours) for reading and writing
template <typename Float>
template <bool use_weights>
sycl::event kernels_fp<Float>::get_cores_send_recv_replace_impl(
    sycl::queue& queue,
    const pr::ndview<Float, 2>& data,
    const pr::ndview<Float, 2>& data_replace,
    const pr::ndview<Float, 2>& weights,
    pr::ndview<std::int32_t, 1>& cores,
    pr::ndview<Float, 1>& neighbours,
    Float epsilon,
    std::int64_t min_observations,
    const bk::event_vector& deps) {
    const std::int64_t column_count = data.get_dimension(1);
    if (column_count > get_recommended_min_width(queue)) {
        return get_core_send_recv_replace_wide_kernel<Float, use_weights>::run(queue,
                                                                               data,
                                                                               data_replace,
                                                                               weights,
                                                                               cores,
                                                                               neighbours,
                                                                               epsilon,
                                                                               min_observations,
                                                                               deps);
    }
    else {
        return get_core_send_recv_replace_narrow_kernel<Float, use_weights>::run(queue,
                                                                                 data,
                                                                                 data_replace,
                                                                                 weights,
                                                                                 cores,
                                                                                 neighbours,
                                                                                 epsilon,
                                                                                 min_observations,
                                                                                 deps);
    }
}

///  A function that dispatches the local core points search function
///  based on the input wieghts
///
/// @tparam Float  Floating-point type used to perform computations
///
/// @param[in]  queue             The SYCL queue
/// @param[in]  data              The input data of size `row_count` x `column_count`
/// @param[in]  weights           The input weights of size `row_count` x `1`
/// @param[in]  cores             The current cores of size `row_count` x `1`
/// @param[in]  neighbours        The current neighbours of size `row_count` x `1`
///                               it contains the counter of neighbours for each point
/// @param[in]  epsilon           The input parameter epsilon
/// @param[in]  min_observations  The input parameter min_observation
/// @param[in]  deps              Events indicating availability of the `data` for reading or writing
///
/// @return A SYCL event indicating the availability
/// of the updated arrays(cores and neighbours) for reading and writing
template <typename Float>
sycl::event kernels_fp<Float>::get_cores(sycl::queue& queue,
                                         const pr::ndview<Float, 2>& data,
                                         const pr::ndview<Float, 2>& weights,
                                         pr::ndview<std::int32_t, 1>& cores,
                                         pr::ndview<Float, 1>& neighbours,
                                         Float epsilon,
                                         std::int64_t min_observations,
                                         const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(get_cores, queue);
    if (weights.get_dimension(0) == data.get_dimension(0)) {
        return get_cores_impl<true>(queue,
                                    data,
                                    weights,
                                    cores,
                                    neighbours,
                                    epsilon,
                                    min_observations,
                                    deps);
    }
    return get_cores_impl<false>(queue,
                                 data,
                                 weights,
                                 cores,
                                 neighbours,
                                 epsilon,
                                 min_observations,
                                 deps);
}

///  A function that dispatches the sendrecv_replaced core points search function
///  based on the input wieghts
///
/// @tparam Float  Floating-point type used to perform computations
///
/// @param[in]  queue             The SYCL queue
/// @param[in]  data              The input data of size `row_count` x `column_count`
/// @param[in]  data_replace      The input data from another rank of size `row_count` x `column_count`
/// @param[in]  weights           The input weights of size `row_count` x `1`
/// @param[in]  cores             The current cores of size `row_count` x `1`
/// @param[in]  neighbours        The current neighbours of size `row_count` x `1`
///                               it contains the counter of neighbours for each point
/// @param[in]  epsilon           The input parameter epsilon
/// @param[in]  min_observations  The input parameter min_observation
/// @param[in]  deps              Events indicating availability of the `data` for reading or writing
///
/// @return A SYCL event indicating the availability
/// of the updated arrays(cores and neighbours) for reading and writing
template <typename Float>
sycl::event kernels_fp<Float>::get_cores_send_recv_replace(sycl::queue& queue,
                                                           const pr::ndview<Float, 2>& data,
                                                           const pr::ndview<Float, 2>& data_replace,
                                                           const pr::ndview<Float, 2>& weights,
                                                           pr::ndview<std::int32_t, 1>& cores,
                                                           pr::ndview<Float, 1>& neighbours,
                                                           Float epsilon,
                                                           std::int64_t min_observations,
                                                           const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(get_cores_send_recv_replace, queue);
    if (weights.get_dimension(0) == data.get_dimension(0)) {
        return get_cores_send_recv_replace_impl<true>(queue,
                                                      data,
                                                      data_replace,
                                                      weights,
                                                      cores,
                                                      neighbours,
                                                      epsilon,
                                                      min_observations,
                                                      deps);
    }
    return get_cores_send_recv_replace_impl<false>(queue,
                                                   data,
                                                   data_replace,
                                                   weights,
                                                   cores,
                                                   neighbours,
                                                   epsilon,
                                                   min_observations,
                                                   deps);
}

///  A function that finds the init/next unassigned core to start clustering
///
/// @tparam Float  Floating-point type used to perform computations
///
/// @param[in]  queue      The SYCL queue
/// @param[in]  cores      The current cores of size `row_count` x `1`
/// @param[in]  responses  The current responses of size `row_count` x `1`
/// @param[in]  deps       Events indicating availability of the `data` for reading or writing
///
/// @return The index of the init/next unassigned core point
template <typename Float>
std::int32_t kernels_fp<Float>::start_next_cluster(sycl::queue& queue,
                                                   const pr::ndview<std::int32_t, 1>& cores,
                                                   pr::ndview<std::int32_t, 1>& responses,
                                                   const bk::event_vector& deps) {
    using oneapi::dal::backend::operator+;
    ONEDAL_PROFILER_TASK(start_next_cluster, queue);

    ONEDAL_ASSERT(cores.get_dimension(0) > 0);
    ONEDAL_ASSERT(cores.get_dimension(0) == responses.get_dimension(0));
    std::int64_t local_row_count = cores.get_dimension(0);
    ONEDAL_ASSERT(local_row_count > 0);

    auto [start_index, start_index_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, { 1 }, local_row_count, sycl::usm::alloc::device);
    auto start_index_ptr = start_index.get_mutable_data();

    const std::int32_t* cores_ptr = cores.get_data();
    std::int32_t* responses_ptr = responses.get_mutable_data();

    const std::int64_t wg_size = get_recommended_wg_size(queue);
    auto full_deps = deps + bk::event_vector{ start_index_event };
    auto index_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(full_deps);
        cgh.parallel_for(
            bk::make_multiple_nd_range_2d({ wg_size, 1 }, { wg_size, 1 }),
            [=](sycl::nd_item<2> item) {
                auto sg = item.get_sub_group();
                const std::uint32_t sg_id = sg.get_group_id()[0];
                if (sg_id > 0)
                    return;
                const std::int32_t local_id = sg.get_local_id();
                const std::int32_t local_size = sg.get_local_range()[0];
                std::int32_t adjusted_block_size =
                    local_size *
                    (local_row_count / local_size + bool(local_row_count % local_size));

                for (std::int32_t i = local_id; i < adjusted_block_size; i += local_size) {
                    const bool found =
                        i < local_row_count ? cores_ptr[i] == 1 && responses_ptr[i] < 0 : false;
                    const std::int32_t index =
                        sycl::reduce_over_group(sg,
                                                (std::int32_t)(found ? i : local_row_count),
                                                sycl::ext::oneapi::minimum<std::int32_t>());
                    if (index < local_row_count) {
                        if (local_id == 0) {
                            *start_index_ptr = index;
                        }
                        break;
                    }
                }
            });
    });
    return start_index.to_host(queue, { index_event }).at(0);
}

///  A function that sets the init point in ndview<bool, 1>& observation_indices
///
/// @param[in]  queue                The SYCL queue
/// @param[in]  observation_indices  The bool array of size `row_count` x `1`
///                                  that shows the points(current observations) which should
///                                  be copied and shared across all ranks
/// @param[in]  index                The index of the point
/// @param[in]  value                The value, can be true or false
/// @param[in]  deps                 Events indicating availability of the `data` for reading or writing
///
/// @return A SYCL event indicating the availability
/// of the updated array for reading and writing
sycl::event set_init_index(sycl::queue& queue,
                           pr::ndview<bool, 1>& observation_indices,
                           std::int32_t index,
                           bool value,
                           const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(set_init_index, queue);

    auto observation_indices_ptr = observation_indices.get_mutable_data();

    return queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(sycl::range<1>{ std::size_t(1) }, [=](sycl::id<1> idx) {
            observation_indices_ptr[index] = value;
        });
    });
}

///  A function that sets the queue sizes in ndview<std::int32_t, 1>& queue_size
///
/// @param[in]  queue              The SYCL queue
/// @param[in]  points_queue_size  The int32_t array of size `1`
///                                that helps to get queue sizes and storage it on GPU
/// @param[in]  index              The index of the point
/// @param[in]  value              The current queue size/value
/// @param[in]  deps               Events indicating availability of the `data` for reading or writing
///
/// @return A SYCL event indicating the availability
/// of the updated array for reading and writing
sycl::event set_arr_value(sycl::queue& queue,
                          pr::ndview<std::int32_t, 1>& points_queue_size,
                          std::int32_t index,
                          std::int32_t value,
                          const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(set_arr_value, queue);

    auto points_queue_size_ptr = points_queue_size.get_mutable_data();

    return queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(sycl::range<1>{ std::size_t(1) }, [=](sycl::id<1> idx) {
            points_queue_size_ptr[index] = value;
        });
    });
}

///  A function that fills the queue with current observations points
///
/// @tparam Float  Floating-point type used to perform computations
///
/// @param[in]  queue                        The SYCL queue
/// @param[in]  data                         The input data of size `row_count` x `column_count`
/// @param[in]  indices                      The indicies of the points which should be copied
///                                          `row_count` x `1`
/// @param[in]  current_points_queue         The array where points should be copied
/// @param[in]  local_points_queue_size_arr  The count of points which should be copied
/// @param[in]  block_start                  The offset for distributed usage
///
/// @return A SYCL event indicating the availability
/// of the updated arrays for reading and writing
template <typename Float>
sycl::event kernels_fp<Float>::fill_current_points_queue(
    sycl::queue& queue,
    const pr::ndview<Float, 2>& data,
    const pr::ndview<bool, 1>& indices,
    pr::ndview<Float, 2>& current_points_queue,
    pr::ndview<std::int32_t, 1>& local_points_queue_size_arr,
    std::int64_t block_start,
    const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(fill_current_points_queue, queue);

    const std::int64_t local_row_count = data.get_dimension(0);
    ONEDAL_ASSERT(local_row_count > 0);
    const std::int64_t column_count = data.get_dimension(1);
    ONEDAL_ASSERT(column_count > 0);

    bool* indices_host_ptr = indices.get_mutable_data();
    const Float* data_host_ptr = data.get_data();
    std::int32_t* queue_size_arr_ptr = local_points_queue_size_arr.get_mutable_data();
    Float* current_queue_ptr = current_points_queue.get_mutable_data();

    const sycl::nd_range<1> nd_range = bk::make_multiple_nd_range_1d(local_row_count, 1);
    return queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> idx) {
            auto row_id = idx.get_global_id(0);
            if (indices_host_ptr[row_id]) {
                sycl::atomic_ref<int,
                                 sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    counter_atomic(queue_size_arr_ptr[0]);
                auto cur_idx = counter_atomic.fetch_add(1);
                for (std::int32_t col_idx = 0; col_idx < column_count; col_idx += 1) {
                    current_queue_ptr[block_start * column_count + cur_idx * column_count +
                                      col_idx] = data_host_ptr[row_id * column_count + col_idx];
                }
                indices_host_ptr[row_id] = false;
            }
        });
    });
}

///  A struct that calculate distances between current observations and unassigned points.
///  Also this function updates the responses and indicies of points which should be copied on the next step
///
/// @tparam Float  Floating-point type used to perform computations
///
/// @param[in]  queue                  The SYCL queue
/// @param[in]  data                   The input data of size `row_count` x `column_count`
/// @param[in]  cores                  The current cores of size `row_count` x `1`
/// @param[in]  current_points_queue   The current observations. It is an array that contains
///                                    only core points in the epsilon area of the init/next point
/// @param[in]  responses              The current responbses of size `row_count` x `1`
/// @param[in]  queue_size_arr         The array(1x1) that contains total number of new observations
/// @param[in]  indices_cores          The indicies of the points which should be copied `row_count` x `1`
/// @param[in]  epsilon                The input parameter epsilon
/// @param[in]  cluster_id             The current cluster id
/// @param[in]  deps                   Events indicating availability of the `data` for reading or writing
///
/// @return A SYCL event indicating the availability
/// of the updated arrays for reading and writing
template <typename Float>
sycl::event kernels_fp<Float>::update_points_queue(sycl::queue& queue,
                                                   const pr::ndview<Float, 2>& data,
                                                   const pr::ndview<std::int32_t, 1>& cores,
                                                   pr::ndview<Float, 2>& current_points_queue,
                                                   pr::ndview<std::int32_t, 1>& responses,
                                                   pr::ndview<std::int32_t, 1>& queue_size_arr,
                                                   pr::ndview<bool, 1>& indices_cores,
                                                   Float epsilon,
                                                   std::int32_t cluster_id,
                                                   const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(update_points_queue, queue);

    const auto local_row_count = data.get_dimension(0);
    ONEDAL_ASSERT(local_row_count > 0);
    const std::int64_t column_count = data.get_dimension(1);
    ONEDAL_ASSERT(column_count > 0);
    const std::int32_t queue_size = current_points_queue.get_dimension(0);
    ONEDAL_ASSERT(queue_size >= 0);

    const Float* data_ptr = data.get_data();
    std::int32_t* cores_ptr = cores.get_mutable_data();
    std::int32_t* queue_size_arr_ptr = queue_size_arr.get_mutable_data();
    const Float* current_queue_ptr = current_points_queue.get_data();
    bool* indices_cores_ptr = indices_cores.get_mutable_data();
    std::int32_t* responses_ptr = responses.get_mutable_data();

    auto fill_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(sycl::range<1>{ std::size_t(1) }, [=](sycl::id<1> idx) {
            queue_size_arr_ptr[0] = 0;
        });
    });

    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(fill_event);
        const std::int64_t wg_size = get_recommended_wg_size(queue, column_count);
        cgh.parallel_for(
            bk::make_multiple_nd_range_2d({ wg_size, local_row_count }, { wg_size, 1 }),
            [=](sycl::nd_item<2> item) {
                auto sg = item.get_sub_group();
                const std::uint32_t sg_id = sg.get_group_id()[0];
                if (sg_id > 0)
                    return;
                const std::uint32_t wg_id = item.get_global_id(1);
                if (wg_id >= local_row_count)
                    return;
                const std::uint32_t local_id = sg.get_local_id();
                const std::uint32_t local_size = sg.get_local_range()[0];

                if (responses_ptr[wg_id] >= 0)
                    return;

                for (std::int64_t j = 0; j < queue_size; j++) {
                    Float sum = 0.0;
                    for (std::int64_t i = local_id; i < column_count; i += local_size) {
                        Float val = data_ptr[wg_id * column_count + i] -
                                    current_queue_ptr[j * column_count + i];
                        sum += val * val;
                    }
                    Float distance =
                        sycl::reduce_over_group(sg, sum, sycl::ext::oneapi::plus<Float>());
                    if (distance > epsilon)
                        continue;
                    if (local_id == 0) {
                        responses_ptr[wg_id] = cluster_id;
                    }

                    if (cores_ptr[wg_id] == 0)
                        continue;
                    if (local_id == 0) {
                        sycl::atomic_ref<std::int32_t,
                                         sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            counter_atomic(queue_size_arr_ptr[0]);
                        counter_atomic.fetch_add(1);
                        indices_cores_ptr[wg_id] = true;
                    }
                    break;
                }
            });
    });
    return event;
}

///  A function that gets the queue sizes in ndview<std::int32_t, 1>& queue_size
///
/// @param[in]  queue              The SYCL queue
/// @param[in]  points_queue_size  The int32_t array of size `1`
///                                that helps to get queue sizes and storage it on GPU
/// @param[in]  deps               Events indicating availability of the `data` for reading or writing
///
/// @return The queue size
template <typename Float>
std::int32_t kernels_fp<Float>::get_points_queue_size(
    sycl::queue& queue,
    const pr::ndarray<std::int32_t, 1>& points_queue_size,
    const bk::event_vector& deps) {
    ONEDAL_ASSERT(points_queue_size.get_dimension(0) == 1);
    return points_queue_size.to_host(queue, deps).get_data()[0];
}

///  A function that counts the number of cores
///
/// @param[in]  queue  The SYCL queue
/// @param[in]  cores  The current cores of size `row_count` x `1`
///                    that helps to get queue sizes and storage it on GPU
///
/// @return The number of cores
std::int64_t count_cores(sycl::queue& queue, const pr::ndview<std::int32_t, 1>& cores) {
    const std::uint64_t row_count = cores.get_dimension(0);
    ONEDAL_ASSERT(row_count > 0);
    std::int64_t sum_result = 0;
    auto cores_ptr = cores.get_data();
    sycl::buffer<std::int64_t> sum_buf{ &sum_result, 1 };
    queue
        .submit([&](sycl::handler& cgh) {
            auto sum_reduction = reduction(sum_buf, cgh, sycl::ext::oneapi::plus<std::int64_t>());
            cgh.parallel_for(sycl::range<1>{ row_count },
                             sum_reduction,
                             [=](sycl::id<1> idx, auto& sum) {
                                 sum.combine(cores_ptr[idx]);
                             });
        })
        .wait_and_throw();
    return sum_buf.get_host_access()[0];
}

#endif

} // namespace oneapi::dal::dbscan::backend
