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

#include <algorithm>

#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

#include "oneapi/dal/backend/primitives/placement/cumulative_sum.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Type>
sycl::event up_sweep(sycl::queue& queue,
                     ndview<Type, 1>& data,
                     std::int64_t base_stride,
                     const event_vector& deps) {
    const auto count = data.get_count();
    const auto iter_count = up_log(count, base_stride) + 1;

    sycl::event event{};
    std::int64_t curr_stride = 1;

    for (std::int64_t i = 0; i < iter_count; ++i) {
        event_vector new_deps(deps);
        new_deps.push_back(event);

        event = detail::block_cumsum(queue, data, base_stride, curr_stride, new_deps);

        curr_stride *= base_stride;
    }

    return event;
}

template <typename Type>
sycl::event down_sweep(sycl::queue& queue,
                       ndview<Type, 1>& data,
                       std::int64_t base_stride,
                       const event_vector& deps) {
    const auto count = data.get_count();
    const auto iter_count = up_log(count, base_stride) + 1;

    sycl::event event{};
    std::int64_t curr_stride = 1;

    for (std::int64_t i = 0; i < iter_count; ++i) {
        curr_stride *= base_stride;

        event_vector new_deps(deps);
        new_deps.push_back(event);

        event = detail::distribute_sum(queue, data, base_stride, curr_stride, new_deps);
    }

    return event;
}

template <typename Type>
sycl::event cumulative_sum_1d(sycl::queue& queue,
                              ndview<Type, 1>& data,
                              std::int64_t base_stride,
                              const event_vector& deps) {
    ONEDAL_ASSERT(base_stride > 0);
    ONEDAL_ASSERT(data.has_mutable_data());
    ONEDAL_ASSERT(base_stride <= device_max_wg_size(queue));

    auto up_event = up_sweep(queue, data, base_stride, deps);
    return down_sweep(queue, data, base_stride, { up_event });
}

template <typename Type>
std::int64_t propose_cumsum_block(const sycl::queue& queue, const ndview<Type, 1>& data) {
    return propose_wg_size(queue);
}

template <typename Type>
sycl::event cumulative_sum_1d(sycl::queue& queue, ndview<Type, 1>& data, const event_vector& deps) {
    const auto stride = propose_cumsum_block(queue, data);
    return cumulative_sum_1d(queue, data, stride, deps);
}

#define INSTANTIATE(T)                                                                        \
    template sycl::event cumulative_sum_1d(sycl::queue&, ndview<T, 1>&, const event_vector&); \
    template sycl::event cumulative_sum_1d(sycl::queue&,                                      \
                                           ndview<T, 1>&,                                     \
                                           std::int64_t,                                      \
                                           const event_vector&);

INSTANTIATE(float)
INSTANTIATE(double)

#undef INSTANTIATE

namespace detail {

template <typename Type>
sycl::event block_cumsum(sycl::queue& queue,
                         ndview<Type, 1>& data,
                         std::int64_t base_stride,
                         std::int64_t curr_stride,
                         const event_vector& deps) {
    ONEDAL_ASSERT(base_stride > 0);
    ONEDAL_ASSERT(curr_stride > 0);
    ONEDAL_ASSERT(data.has_mutable_data());
    ONEDAL_ASSERT(base_stride <= device_max_wg_size(queue));

    const auto count = data.get_count();
    auto* const ptr = data.get_mutable_data();

    const auto elem_handle = (count - curr_stride + 1) / curr_stride;

    if (elem_handle < 1)
        return wait_or_pass(deps);

    const auto range = make_multiple_nd_range_1d(elem_handle, base_stride);

    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);

        h.parallel_for(range, [=](sycl::nd_item<1> item) {
            constexpr Type zero = 0;
            constexpr sycl::plus<Type> plus{};

            auto group = item.get_group();
            const std::int64_t lid = item.get_local_linear_id();
            const std::int64_t gid = item.get_global_linear_id();

            //const std::int64_t idx = curr_stride - 1 + gid * curr_stride;
            const std::int64_t idx = curr_stride - 1 + gid * curr_stride;
            const Type val = (idx < count) ? ptr[idx] : zero;
            auto res = sycl::inclusive_scan_over_group(group, val, plus);
            if (idx < count && lid < base_stride)
                ptr[idx] = res;
        });
    });
}

template <typename Type>
sycl::event distribute_sum(sycl::queue& queue,
                           ndview<Type, 1>& data,
                           std::int64_t base_stride,
                           std::int64_t curr_stride,
                           const event_vector& deps) {
    ONEDAL_ASSERT(base_stride > 0);
    ONEDAL_ASSERT(curr_stride > 0);
    ONEDAL_ASSERT(data.has_mutable_data());

    const auto count = data.get_count();
    auto* const ptr = data.get_mutable_data();

    const auto range = make_range_1d(count);

    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);

        h.parallel_for(range, [=](sycl::id<1> gid) {
            const std::int64_t idx = gid;
            const auto block_id = idx / curr_stride;
            const auto prev = block_id * curr_stride - 1;
            const bool handle = (block_id % base_stride != 0) && (idx != prev + curr_stride);
            if (idx < count && handle)
                ptr[idx] += ptr[prev];
        });
    });
}

#define INSTANTIATE(T)                                      \
    template sycl::event block_cumsum(sycl::queue&,         \
                                      ndview<T, 1>&,        \
                                      std::int64_t,         \
                                      std::int64_t,         \
                                      const event_vector&); \
    template sycl::event distribute_sum(sycl::queue&,       \
                                        ndview<T, 1>&,      \
                                        std::int64_t,       \
                                        std::int64_t,       \
                                        const event_vector&);

INSTANTIATE(float)
INSTANTIATE(double)

#undef INSTANTIATE

} // namespace detail

} // namespace oneapi::dal::backend::primitives
