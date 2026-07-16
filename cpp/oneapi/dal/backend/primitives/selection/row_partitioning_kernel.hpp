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

#include "oneapi/dal/backend/common.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

constexpr std::uint32_t partitioning_preffered_sg_size = 16;

inline sycl::nd_range<2> get_row_partitioning_range(std::int64_t row_count,
                                                    std::int64_t col_count) {
    return make_multiple_nd_range_2d({ partitioning_preffered_sg_size, row_count },
                                     { partitioning_preffered_sg_size, 1 });
}

template <typename Float>
inline std::int32_t row_partitioning_kernel(sycl::nd_item<2> item,
                                            Float* values,
                                            std::int32_t* indices,
                                            std::int32_t partition_start,
                                            std::int32_t partition_end,
                                            Float pivot) {
    auto sg = item.get_sub_group();
    const std::int32_t local_id = sg.get_local_id()[0];
    const std::int32_t local_size = sg.get_local_range()[0];

    const std::int32_t seq_size = partition_end - partition_start;
    const std::int32_t last_group_size = seq_size % local_size;
    std::int32_t full_group_size = (seq_size / local_size) * local_size;
    full_group_size += last_group_size > 0 ? local_size : 0;
    std::int32_t split_index = 0;

    for (std::int32_t i = partition_start + local_id; i < partition_start + full_group_size;
         i += local_size) {
        sycl::group_barrier(sg);
        bool inside = i < partition_end;
        Float cur_value = inside ? values[i] : 0.0;
        std::int32_t cur_index = i < partition_end ? indices[i] : -1;
        std::int32_t is_small = cur_value < pivot ? 1 : 0;
        const std::int32_t num_of_small =
            sycl::reduce_over_group(sg, is_small && inside ? 1 : 0, sycl::plus<int>());
        std::int32_t min_ind = sycl::reduce_over_group(sg, i, sycl::minimum<int>());
        if (num_of_small > 0) {
            const std::int32_t pos_in_group_small =
                sycl::exclusive_scan_over_group(sg, is_small && inside ? 1 : 0, sycl::plus<int>());
            const std::int32_t pos_in_group_great =
                num_of_small +
                sycl::exclusive_scan_over_group(sg, is_small || !inside ? 0 : 1, sycl::plus<int>());
            const std::int32_t pos_small = partition_start + split_index + pos_in_group_small;
            const std::int32_t pos_great = min_ind + pos_in_group_great;
            const std::int32_t pos = is_small ? pos_small : pos_great;
            const std::int32_t pos_to_move =
                pos < min_ind ? min_ind + num_of_small - 1 - pos_in_group_small : -1;
            if (inside) {
                if (is_small) {
                    Float value_to_move = values[pos];
                    std::int32_t index_to_move = indices[pos];
                    if (pos_to_move > -1) {
                        values[pos_to_move] = value_to_move;
                        indices[pos_to_move] = index_to_move;
                    }
                }
                values[pos] = cur_value;
                indices[pos] = cur_index;
            }
        }
        split_index += num_of_small;
    }
    split_index = -sycl::reduce_over_group(sg, -split_index, sycl::minimum<std::int32_t>());
    return split_index + partition_start;
}

#endif // ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::backend::primitives
