/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/backend/atomic.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::decision_forest::backend {

using sycl::ext::oneapi::plus;
using sycl::ext::oneapi::minimum;
using sycl::ext::oneapi::maximum;

template <typename Data>
using local_accessor_rw_t = sycl::local_accessor<Data, 1>;

template <typename Float>
inline void merge_stat(Float& dst_count,
                       Float& dst_mean,
                       Float& dst_s2c,
                       Float src_count,
                       Float src_mean,
                       Float src_s2c) {
    if (Float(0) == src_count)
        return;
    if (!dst_count) {
        dst_count = src_count;
        dst_mean = src_mean;
        dst_s2c = src_s2c;
        return;
    }

    Float sum_n1n2 = dst_count + src_count;
    Float mul_n1n2 = dst_count * src_count;
    Float delta_scl = mul_n1n2 / sum_n1n2;
    Float mean_scl = Float(1) / sum_n1n2;
    Float delta = src_mean - dst_mean;

    dst_s2c = dst_s2c + src_s2c + delta * delta * delta_scl;
    dst_mean = (dst_mean * dst_count + src_mean * src_count) * mean_scl;
    dst_count = sum_n1n2;
}

template <typename T, typename Index>
T* get_buf_ptr(byte_t** buf_ptr, Index elem_count) {
    T* res_ptr = reinterpret_cast<T*>(*buf_ptr);
    (*buf_ptr) += elem_count * sizeof(T);
    return res_ptr;
}

template <typename Float, typename Index = std::int32_t, typename ItemT>
inline void reduce_hist_over_group(ItemT& item,
                                   Float* slm_buf_ptr,
                                   Float& count,
                                   Float& mean,
                                   Float& sum2cent) {
    auto sbg = item.get_sub_group();
    Index sub_group_id = sbg.get_group_id();
    Index sub_group_count = item.get_local_range()[0] / sbg.get_local_range()[0];
    Index local_id = item.get_local_id(0);

    Float* count_buf_ptr = slm_buf_ptr + sub_group_count * 0;
    Float* mean_buf_ptr = slm_buf_ptr + sub_group_count * 1;
    Float* sum2cent_buf_ptr = slm_buf_ptr + sub_group_count * 2;

    if (sbg.get_local_id() == 0) {
        count_buf_ptr[sub_group_id] = count;
        mean_buf_ptr[sub_group_id] = mean;
        sum2cent_buf_ptr[sub_group_id] = sum2cent;
    }

    for (Index stride = sub_group_count / 2; stride > 0; stride /= 2) {
        item.barrier(sycl::access::fence_space::local_space);
        if (local_id < stride) {
            merge_stat(count_buf_ptr[local_id],
                       mean_buf_ptr[local_id],
                       sum2cent_buf_ptr[local_id],
                       count_buf_ptr[local_id + stride],
                       mean_buf_ptr[local_id + stride],
                       sum2cent_buf_ptr[local_id + stride]);
        }
    }
    item.barrier(sycl::access::fence_space::local_space);

    count = count_buf_ptr[0];
    mean = mean_buf_ptr[0];
    sum2cent = sum2cent_buf_ptr[0];
}

} // namespace oneapi::dal::decision_forest::backend

#endif
