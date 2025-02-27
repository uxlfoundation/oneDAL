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

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <typename Float>
struct float2uint_map;

template <>
struct float2uint_map<float> {
    using integer_t = std::uint32_t;
};

template <>
struct float2uint_map<double> {
    using integer_t = std::uint64_t;
};

template <typename Float, typename Index = std::uint32_t>
sycl::event radix_sort_indices_inplace(sycl::queue& queue,
                                       ndview<Float, 1>& val,
                                       ndview<Index, 1>& ind,
                                       const event_vector& deps = {});

template <typename Integer>
sycl::event radix_sort(sycl::queue& queue,
                       ndview<Integer, 2>& val_in,
                       ndview<Integer, 2>& val_out,
                       std::int64_t sorted_elem_count,
                       const event_vector& deps = {});

template <typename Integer>
sycl::event radix_sort(sycl::queue& queue,
                       ndview<Integer, 2>& val_in,
                       ndview<Integer, 2>& val_out,
                       const event_vector& deps = {});

#endif

} // namespace oneapi::dal::backend::primitives
