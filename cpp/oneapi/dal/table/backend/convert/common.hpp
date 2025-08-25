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

#pragma once

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/common.hpp"

#include "oneapi/dal/backend/common.hpp"

namespace oneapi::dal::backend {

using shape_t = std::pair<std::int64_t, std::int64_t>;

template <typename T1, typename T2>
inline std::pair<T2, T1> transpose(const std::pair<T1, T2>& p) {
    return std::pair<T2, T1>{ p.second, p.first };
}

bool is_known_data_type(data_type dtype) noexcept;

template <typename Index, typename Type>
inline dal::array<Type> extract_by_indices(const Index* indices,
                                           const Type* values,
                                           std::int64_t count) {
    auto result = dal::array<Type>::empty(count);
    auto* const output = result.get_mutable_data();

    PRAGMA_FORCE_SIMD
    for (std::int64_t i = 0l; i < count; ++i) {
        const Index idx = indices[i];
        output[i] = values[idx];
    }

    return result;
}

template <typename Index, typename Type>
inline dal::array<Type> extract_by_indices(const dal::array<Index>& indices,
                                           const dal::array<Type>& values) {
    const std::int64_t count = indices.get_count();
    return extract_by_indices(indices.get_data(), values.get_data(), count);
}

dal::array<std::int64_t> compute_lower_bounds(const shape_t& input_shape,
                                              const dal::array<data_type>& input_types);

dal::array<std::int64_t> compute_lower_bounds(const shape_t& input_shape,
                                              const data_type* input_types);

inline dal::array<std::int64_t> compute_offsets(const shape_t& input_shape,
                                                const dal::array<data_type>& input_types) {
    return compute_lower_bounds(input_shape, input_types);
}

inline dal::array<std::int64_t> compute_offsets(const shape_t& input_shape,
                                                const data_type* input_types) {
    return compute_lower_bounds(input_shape, input_types);
}

std::int64_t count_unique_chunk_offsets(const dal::array<std::int64_t>& indices, //
                                        const data_type* inp,
                                        const data_type* out);

dal::array<std::int64_t> find_unique_chunk_offsets(const dal::array<std::int64_t>& indices, //
                                                   const data_type* inp,
                                                   const data_type* out);

dal::array<std::int64_t> find_sets_of_unique_pairs(const data_type* inp,
                                                   const data_type* out,
                                                   std::int64_t count);

dal::array<std::int64_t> find_unique_chunk_offsets(const dal::array<std::int64_t>& indices, //
                                                   const dal::array<data_type>& inp,
                                                   const dal::array<data_type>& out);

dal::array<std::int64_t> find_sets_of_unique_pairs(const dal::array<data_type>& inp,
                                                   const dal::array<data_type>& out);

} // namespace oneapi::dal::backend
