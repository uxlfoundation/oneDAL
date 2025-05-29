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
#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/array.hpp"
#include <cmath>

namespace dal = oneapi::dal;

/// Converts a homogen table to one-based CSR table.
///
/// @param[in] data     an input homogen table
template <typename Float = float, typename Index = std::int64_t>
inline const dal::csr_table convert_to_csr(const dal::table& data) {
    Index non_zero_count = 0;
    dal::row_accessor<const Float> accessor{ data };
    const auto data_ptr = accessor.pull({ 0, -1 });
    const Index column_count = data.get_column_count();
    const Index row_count = data.get_row_count();

    for (Index i = 0; i < data_ptr.get_count(); ++i) {
        non_zero_count += (std::fabs(data_ptr[i]) > std::numeric_limits<Float>::epsilon());
    }

    const auto compressed_data = dal::array<Float>::empty(non_zero_count);
    const auto row_offsets = dal::array<Index>::empty(row_count + 1);
    const auto col_indices = dal::array<Index>::empty(non_zero_count);

    auto comp_data_ptr = compressed_data.get_mutable_data();
    auto row_offsets_ptr = row_offsets.get_mutable_data();
    auto col_indices_ptr = col_indices.get_mutable_data();

    Index compressed_idx = 1;
    for (Index i = 0; i <= row_count; ++i) {
        row_offsets_ptr[i] = 1;
    }

    for (Index i = 0; i < data_ptr.get_count(); ++i) {
        if (std::fabs(data_ptr[i]) > std::numeric_limits<Float>::epsilon()) {
            comp_data_ptr[compressed_idx - 1] = data_ptr[i];
            Index row_idx = i / column_count + 1;
            Index col_idx = i % column_count + 1;
            row_offsets_ptr[row_idx] = compressed_idx;
            col_indices_ptr[compressed_idx - 1] = col_idx;
            ++compressed_idx;
        }
    }
    // Fill empty rows with previous value
    for (Index i = 1; i <= row_count; ++i) {
        row_offsets_ptr[i] = std::max(row_offsets_ptr[i - 1], row_offsets_ptr[i]);
    }
    row_offsets_ptr[row_count] = non_zero_count + 1;

    auto t = dal::csr_table::wrap(compressed_data,
                                  col_indices,
                                  row_offsets,
                                  column_count,
                                  oneapi::dal::sparse_indexing::one_based);
    return t;
}
