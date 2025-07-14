/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#include <fstream>
#include <sstream>
#include <vector>

#include "oneapi/dal/io/csv/common.hpp"

namespace oneapi::dal::csv::backend {

template <typename Float>
inline std::tuple<std::vector<std::int64_t>, std::vector<std::int64_t>, std::vector<Float>>
read_csr_data(const detail::data_source_base& ds, const read_args<csr_table>& args) {
    std::ifstream file(ds.get_file_name());
    if (!file.is_open()) {
        throw dal::range_error(dal::detail::error_messages::file_not_found());
    }

    // csr table format:
    // first line: row offsets, i.e., the indices (within non-zero elems array) of the first non-zero element in each row
    // second line: column indices of the non-zero elements
    // third line: non-zero elements of the original matrix

    std::string line;
    std::vector<std::int64_t> row_offsets;
    std::vector<std::int64_t> col_indices;
    std::vector<Float> values;

    char delimiter = ds.get_delimiter();
    std::int64_t feature_count = args.get_feature_count();
    sparse_indexing indexing = args.get_sparse_indexing();

    // Read the first line (row offsets)
    if (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string value;
        while (std::getline(ss, value, delimiter)) {
            row_offsets.push_back(static_cast<std::int64_t>(std::stoi(value)));
        }
    }
    else {
        throw dal::invalid_argument(dal::detail::error_messages::invalid_csr_format());
    }

    // Read the second line (column indices)
    if (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string value;
        while (std::getline(ss, value, delimiter)) {
            col_indices.push_back(static_cast<std::int64_t>(std::stoi(value)));
        }
    }
    else {
        throw dal::invalid_argument(dal::detail::error_messages::invalid_csr_format());
    }

    // Read the third line (non-zero elements)
    if (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string value;
        while (std::getline(ss, value, delimiter)) {
            values.push_back(static_cast<Float>(std::stod(value)));
        }
    }
    else {
        throw dal::invalid_argument(dal::detail::error_messages::invalid_csr_format());
    }

    std::int64_t row_offsets_checker =
        static_cast<std::int64_t>(values.size()) + (indexing == sparse_indexing::one_based ? 1 : 0);
    if (values.size() != col_indices.size() || row_offsets.size() == 0 ||
        row_offsets[row_offsets.size() - 1] != row_offsets_checker) {
        throw dal::invalid_argument(dal::detail::error_messages::invalid_csr_format());
    }

    if (!col_indices.empty() && feature_count == 0) {
        // If feature count is not specified, we estimate it from the column indices
        // This is a best-effort estimation
        feature_count = *std::max_element(col_indices.begin(), col_indices.end());
        if (indexing == sparse_indexing::zero_based) {
            feature_count += 1;
        }
    }

    return std::make_tuple(std::move(row_offsets), std::move(col_indices), std::move(values));
}

} // namespace oneapi::dal::csv::backend
