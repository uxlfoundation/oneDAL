/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <vector>
#include <string>
#include <fstream>

#include "oneapi/dal/table/row_accessor.hpp"

namespace dal = oneapi::dal;

inline bool check_file(const std::string& name) {
    return std::ifstream{ name }.good();
}

inline std::string get_data_path(const std::string& name) {
    const std::vector<std::string> paths = { "../data", "examples/oneapi/data" };

    for (const auto& path : paths) {
        const std::string try_path = path + "/" + name;
        if (check_file(try_path)) {
            return try_path;
        }
    }

    return name;
}

template <typename Float>
std::vector<dal::table> split_table_by_rows(const dal::table& t, std::int64_t split_count) {
    ONEDAL_ASSERT(split_count > 0);
    ONEDAL_ASSERT(split_count <= t.get_row_count());

    const std::int64_t row_count = t.get_row_count();
    const std::int64_t column_count = t.get_column_count();
    const std::int64_t block_size_regular = row_count / split_count;
    const std::int64_t block_size_tail = row_count % split_count;

    std::vector<dal::table> result(split_count);

    std::int64_t row_offset = 0;
    for (std::int64_t i = 0; i < split_count; i++) {
        const std::int64_t tail = std::int64_t(i + 1 == split_count) * block_size_tail;
        const std::int64_t block_size = block_size_regular + tail;

        const auto row_range = dal::range{ row_offset, row_offset + block_size };
        const auto block = dal::row_accessor<const Float>{ t }.pull(row_range);
        result[i] = dal::homogen_table::wrap(block, block_size, column_count);
        row_offset += block_size;
    }

    return result;
}
