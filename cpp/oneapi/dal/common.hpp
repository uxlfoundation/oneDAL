/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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

#include <cstdint>

namespace dal {

using byte_t = std::uint8_t;

class base {
public:
    virtual ~base() = default;
};

enum class data_type {
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float32,
    float64,
    bfloat16
};

struct range {
public:
    range(std::int64_t start, std::int64_t end)
        : start_idx(start), end_idx(end) {}

    std::int64_t get_element_count(std::int64_t max_end_index) const noexcept {
        // TODO: handle error if (max_end_index + end_idx) < 0
        std::int64_t final_row = (end_idx < 0) ? max_end_index + end_idx + 1 : end_idx;
        return (final_row - start_idx - 1) + 1;
    }

    std::int64_t start_idx;
    std::int64_t end_idx;
};

} // namespace dal
