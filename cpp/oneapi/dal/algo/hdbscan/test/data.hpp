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

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"

namespace oneapi::dal::hdbscan::test {

namespace te = dal::test::engine;

class gold_dataset {
public:
    gold_dataset() = delete;

    static std::int64_t get_row_count() {
        return row_count;
    }

    static std::int64_t get_column_count() {
        return column_count;
    }

    static std::int64_t get_min_cluster_size() {
        return min_cluster_size;
    }

    static std::int64_t get_min_samples() {
        return min_samples;
    }

    // 4 well-separated clusters of 5 points each, plus 1 noise point
    // Cluster 0: around (0, 0)
    // Cluster 1: around (10, 10)
    // Cluster 2: around (0, 10)
    // Cluster 3: around (10, 0)
    // Point 20: noise at (5, 5)
    static te::dataframe get_data() {
        static std::array<float, row_count* column_count> data = {
            // Cluster 0: around (0, 0)
            0.1f,
            0.2f, //
            0.15f,
            0.22f, //
            0.12f,
            0.18f, //
            0.11f,
            0.25f, //
            0.13f,
            0.21f, //
            // Cluster 1: around (10, 10)
            10.0f,
            10.0f, //
            10.1f,
            10.05f, //
            9.95f,
            9.98f, //
            10.05f,
            10.1f, //
            9.98f,
            9.95f, //
            // Cluster 2: around (0, 10)
            0.05f,
            10.0f, //
            0.1f,
            10.1f, //
            0.08f,
            9.95f, //
            0.12f,
            10.05f, //
            0.07f,
            9.98f, //
            // Cluster 3: around (10, 0)
            10.0f,
            0.1f, //
            10.1f,
            0.05f, //
            9.95f,
            0.08f, //
            10.05f,
            0.12f, //
            9.98f,
            0.07f, //
            // Noise point
            5.0f,
            5.0f, //
        };
        return te::dataframe{ data.data(), row_count, column_count };
    }

private:
    static constexpr std::int64_t row_count = 21;
    static constexpr std::int64_t column_count = 2;
    static constexpr std::int64_t min_cluster_size = 5;
    static constexpr std::int64_t min_samples = 5;
};

} // namespace oneapi::dal::hdbscan::test
