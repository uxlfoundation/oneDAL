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

#include "oneapi/dal/table/homogen.hpp"
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
        static std::array<float, row_count * column_count> data = {
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

/// Wrap a row-major literal as a table of the floating-point type under test.
///
/// @tparam Float Floating-point type of the table
///
/// @param[in] src       Source values, `row_count * col_count` of them
/// @param[in] row_count Number of rows
/// @param[in] col_count Number of columns
template <typename Float>
inline table make_table(const double* src, std::int64_t row_count, std::int64_t col_count) {
    auto arr = dal::array<Float>::empty(row_count * col_count);
    Float* const dst = arr.get_mutable_data();
    for (std::int64_t i = 0; i < row_count * col_count; i++) {
        dst[i] = static_cast<Float>(src[i]);
    }
    return homogen_table::wrap(arr, row_count, col_count);
}

/// Two tight 5-point blobs ten units apart.
constexpr inline std::int64_t two_blob_row_count = 10;
constexpr inline double two_blob_data[] = {
    0.0,  0.0,  0.1,  0.1,  0.2,  0.0,  0.0,  0.2,  0.15,  0.15, //
    10.0, 10.0, 10.1, 10.1, 10.2, 10.0, 10.0, 10.2, 10.15, 10.15,
};

/// The two blobs above plus one point halfway between them.
constexpr inline std::int64_t two_blob_noise_row_count = 11;
constexpr inline double two_blob_noise_data[] = {
    0.0,  0.0,  0.1,  0.1,  0.2,  0.0,  0.0,  0.2,  0.15,  0.15, //
    10.0, 10.0, 10.1, 10.1, 10.2, 10.0, 10.0, 10.2, 10.15, 10.15, //
    5.0,  5.0,
};

/// Two 3-point groups five units apart.
constexpr inline std::int64_t two_small_groups_row_count = 6;
constexpr inline double two_small_groups_data[] = {
    0.0, 0.0, 0.1, 0.1, 0.2, 0.0, //
    5.0, 5.0, 5.1, 5.1, 5.2, 5.0,
};

/// Three 5-point clusters on a line, one column.
constexpr inline std::int64_t three_blob_1d_row_count = 15;
constexpr inline double three_blob_1d_data[] = {
    0.0,  0.1,  0.2,  0.15,  0.05, //
    5.0,  5.1,  5.2,  5.15,  5.05, //
    10.0, 10.1, 10.2, 10.15, 10.05,
};

/// Two blobs of different density, a point that barely joins the dense one and a noise point.
/// The references of the tests that use it come from
/// `sklearn.cluster.HDBSCAN(min_cluster_size=5, min_samples=5)`.
constexpr inline std::int64_t reference_row_count = 20;
constexpr inline double reference_data[] = {
    0.5917,  -0.1631, 0.0115, 0.1426,  -0.2761, 0.0007,  -0.0003, -0.6142, 0.3562, 0.2102, //
    -0.2189, -0.06,   0.1769, -0.0915, -0.085,  -0.5086, 0.1941,  0.0434,  6.151,  5.1604, //
    6.9079,  6.0849,  5.7871, 7.116,   5.975,   5.2021,  5.7771,  4.7414,  6.5772, 5.7709, //
    5.5916,  6.5899,  5.0919, 6.2945,  4.8646,  5.6358,  3.0,     -4.0,    -5.0,   7.0,
};

} // namespace oneapi::dal::hdbscan::test
