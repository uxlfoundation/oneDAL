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

#include <cstdint>
#include <random>
#include <vector>

namespace oneapi::dal::test::engine {

/// Dense, deterministic, well-separated blobs in row-major order, `per_cluster` consecutive rows
/// per cluster. Cluster `c` is centered at `separation * (c + 1)`, with the sign alternating over
/// columns, and each coordinate is offset uniformly by up to `spread / 2`.
///
/// @tparam Float Floating-point type of the data
///
/// @param[in] per_cluster   Number of rows in each cluster
/// @param[in] cluster_count Number of clusters
/// @param[in] column_count  Number of columns
/// @param[in] separation    Distance between consecutive cluster centers along each column
/// @param[in] spread        Width of each cluster along each column
/// @param[in] seed          Seed of the generator
///
/// @return `per_cluster * cluster_count` rows of `column_count` values
template <typename Float>
std::vector<Float> make_blobs(std::int64_t per_cluster,
                              std::int64_t cluster_count,
                              std::int64_t column_count,
                              Float separation,
                              Float spread,
                              std::uint32_t seed = 777u) {
    std::vector<Float> data(per_cluster * cluster_count * column_count);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<Float> offset(-spread / 2, spread / 2);

    std::int64_t pos = 0;
    for (std::int64_t c = 0; c < cluster_count; c++) {
        for (std::int64_t i = 0; i < per_cluster; i++) {
            for (std::int64_t j = 0; j < column_count; j++) {
                const Float center = separation * static_cast<Float>(c + 1) *
                                     static_cast<Float>(j % 2 == 0 ? 1 : -1);
                data[pos++] = center + offset(rng);
            }
        }
    }
    return data;
}

} // namespace oneapi::dal::test::engine
