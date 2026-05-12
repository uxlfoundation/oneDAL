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

#include <cstring>
#include <limits>

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::hdbscan::backend {

template <typename Float>
static void compute_centroids(const Float* data,
                              const std::int32_t* labels,
                              std::int64_t row_count,
                              std::int64_t col_count,
                              std::int64_t cluster_count,
                              Float* centroids) {
    ONEDAL_ASSERT(cluster_count > 0);

    auto counts_arr = dal::array<std::int64_t>::zeros(cluster_count);
    std::int64_t* counts = counts_arr.get_mutable_data();
    std::memset(centroids, 0, cluster_count * col_count * sizeof(Float));

    for (std::int64_t i = 0; i < row_count; i++) {
        const std::int32_t label = labels[i];
        if (label < 0 || label >= cluster_count)
            continue;
        counts[label]++;
        Float* row_out = centroids + label * col_count;
        const Float* row_in = data + i * col_count;
        for (std::int64_t d = 0; d < col_count; d++) {
            row_out[d] += row_in[d];
        }
    }

    for (std::int64_t k = 0; k < cluster_count; k++) {
        if (counts[k] == 0)
            continue;
        Float* row = centroids + k * col_count;
        const Float inv = Float(1) / static_cast<Float>(counts[k]);
        for (std::int64_t d = 0; d < col_count; d++) {
            row[d] *= inv;
        }
    }
}

template <typename Float>
static void compute_medoids(const Float* data,
                            const std::int32_t* labels,
                            std::int64_t row_count,
                            std::int64_t col_count,
                            std::int64_t cluster_count,
                            const Float* centroids,
                            Float* medoids) {
    ONEDAL_ASSERT(cluster_count > 0);

    auto best_dist_arr = dal::array<Float>::empty(cluster_count);
    auto best_idx_arr = dal::array<std::int64_t>::empty(cluster_count);
    Float* best_dist = best_dist_arr.get_mutable_data();
    std::int64_t* best_idx = best_idx_arr.get_mutable_data();
    for (std::int64_t k = 0; k < cluster_count; k++) {
        best_dist[k] = std::numeric_limits<Float>::max();
        best_idx[k] = -1;
    }

    for (std::int64_t i = 0; i < row_count; i++) {
        const std::int32_t label = labels[i];
        if (label < 0 || label >= cluster_count)
            continue;
        const Float* pt = data + i * col_count;
        const Float* center = centroids + label * col_count;
        Float dist = Float(0);
        for (std::int64_t d = 0; d < col_count; d++) {
            const Float diff = pt[d] - center[d];
            dist += diff * diff;
        }
        if (dist < best_dist[label]) {
            best_dist[label] = dist;
            best_idx[label] = i;
        }
    }

    for (std::int64_t k = 0; k < cluster_count; k++) {
        Float* row = medoids + k * col_count;
        if (best_idx[k] >= 0) {
            std::memcpy(row, data + best_idx[k] * col_count, col_count * sizeof(Float));
        }
        else {
            std::memset(row, 0, col_count * sizeof(Float));
        }
    }
}

} // namespace oneapi::dal::hdbscan::backend
