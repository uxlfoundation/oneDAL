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

#include <cmath>
#include <limits>

#include "oneapi/dal/algo/hdbscan/backend/cpu/cluster_utils.hpp"
#include "oneapi/dal/array.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/threading.hpp"

namespace oneapi::dal::hdbscan::backend {

template <typename Float>
static Float pair_distance(const Float* a,
                           const Float* b,
                           std::int64_t col_count,
                           distance_metric metric,
                           double degree) {
    switch (metric) {
        case distance_metric::manhattan: {
            Float sum = 0;
            for (std::int64_t d = 0; d < col_count; d++)
                sum += std::abs(a[d] - b[d]);
            return sum;
        }
        case distance_metric::chebyshev: {
            Float max = 0;
            for (std::int64_t d = 0; d < col_count; d++)
                max = std::max(max, Float(std::abs(a[d] - b[d])));
            return max;
        }
        case distance_metric::minkowski: {
            double sum = 0;
            for (std::int64_t d = 0; d < col_count; d++)
                sum += std::pow(std::abs(double(a[d]) - double(b[d])), degree);
            return Float(std::pow(sum, 1.0 / degree));
        }
        case distance_metric::cosine: {
            // scikit-learn normalizes a zero row to the zero vector, i.e. distance 1.
            double dot = 0, aa = 0, bb = 0;
            for (std::int64_t d = 0; d < col_count; d++) {
                dot += double(a[d]) * b[d];
                aa += double(a[d]) * a[d];
                bb += double(b[d]) * b[d];
            }
            if (a == b)
                return Float(0);
            if (aa == 0 || bb == 0)
                return Float(1);
            return Float(std::max(0.0, 1.0 - dot / (std::sqrt(aa) * std::sqrt(bb))));
        }
        default: {
            Float sum = 0;
            for (std::int64_t d = 0; d < col_count; d++) {
                const Float diff = a[d] - b[d];
                sum += diff * diff;
            }
            return std::sqrt(sum);
        }
    }
}

template <typename Cpu, typename Float>
void compute_centroids(const Float* data,
                       const std::int32_t* labels,
                       const Float* weights,
                       std::int64_t row_count,
                       std::int64_t col_count,
                       std::int64_t cluster_count,
                       Float* centroids) {
    ONEDAL_ASSERT(cluster_count > 0);

    auto weight_sums_arr = dal::array<double>::zeros(cluster_count);
    double* weight_sums = weight_sums_arr.get_mutable_data();
    auto sums_arr = dal::array<double>::zeros(cluster_count * col_count);
    double* sums = sums_arr.get_mutable_data();

    for (std::int64_t i = 0; i < row_count; i++) {
        const std::int32_t label = labels[i];
        if (label < 0 || label >= cluster_count)
            continue;
        const double w = weights[i];
        weight_sums[label] += w;
        double* row_out = sums + label * col_count;
        const Float* row_in = data + i * col_count;
        PRAGMA_OMP_SIMD
        for (std::int64_t d = 0; d < col_count; d++) {
            row_out[d] += w * row_in[d];
        }
    }

    for (std::int64_t k = 0; k < cluster_count; k++) {
        const double inv = (weight_sums[k] > 0) ? 1.0 / weight_sums[k] : 0.0;
        PRAGMA_OMP_SIMD
        for (std::int64_t d = 0; d < col_count; d++) {
            centroids[k * col_count + d] = Float(sums[k * col_count + d] * inv);
        }
    }
}

template <typename Cpu, typename Float>
void compute_medoids(const Float* data,
                     const std::int32_t* labels,
                     const Float* weights,
                     std::int64_t row_count,
                     std::int64_t col_count,
                     std::int64_t cluster_count,
                     distance_metric metric,
                     double degree,
                     Float* medoids) {
    ONEDAL_ASSERT(cluster_count > 0);

    // Members of each cluster in row order, CSR style.
    auto offsets_arr = dal::array<std::int64_t>::zeros(cluster_count + 1);
    std::int64_t* offsets = offsets_arr.get_mutable_data();
    for (std::int64_t i = 0; i < row_count; i++) {
        if (labels[i] >= 0 && labels[i] < cluster_count)
            offsets[labels[i] + 1]++;
    }
    for (std::int64_t k = 0; k < cluster_count; k++)
        offsets[k + 1] += offsets[k];
    const std::int64_t member_count = offsets[cluster_count];
    auto members_arr = dal::array<std::int64_t>::empty(member_count > 0 ? member_count : 1);
    std::int64_t* members = members_arr.get_mutable_data();
    {
        auto fill_arr = dal::array<std::int64_t>::empty(cluster_count);
        std::int64_t* fill = fill_arr.get_mutable_data();
        for (std::int64_t k = 0; k < cluster_count; k++)
            fill[k] = offsets[k];
        for (std::int64_t i = 0; i < row_count; i++) {
            if (labels[i] >= 0 && labels[i] < cluster_count)
                members[fill[labels[i]]++] = i;
        }
    }

    auto scores_arr = dal::array<double>::empty(member_count > 0 ? member_count : 1);
    double* scores = scores_arr.get_mutable_data();
    dal::detail::threader_for(member_count, 1, [&](std::int64_t m) {
        const std::int64_t i = members[m];
        const std::int32_t k = labels[i];
        double score = 0;
        for (std::int64_t q = offsets[k]; q < offsets[k + 1]; q++) {
            const std::int64_t j = members[q];
            score += double(pair_distance(data + i * col_count,
                                          data + j * col_count,
                                          col_count,
                                          metric,
                                          degree)) *
                     weights[j];
        }
        scores[m] = score;
    });

    for (std::int64_t k = 0; k < cluster_count; k++) {
        Float* row = medoids + k * col_count;
        std::int64_t best = -1;
        for (std::int64_t q = offsets[k]; q < offsets[k + 1]; q++) {
            if (best < 0 || scores[q] < scores[best])
                best = q;
        }
        for (std::int64_t d = 0; d < col_count; d++) {
            row[d] = (best >= 0) ? data[members[best] * col_count + d] : Float(0);
        }
    }
}

#define INSTANTIATE(F)                                                   \
    template void compute_centroids<__CPU_TAG__, F>(const F*,            \
                                                    const std::int32_t*, \
                                                    const F*,            \
                                                    std::int64_t,        \
                                                    std::int64_t,        \
                                                    std::int64_t,        \
                                                    F*);                 \
    template void compute_medoids<__CPU_TAG__, F>(const F*,              \
                                                  const std::int32_t*,   \
                                                  const F*,              \
                                                  std::int64_t,          \
                                                  std::int64_t,          \
                                                  std::int64_t,          \
                                                  distance_metric,       \
                                                  double,                \
                                                  F*);

INSTANTIATE(float)
INSTANTIATE(double)

} // namespace oneapi::dal::hdbscan::backend
