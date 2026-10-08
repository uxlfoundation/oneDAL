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

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <new>

#include <daal/src/algorithms/service_kernel_math.h>

#include "oneapi/dal/algo/hdbscan/backend/cpu/cluster_utils.hpp"
#include "oneapi/dal/array.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/threading.hpp"

namespace oneapi::dal::hdbscan::backend {

namespace daal_math = daal::algorithms::internal;
namespace interop = dal::backend::interop;

/// Create the DAAL pairwise distance of `metric` between the rows of `nt` and themselves.
///
/// Euclidean and cosine are the final distances after `finalize`; manhattan is minkowski with
/// degree 1, which `finalize` leaves as is.
///
/// @tparam Float Floating-point type
/// @tparam cpu   DAAL CPU type
///
/// @param[in] nt     Row-major rows the distances are taken between
/// @param[in] metric Distance metric of the fit
/// @param[in] degree Minkowski degree, used only for `distance_metric::minkowski`
template <typename Float, daal::internal::CpuType cpu>
static std::unique_ptr<daal_math::PairwiseDistances<Float, cpu>> make_pairwise_distances(
    const daal::data_management::NumericTable& nt,
    distance_metric metric,
    double degree) {
    switch (metric) {
        case distance_metric::euclidean:
            return std::make_unique<daal_math::EuclideanDistances<Float, cpu>>(nt, nt, true);
        case distance_metric::cosine:
            return std::make_unique<daal_math::CosineDistances<Float, cpu>>(nt, nt);
        case distance_metric::manhattan:
            return std::make_unique<daal_math::MinkowskiDistances<Float, cpu>>(nt, nt, true, 1.0);
        case distance_metric::minkowski:
            return std::make_unique<daal_math::MinkowskiDistances<Float, cpu>>(nt,
                                                                               nt,
                                                                               true,
                                                                               degree);
        case distance_metric::chebyshev:
            return std::make_unique<daal_math::ChebyshevDistances<Float, cpu>>(nt, nt);
    }
    throw invalid_argument(dal::detail::error_messages::unknown_distance_type());
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

    if (member_count == 0) {
        for (std::int64_t i = 0; i < cluster_count * col_count; i++)
            medoids[i] = Float(0);
        return;
    }

    // The members of each cluster are contiguous here, so a cluster's distance block is one
    // batch of the DAAL primitive against a contiguous range of rows.
    auto sorted_arr = dal::array<Float>::empty(member_count * col_count);
    Float* sorted = sorted_arr.get_mutable_data();
    auto zero_row_arr = dal::array<std::uint8_t>::zeros(member_count);
    std::uint8_t* zero_row = zero_row_arr.get_mutable_data();
    dal::detail::threader_for(member_count, 1, [&](std::int64_t m) {
        const Float* src = data + members[m] * col_count;
        bool is_zero = true;
        for (std::int64_t d = 0; d < col_count; d++) {
            sorted[m * col_count + d] = src[d];
            is_zero = is_zero && src[d] == Float(0);
        }
        zero_row[m] = is_zero;
    });

    constexpr auto cpu = interop::to_daal_cpu_type<Cpu>::value;
    const auto nt = interop::convert_to_daal_homogen_table(sorted_arr, member_count, col_count);
    const auto dist = make_pairwise_distances<Float, cpu>(*nt, metric, degree);
    interop::status_to_exception(dist->init());

    // 128 x 256 tiles keep each `finalize` below the size at which it would thread itself.
    constexpr std::int64_t row_block = 128;
    constexpr std::int64_t col_block = 256;
    auto task_offsets_arr = dal::array<std::int64_t>::zeros(cluster_count + 1);
    std::int64_t* task_offsets = task_offsets_arr.get_mutable_data();
    for (std::int64_t k = 0; k < cluster_count; k++) {
        const std::int64_t size = offsets[k + 1] - offsets[k];
        task_offsets[k + 1] = task_offsets[k] + (size + row_block - 1) / row_block;
    }

    auto scores_arr = dal::array<double>::zeros(member_count);
    double* scores = scores_arr.get_mutable_data();
    const bool is_cosine = metric == distance_metric::cosine;
    dal::detail::tls<Float*> tile_tls([]() {
        return new (std::nothrow) Float[row_block * col_block];
    });
    std::atomic<bool> alloc_failed{ false };
    daal::services::Status status;
    dal::detail::mutex status_mutex;
    dal::detail::threader_for(task_offsets[cluster_count], 1, [&](std::int64_t task) {
        const std::int64_t k =
            std::upper_bound(task_offsets, task_offsets + cluster_count + 1, task) - task_offsets -
            1;
        const std::int64_t first = offsets[k];
        const std::int64_t last = offsets[k + 1];
        const std::int64_t i_begin = first + (task - task_offsets[k]) * row_block;
        const std::int64_t i_count = std::min(row_block, last - i_begin);

        Float* block = tile_tls.local();
        if (!block) {
            alloc_failed = true;
            return;
        }
        for (std::int64_t j_begin = first; j_begin < last; j_begin += col_block) {
            const std::int64_t j_count = std::min(col_block, last - j_begin);
            auto s = dist->computeBatch(sorted + i_begin * col_count,
                                        sorted + j_begin * col_count,
                                        i_begin,
                                        i_count,
                                        j_begin,
                                        j_count,
                                        block);
            if (s) {
                s = dist->finalize(i_count * j_count, block);
            }
            if (!s) {
                std::lock_guard<dal::detail::mutex> lock(status_mutex);
                status = s;
                return;
            }
            for (std::int64_t i = 0; i < i_count; i++) {
                const std::int64_t mi = i_begin + i;
                double score = 0;
                for (std::int64_t j = 0; j < j_count; j++) {
                    const std::int64_t mj = j_begin + j;
                    double d = block[i * j_count + j];
                    if (mi == mj) {
                        d = 0;
                    }
                    else if (is_cosine) {
                        // scikit-learn normalizes a zero row to the zero vector, i.e. distance 1.
                        d = (zero_row[mi] || zero_row[mj]) ? 1.0 : std::max(0.0, d);
                    }
                    score += d * weights[members[mj]];
                }
                scores[mi] += score;
            }
        }
    });
    tile_tls.reduce([](Float* block) {
        delete[] block;
    });
    if (alloc_failed) {
        throw host_bad_alloc();
    }
    interop::status_to_exception(status);

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
