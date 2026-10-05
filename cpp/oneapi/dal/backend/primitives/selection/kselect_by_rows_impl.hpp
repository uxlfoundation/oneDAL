/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/backend/primitives/selection/kselect_by_rows.hpp"
#include "oneapi/dal/backend/primitives/selection/kselect_by_rows_heap.hpp"
#include "oneapi/dal/backend/primitives/selection/kselect_by_rows_simd.hpp"
#include "oneapi/dal/backend/primitives/selection/kselect_by_rows_quick.hpp"
#include "oneapi/dal/backend/primitives/selection/kselect_by_rows_single_col.hpp"

namespace oneapi::dal::backend::primitives {

constexpr std::uint32_t simd8 = 8;
constexpr std::uint32_t simd16 = 16;
constexpr std::uint32_t simd32 = 32;
constexpr std::uint32_t simd64 = 64;
constexpr std::uint32_t simd128 = 128;

#ifdef ONEDAL_DATA_PARALLEL

/// Widest sub-group the device reports
///
/// @param[in] queue The queue whose device is asked
///
/// @return The largest supported sub-group size
inline std::uint32_t get_kselect_simd_width(const sycl::queue& queue) {
    const auto sg_sizes = queue.get_device().get_info<sycl::info::device::sub_group_sizes>();
    ONEDAL_ASSERT(!sg_sizes.empty());
    auto max_sg_size_iter = std::max_element(sg_sizes.begin(), sg_sizes.end());
    ONEDAL_ASSERT(max_sg_size_iter != sg_sizes.end());
    return static_cast<std::uint32_t>(*max_sg_size_iter);
}

/// Which `kselect_by_rows_base` implementation the dispatch picks
enum class kselect_by_rows_kind { single_col, simd, heap, quick };

/// Pick the k-selection implementation for the given `k` and device
///
/// Shared by the `kselect_by_rows` constructor and
/// `kselect_by_rows_scratch_size`, so the thresholds have a single definition.
/// A sub-group width without a `kselect_by_rows_simd` instantiation falls
/// through to the heap or quick path, as it did before the thresholds moved
/// here.
///
/// @tparam Float Floating-point type of the values being selected
///
/// @param[in] queue The queue the selector is constructed on
/// @param[in] k     Number of smallest values selected per row
///
/// @return The implementation the constructor instantiates
template <typename Float>
inline kselect_by_rows_kind get_kselect_by_rows_kind(const sycl::queue& queue, std::int64_t k) {
    if (k == 1) {
        return kselect_by_rows_kind::single_col;
    }
    const std::uint32_t simd_width = get_kselect_simd_width(queue);
    const bool simd_available = (simd_width == simd8) || (simd_width == simd16) ||
                                (simd_width == simd32) || (simd_width == simd64) ||
                                (simd_width == simd128);
    if (k <= simd_width && simd_available) {
        return kselect_by_rows_kind::simd;
    }
    if ((get_heap_min_k<Float>(queue) < k) && (k < get_heap_max_k<Float>(queue))) {
        return kselect_by_rows_kind::heap;
    }
    return kselect_by_rows_kind::quick;
}

template <typename Float>
kselect_by_rows<Float>::kselect_by_rows(sycl::queue& queue,
                                        const ndshape<2>& shape,
                                        std::int64_t k) {
    switch (get_kselect_by_rows_kind<Float>(queue, k)) {
        case kselect_by_rows_kind::single_col:
            base_.reset(new kselect_by_rows_single_col<Float>{});
            return;
        case kselect_by_rows_kind::heap: base_.reset(new kselect_by_rows_heap<Float>{}); return;
        case kselect_by_rows_kind::quick:
            base_.reset(new kselect_by_rows_quick<Float>{ queue, shape });
            return;
        case kselect_by_rows_kind::simd: break;
    }

    switch (get_kselect_simd_width(queue)) {
        case simd8:
        case simd16: base_.reset(new kselect_by_rows_simd<Float, simd16>{}); return;
        case simd32: base_.reset(new kselect_by_rows_simd<Float, simd32>{}); return;
        case simd64: base_.reset(new kselect_by_rows_simd<Float, simd64>{}); return;
        case simd128: base_.reset(new kselect_by_rows_simd<Float, simd128>{}); return;
        default: ONEDAL_ASSERT(false); return;
    }
}

template <typename Float>
std::int64_t kselect_by_rows_scratch_size(const sycl::queue& queue,
                                          const ndshape<2>& shape,
                                          std::int64_t k) {
    if (get_kselect_by_rows_kind<Float>(queue, k) != kselect_by_rows_kind::quick) {
        return 0;
    }
    // Quick select keeps a private copy of the matrix, a matching index matrix
    // and the random pivot sequence.
    constexpr std::int64_t saturated = oneapi::dal::detail::limits<std::int64_t>::max();
    constexpr std::int64_t per_element =
        static_cast<std::int64_t>(sizeof(Float)) + static_cast<std::int64_t>(sizeof(std::int32_t));
    const std::int64_t row_count = shape[0];
    const std::int64_t column_count = shape[1];
    const std::int64_t rnd_seq_bytes =
        std::min(column_count, kselect_by_rows_quick<Float>::max_rnd_seq_size) *
        static_cast<std::int64_t>(sizeof(Float));
    if (column_count > (saturated - rnd_seq_bytes) / per_element / row_count) {
        return saturated;
    }
    return row_count * column_count * per_element + rnd_seq_bytes;
}

#endif // ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::backend::primitives
