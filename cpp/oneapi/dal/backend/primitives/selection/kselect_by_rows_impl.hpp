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

#include <iostream>

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

template <typename Float>
kselect_by_rows<Float>::kselect_by_rows(sycl::queue& queue,
                                        const ndshape<2>& shape,
                                        std::int64_t k) {
    if (k == 1) {
        base_.reset(new kselect_by_rows_single_col<Float>{});
        return;
    }
    const auto sg_sizes = queue.get_device().get_info<sycl::info::device::sub_group_sizes>();
    ONEDAL_ASSERT(!sg_sizes.empty());
    auto max_sg_size_iter = std::max_element(sg_sizes.begin(), sg_sizes.end());
    ONEDAL_ASSERT(max_sg_size_iter != sg_sizes.end());
    const std::uint32_t simd_width = static_cast<std::uint32_t>(*max_sg_size_iter);

    if (k <= simd_width) {
        if (simd_width == simd8) {
            std::cerr << "simd8" << std::endl;
            base_.reset(new kselect_by_rows_simd<Float, simd16>{});
            return;
        }
        if (simd_width == simd16) {
            std::cerr << "simd16" << std::endl;
            base_.reset(new kselect_by_rows_simd<Float, simd16>{});
            return;
        }
        else if (simd_width == simd32) {
            std::cerr << "simd32" << std::endl;
            base_.reset(new kselect_by_rows_simd<Float, simd32>{});
            return;
        }
        else if (simd_width == simd64) {
            std::cerr << "simd64" << std::endl;
            base_.reset(new kselect_by_rows_simd<Float, simd64>{});
            return;
        }
        else if (simd_width == simd128) {
            std::cerr << "simd128" << std::endl;
            base_.reset(new kselect_by_rows_simd<Float, simd128>{});
            return;
        }
        ONEDAL_ASSERT(false);
    }

    if ((get_heap_min_k<Float>(queue) < k) && (k < get_heap_max_k<Float>(queue))) {
            std::cerr << "kselect heap" << std::endl;
        base_.reset(new kselect_by_rows_heap<Float>{});
        return;
    }
    std::cerr << "kselect quick" << std::endl;

    { base_.reset(new kselect_by_rows_quick<Float>{ queue, shape }); }
}

#endif // ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::backend::primitives
