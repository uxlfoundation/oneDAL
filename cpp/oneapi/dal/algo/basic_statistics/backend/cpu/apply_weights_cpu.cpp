/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/algo/basic_statistics/backend/cpu/apply_weights.hpp"
#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/backend/common.hpp"

namespace oneapi::dal::basic_statistics::backend {

template <typename Cpu, typename Float>
std::int64_t propose_threading_block_size(std::int64_t row_count, std::int64_t col_count) {
    using idx_t = std::int64_t;
    ONEDAL_ASSERT(row_count > 0);
    ONEDAL_ASSERT(col_count > 0);
    constexpr idx_t max_block_mem_size = 512 * 1024;
    const idx_t block_of_rows_size = max_block_mem_size / (col_count * sizeof(Float));
    return std::max<idx_t>(std::min<idx_t>(row_count, idx_t(128l)), block_of_rows_size);
}

template <typename Cpu, typename Float>
std::pair<std::int64_t, std::int64_t> extract_and_check_dimensions(
    const pr::ndview<Float, 1>& weights,
    pr::ndview<Float, 2>& samples) {
    ONEDAL_ASSERT(weights.has_data());
    ONEDAL_ASSERT(samples.has_mutable_data());

    const auto r_count = samples.get_dimension(0);
    const auto c_count = samples.get_dimension(1);
    ONEDAL_ASSERT(weights.get_count() == r_count);

    return { r_count, c_count };
}

template <typename Cpu, typename Float>
void apply_weights_single_thread(const pr::ndview<Float, 1>& weights,
                                 pr::ndview<Float, 2>& samples) {
    const auto [r_count, c_count] = extract_and_check_dimensions<Cpu, Float>(weights, samples);

    const auto* const weights_ptr = weights.get_data();
    auto* const samples_ptr = samples.get_mutable_data();
    const auto samples_str = samples.get_leading_stride();

    for (std::int64_t r = 0; r < r_count; ++r) {
        const auto weight = weights_ptr[r];
        auto* const row = samples_ptr + r * samples_str;

        for (std::int64_t c = 0; c < c_count; ++c) {
            row[c] *= weight;
        }
    }
}

template <typename Cpu, typename Float>
void apply_weights(const pr::ndview<Float, 1>& weights, pr::ndview<Float, 2>& samples) {
    const auto [r_count, c_count] = extract_and_check_dimensions<Cpu, Float>(weights, samples);

    const auto threading_block = propose_threading_block_size<Cpu, Float>(r_count, c_count);

    const bk::uniform_blocking blocking(r_count, threading_block);
    const auto block_count = blocking.get_block_count();

    de::threader_for(block_count, block_count, [&](std::int64_t b) -> void {
        const auto f_row = blocking.get_block_start_index(b);
        const auto l_row = blocking.get_block_end_index(b);

        const auto w_block = weights.get_slice(f_row, l_row);
        auto s_block = samples.get_row_slice(f_row, l_row);

        apply_weights_single_thread<Cpu, Float>(w_block, s_block);
    });
}

#define INSTANTIATE(F)                                                                    \
    template std::int64_t propose_threading_block_size<__CPU_TAG__, F>(std::int64_t,      \
                                                                       std::int64_t);     \
    template void apply_weights<__CPU_TAG__>(const pr::ndview<F, 1>&, pr::ndview<F, 2>&); \
    template void apply_weights_single_thread<__CPU_TAG__>(const pr::ndview<F, 1>&,       \
                                                           pr::ndview<F, 2>&);

INSTANTIATE(float)
INSTANTIATE(double)

} // namespace oneapi::dal::basic_statistics::backend
