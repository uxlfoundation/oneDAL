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

#include <daal/src/services/service_defines.h>

#include "oneapi/dal/detail/parameters/common.hpp"

namespace oneapi::dal::detail {

inline std::int64_t ceil_div(std::int64_t a, std::int64_t b) {
    return (a + b - 1) / b;
}

template <typename Float>
std::int64_t covariance_block_size(const dal::backend::context_cpu& ctx,
                                   const system_parameters& sys_params,
                                   const std::int64_t row_count,
                                   const std::int64_t column_count) {
#if defined(TARGET_X86_64)
    if (daal_check_is_intel_cpu()) {
        const std::int64_t threads =
            std::max<std::int64_t>(1, sys_params.get_max_number_of_threads());

        // Bytes one block of rows may occupy: alpha * C.
        static const std::int64_t budget_avx512 =
            static_cast<std::int64_t>(0.353553 * double(sys_params.get_l2_cache_size()));
        static const std::int64_t budget_avx2 = sys_params.get_l2_cache_size() / 4;
        const bool avx512 = ctx.get_enabled_cpu_extensions() >= dal::detail::cpu_extension::avx512;
        const std::int64_t budget = avx512 ? budget_avx512 : budget_avx2;

        std::int64_t row_bytes = column_count * std::int64_t(sizeof(Float));
        std::int64_t b_cache = std::max<std::int64_t>(1, budget / row_bytes);

        // Whole number of blocks per thread, so the static partition gives every thread the same count.
        const std::int64_t m = std::max<std::int64_t>(2, ceil_div(row_count, threads * b_cache));
        std::int64_t block = ceil_div(row_count, m * threads);
        return std::min(block, row_count);
    }
#endif
    /// The constants are defined as the values that show the best performance results
    /// in the series of performance measurements with the varying block sizes and dataset sizes.
    std::int64_t block_size = 140l;
    if (ctx.get_enabled_cpu_extensions() == CPU_EXTENSION) {
        if (5000l < row_count && row_count <= 50000l) {
            block_size = 1024l;
        }
    }
    return block_size;
}

#define INSTANTIATE(T)                                            \
    template ONEDAL_EXPORT std::int64_t covariance_block_size<T>( \
        const dal::backend::context_cpu& ctx,                     \
        const system_parameters& sys_params,                      \
        const std::int64_t row_count,                             \
        const std::int64_t column_count);

INSTANTIATE(float)
INSTANTIATE(double)

} // namespace oneapi::dal::detail
