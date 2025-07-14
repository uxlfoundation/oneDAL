/*******************************************************************************
* Copyright 2023 Intel Corporation
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
#include <daal/include/services/daal_defines.h>

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/algo/covariance/common.hpp"
#include "oneapi/dal/algo/covariance/compute_types.hpp"

#include "oneapi/dal/algo/covariance/parameters/cpu/compute_parameters.hpp"

#if defined(TARGET_X86_64)
#define CPU_EXTENSION dal::detail::cpu_extension::avx512
#elif defined(TARGET_ARM)
#define CPU_EXTENSION dal::detail::cpu_extension::sve
#elif defined(TARGET_RISCV64)
#define CPU_EXTENSION dal::detail::cpu_extension::rv64
#endif

namespace oneapi::dal::covariance::parameters {

using dal::backend::context_cpu;

/// Proposes the number of rows in the data block used in variance-covariance matrix computations on CPU.
///
/// @tparam Float   The type of elements that is used in computations in covariance algorithm.
///                 The :literal:`Float` type should be at least :expr:`float` or :expr:`double`.
///
/// @param[in] ctx       Context that stores the information about the available CPU extensions
///                      and available data communication mechanisms, parallel or distributed.
/// @param[in] row_count Number of rows in the input dataset.
///
/// @return Number of rows in the data block used in variance-covariance matrix computations on CPU.
template <typename Float>
std::int64_t propose_block_size(const context_cpu& ctx, const std::int64_t row_count) {
    /// The constants are defined as the values that show the best performance results
    /// in the series of performance measurements with the varying block sizes and dataset sizes.
    std::int64_t block_size = 140l;
    if (ctx.get_enabled_cpu_extensions() == CPU_EXTENSION) {
        /// Here if AVX512 extensions are available on CPU
        if (5000l < row_count && row_count <= 50000l) {
            block_size = 1024l;
        }
    }
    return block_size;
}

std::int64_t propose_max_cols_batched(const context_cpu& ctx, const std::int64_t row_count) {
    return 4096;
}

std::int64_t propose_small_rows_threshold(const context_cpu& ctx, const std::int64_t row_count) {
    return 10'000;
}

std::int64_t propose_small_rows_max_cols_batched(const context_cpu& ctx,
                                                 const std::int64_t row_count) {
    return 1024;
}

template <typename Float, typename Task>
struct compute_parameters_cpu<Float, method::dense, Task> {
    using params_t = detail::compute_parameters<Task>;
    params_t operator()(const context_cpu& ctx,
                        const detail::descriptor_base<Task>& desc,
                        const compute_input<Task>& input) const {
        const auto& x = input.get_data();

        const auto row_count = x.get_row_count();

        const auto block = propose_block_size<Float>(ctx, row_count);
        const auto max_cols_batched = propose_max_cols_batched(ctx, row_count);
        const auto small_rows_threshold = propose_small_rows_threshold(ctx, row_count);
        const auto small_rows_max_cols_batched =
            propose_small_rows_max_cols_batched(ctx, row_count);

        return params_t{}
            .set_cpu_macro_block(block)
            .set_cpu_max_cols_batched(max_cols_batched)
            .set_cpu_small_rows_threshold(small_rows_threshold)
            .set_cpu_small_rows_max_cols_batched(small_rows_max_cols_batched);
    }
};

template struct ONEDAL_EXPORT compute_parameters_cpu<float, method::dense, task::compute>;
template struct ONEDAL_EXPORT compute_parameters_cpu<double, method::dense, task::compute>;

} // namespace oneapi::dal::covariance::parameters
