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

#include <daal/include/services/daal_defines.h>
#include <daal/src/services/service_defines.h>

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/detail/parameters/common.hpp"

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/algo/covariance/common.hpp"
#include "oneapi/dal/algo/covariance/compute_types.hpp"

#include "oneapi/dal/algo/covariance/parameters/cpu/compute_parameters.hpp"

namespace oneapi::dal::covariance::parameters {

using dal::backend::context_cpu;

/// Given the input dataset with :literal:`n` rows and :literal:`p` columns,
///the function proposes an appropriate block size for CPU computations.
///
/// The heuristic for choosing the block size is as follows:
///
/// rows_in_cache     = max(1, floor(alpha * C / (p * sizeof(Float))))
/// blocks_per_thread = max(k, ceil(n / (threads * rows_in_cache)))
/// block_size        = ceil(n / (blocks_per_thread * threads))
///
/// where
/// :literal:`C`     - the size of some cache level (e.g., L2) in bytes,
/// :literal:`alpha` - a proportion of the cache size to be used for the blocks of rows processed by each thread,
/// :literal:`k`     - a minimum number of blocks per thread.
///
/// Chosen coefficients for the block size computation:
/// C = size of L2
/// alpha = 2^-1.5 (AVX-512) / 1/4 (AVX2)
/// k = 2
///
/// @tparam Float   The type of elements that is used in computations in covariance algorithm.
///                 The :literal:`Float` type should be at least :expr:`float` or :expr:`double`.
///
/// @param[in] ctx          Context that stores the information about the available CPU extensions
///                         and available data communication mechanisms, parallel or distributed.
/// @param[in] sys_params   System parameters that provide the sizes of the CPU caches
///                         and the maximal number of threads.
/// @param[in] row_count    Number of rows in the input dataset, n.
/// @param[in] column_count Number of columns in the input dataset, p.
///
/// @return Number of rows in the data block used in variance-covariance matrix computations on CPU.
template <typename Float>
std::int64_t propose_block_size(const context_cpu& ctx,
                                const dal::detail::system_parameters& sys_params,
                                const std::int64_t row_count,
                                const std::int64_t column_count) {
    return dal::detail::covariance_block_size<Float>(ctx, sys_params, row_count, column_count);
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
        const auto column_count = x.get_column_count();

        auto params = params_t{};
        const auto block = propose_block_size<Float>(ctx, params, row_count, column_count);
        const auto max_cols_batched = propose_max_cols_batched(ctx, row_count);
        const auto small_rows_threshold = propose_small_rows_threshold(ctx, row_count);
        const auto small_rows_max_cols_batched =
            propose_small_rows_max_cols_batched(ctx, row_count);

        return params.set_cpu_macro_block(block)
            .set_cpu_max_cols_batched(max_cols_batched)
            .set_cpu_small_rows_threshold(small_rows_threshold)
            .set_cpu_small_rows_max_cols_batched(small_rows_max_cols_batched);
    }
};

template struct ONEDAL_EXPORT compute_parameters_cpu<float, method::dense, task::compute>;
template struct ONEDAL_EXPORT compute_parameters_cpu<double, method::dense, task::compute>;

} // namespace oneapi::dal::covariance::parameters
