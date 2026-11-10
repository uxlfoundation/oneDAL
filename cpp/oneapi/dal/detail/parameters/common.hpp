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

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/detail/cpu.hpp"
#include "oneapi/dal/detail/parameters/system_parameters.hpp"

#if defined(TARGET_X86_64)
#define CPU_EXTENSION dal::detail::cpu_extension::avx512
#elif defined(TARGET_ARM)
#define CPU_EXTENSION dal::detail::cpu_extension::sve
#elif defined(TARGET_RISCV64)
#define CPU_EXTENSION dal::detail::cpu_extension::rv64
#endif

namespace oneapi::dal::detail {

/// Given the input dataset with :literal:`n` rows and :literal:`p` columns,
/// the function proposes an appropriate block size for CPU computations.
///
/// The heuristic for choosing the block size for the covariance computation is as follows:
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
std::int64_t covariance_block_size(const dal::backend::context_cpu& ctx,
                                   const system_parameters& sys_params,
                                   const std::int64_t row_count,
                                   const std::int64_t column_count);

} // namespace oneapi::dal::detail
