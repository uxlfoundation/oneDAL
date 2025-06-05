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

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/dispatcher.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/algo/pca/common.hpp"
#include "oneapi/dal/algo/pca/train_types.hpp"

#include "oneapi/dal/algo/pca/parameters/cpu/train_parameters.hpp"

#if defined(TARGET_X86_64)
#define CPU_EXTENSION dal::detail::cpu_extension::avx512
#elif defined(TARGET_ARM)
#define CPU_EXTENSION dal::detail::cpu_extension::sve
#elif defined(TARGET_RISCV64)
#define CPU_EXTENSION dal::detail::cpu_extension::rv64
#endif

namespace oneapi::dal::pca::parameters {

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

template <typename Float>
std::int64_t propose_grain_size(const context_cpu& ctx, const std::int64_t row_count) {
    return detail::train_parameters<task::dim_reduction>{}.get_cpu_grain_size();
}

template <typename Float>
std::int64_t propose_max_cols_batched(const context_cpu& ctx, const std::int64_t row_count) {
    return 4096;
}

template <typename Float>
std::int64_t propose_small_rows_threshold(const context_cpu& ctx, const std::int64_t row_count) {
    return 10'000;
}

template <typename Float>
std::int64_t propose_small_rows_max_cols_batched(const context_cpu& ctx,
                                                 const std::int64_t row_count) {
    return 1024;
}

template <typename Float, typename Method>
struct train_parameters_cpu<Float, Method, task::dim_reduction> {
    using params_t = detail::train_parameters<task::dim_reduction>;
    params_t operator()(const context_cpu& ctx,
                        const detail::descriptor_base<task::dim_reduction>& desc,
                        const train_input<task::dim_reduction>& input) const {
        return params_t{};
    }
    params_t operator()(const context_cpu& ctx,
                        const detail::descriptor_base<task::dim_reduction>& desc,
                        const partial_train_input<task::dim_reduction>& input) const {
        return params_t{};
    }
    params_t operator()(const context_cpu& ctx,
                        const detail::descriptor_base<task::dim_reduction>& desc,
                        const partial_train_result<task::dim_reduction>& input) const {
        return params_t{};
    }
};

template <typename Float>
struct train_parameters_cpu<Float, method::cov, task::dim_reduction> {
    using params_t = detail::train_parameters<task::dim_reduction>;
    params_t operator()(const context_cpu& ctx,
                        const detail::descriptor_base<task::dim_reduction>& desc,
                        const train_input<task::dim_reduction>& input) const {
        const auto& x_train = input.get_data();

        const auto r_count = x_train.get_row_count();

        const std::int64_t block = propose_block_size<Float>(ctx, r_count);
        const std::int64_t grain_size = propose_grain_size<Float>(ctx, r_count);
        const std::int64_t max_cols_batched = propose_max_cols_batched<Float>(ctx, r_count);
        const std::int64_t small_rows_threshold = propose_small_rows_threshold<Float>(ctx, r_count);
        const std::int64_t small_rows_max_cols_batched =
            propose_small_rows_max_cols_batched<Float>(ctx, r_count);

        params_t out{};
        out.set_cpu_macro_block(block);
        out.set_cpu_grain_size(grain_size);
        out.set_cpu_max_cols_batched(max_cols_batched);
        out.set_cpu_small_rows_threshold(small_rows_threshold);
        out.set_cpu_small_rows_max_cols_batched(small_rows_max_cols_batched);
        return out;
    }
    params_t operator()(const context_cpu& ctx,
                        const detail::descriptor_base<task::dim_reduction>& desc,
                        const partial_train_input<task::dim_reduction>& input) const {
        const auto& x_train = input.get_data();

        const auto r_count = x_train.get_row_count();

        const std::int64_t block = propose_block_size<Float>(ctx, r_count);
        const std::int64_t grain_size = propose_grain_size<Float>(ctx, r_count);
        const std::int64_t max_cols_batched = propose_max_cols_batched<Float>(ctx, r_count);
        const std::int64_t small_rows_threshold = propose_small_rows_threshold<Float>(ctx, r_count);
        const std::int64_t small_rows_max_cols_batched =
            propose_small_rows_max_cols_batched<Float>(ctx, r_count);

        params_t out{};
        out.set_cpu_macro_block(block);
        out.set_cpu_grain_size(grain_size);
        out.set_cpu_max_cols_batched(max_cols_batched);
        out.set_cpu_small_rows_threshold(small_rows_threshold);
        out.set_cpu_small_rows_max_cols_batched(small_rows_max_cols_batched);
        return out;
    }
    params_t operator()(const context_cpu& ctx,
                        const detail::descriptor_base<task::dim_reduction>& desc,
                        const partial_train_result<task::dim_reduction>& input) const {
        const auto& r_count_table = input.get_partial_n_rows();

        row_accessor<const std::int32_t> acc{ r_count_table };
        const auto ary = acc.pull({ 0, 1 });
        const auto r_count = (ary.get_count() > 0 ? ary[0] : 0);

        const std::int64_t block = propose_block_size<Float>(ctx, r_count);
        const std::int64_t grain_size = propose_grain_size<Float>(ctx, r_count);
        const std::int64_t max_cols_batched = propose_max_cols_batched<Float>(ctx, r_count);
        const std::int64_t small_rows_threshold = propose_small_rows_threshold<Float>(ctx, r_count);
        const std::int64_t small_rows_max_cols_batched =
            propose_small_rows_max_cols_batched<Float>(ctx, r_count);

        params_t out{};
        out.set_cpu_macro_block(block);
        out.set_cpu_grain_size(grain_size);
        out.set_cpu_max_cols_batched(max_cols_batched);
        out.set_cpu_small_rows_threshold(small_rows_threshold);
        out.set_cpu_small_rows_max_cols_batched(small_rows_max_cols_batched);
        return out;
    }
};

template struct ONEDAL_EXPORT train_parameters_cpu<float, method::cov, task::dim_reduction>;
template struct ONEDAL_EXPORT train_parameters_cpu<double, method::cov, task::dim_reduction>;
template struct ONEDAL_EXPORT train_parameters_cpu<float, method::precomputed, task::dim_reduction>;
template struct ONEDAL_EXPORT
    train_parameters_cpu<double, method::precomputed, task::dim_reduction>;
template struct ONEDAL_EXPORT train_parameters_cpu<float, method::svd, task::dim_reduction>;
template struct ONEDAL_EXPORT train_parameters_cpu<double, method::svd, task::dim_reduction>;
} // namespace oneapi::dal::pca::parameters
