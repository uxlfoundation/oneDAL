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
#include "oneapi/dal/algo/basic_statistics/backend/cpu/compute_kernel.hpp"
#include "oneapi/dal/algo/basic_statistics/backend/basic_statistics_interop.hpp"

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include <daal/src/algorithms/low_order_moments/moments_online.h>
#include <daal/src/algorithms/low_order_moments/moments_batch.h>
#include <daal/src/algorithms/low_order_moments/low_order_moments_kernel.h>

namespace oneapi::dal::basic_statistics::backend {

using dal::backend::context_cpu;
using task_t = task::compute;
using input_t = compute_input<task_t>;
using result_t = compute_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

namespace daal_lom = daal::algorithms::low_order_moments;
namespace interop = dal::backend::interop;

template <daal_lom::Method Value>
using daal_method_constant = std::integral_constant<daal_lom::Method, Value>;

template <typename Method>
struct to_daal_method;

template <>
struct to_daal_method<method::dense> : daal_method_constant<daal_lom::defaultDense> {};

template <>
struct to_daal_method<method::sparse> : daal_method_constant<daal_lom::fastCSR> {};

template <typename Float, daal::internal::CpuType Cpu, typename Method>
using batch_kernel_t =
    daal_lom::internal::LowOrderMomentsBatchKernel<Float, to_daal_method<Method>::value, Cpu>;

template <typename Float, daal::internal::CpuType Cpu>
using daal_lom_online_kernel_t =
    daal_lom::internal::LowOrderMomentsOnlineKernel<Float, daal_lom::defaultDense, Cpu>;

template <typename Float>
std::int64_t propose_block_size(std::int64_t row_count, std::int64_t col_count) {
    using idx_t = std::int64_t;
    ONEDAL_ASSERT(row_count > 0);
    ONEDAL_ASSERT(col_count > 0);
    constexpr idx_t max_block_mem_size = 16 * 1024 * 1024;
    const idx_t block_of_rows_size = max_block_mem_size / (col_count * sizeof(Float));
    return std::max<idx_t>(std::min<idx_t>(row_count, idx_t(1024l)), block_of_rows_size);
}

template <typename Float>
result_t call_daal_kernel_with_weights(const context_cpu& ctx,
                                       const descriptor_t& desc,
                                       const table& data,
                                       const table& weights) {
    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(weights.has_data());

    constexpr bool is_online = true;

    const auto sample_count = data.get_row_count();
    const auto feature_count = data.get_column_count();

    ONEDAL_ASSERT(weights.get_row_count() == sample_count);
    ONEDAL_ASSERT(weights.get_column_count() == std::int64_t(1));

    row_accessor<const Float> data_accessor(data);
    row_accessor<const Float> weights_accessor(weights);

    const auto block = propose_block_size<Float>(sample_count, feature_count);
    const bk::uniform_blocking blocking(sample_count, block);

    auto daal_input = daal_lom::Input();
    auto daal_result = daal_lom::Result();
    auto daal_partial = daal_lom::PartialResult();

    const auto result_ids = get_daal_estimates_to_compute(desc);
    const auto daal_parameter = daal_lom::Parameter(result_ids);

    const auto block_count = blocking.get_block_count();
    for (std::int64_t b = 0; b < block_count; ++b) {
        const auto f_row = blocking.get_block_start_index(b);
        const auto l_row = blocking.get_block_end_index(b);
        const std::int64_t len = l_row - f_row;
        ONEDAL_ASSERT(l_row > f_row);

        auto weights_arr = weights_accessor.pull({ f_row, l_row });
        auto gen_data_block = data_accessor.pull({ f_row, l_row });
        auto data_arr = copy_immutable(std::move(gen_data_block));

        {
            auto data_ndarr = pr::ndarray<Float, 2>::wrap_mutable(data_arr, { len, feature_count });
            auto weights_ndarr = pr::ndarray<Float, 1>::wrap(weights_arr, len);

            apply_weights(ctx, weights_ndarr, data_ndarr);
        }

        const auto onedal_data = homogen_table::wrap(data_arr, len, feature_count);

        const auto daal_data = interop::convert_to_daal_table<Float>(onedal_data);

        daal_input.set(daal_lom::InputId::data, daal_data);

        if (b == std::int64_t(0)) {
            alloc_result<Float>(daal_partial, &daal_input, &daal_parameter, result_ids);
            initialize_result<Float>(daal_partial, &daal_input, &daal_parameter, result_ids);
        }

        {
            const auto status =
                interop::call_daal_kernel<Float, daal_lom_online_kernel_t>(ctx,
                                                                           daal_data.get(),
                                                                           &daal_partial,
                                                                           &daal_parameter,
                                                                           is_online);
            interop::status_to_exception(status);
        }
    }

    {
        alloc_result<Float>(daal_result, &daal_input, &daal_parameter, result_ids);

        daal_result.set(daal_lom::ResultId::maximum,
                        daal_partial.get(daal_lom::PartialResultId::partialMaximum));
        daal_result.set(daal_lom::ResultId::minimum,
                        daal_partial.get(daal_lom::PartialResultId::partialMinimum));

        daal_result.set(daal_lom::ResultId::sum,
                        daal_partial.get(daal_lom::PartialResultId::partialSum));
        daal_result.set(daal_lom::ResultId::sumSquares,
                        daal_partial.get(daal_lom::PartialResultId::partialSumSquares));
        daal_result.set(daal_lom::ResultId::sumSquaresCentered,
                        daal_partial.get(daal_lom::PartialResultId::partialSumSquaresCentered));
    }
    interop::status_to_exception(
        interop::call_daal_kernel_finalize_compute<Float, daal_lom_online_kernel_t>(
            ctx,
            daal_partial.get(daal_lom::PartialResultId::nObservations).get(),
            daal_partial.get(daal_lom::PartialResultId::partialSum).get(),
            daal_partial.get(daal_lom::PartialResultId::partialSumSquares).get(),
            daal_partial.get(daal_lom::PartialResultId::partialSumSquaresCentered).get(),
            daal_result.get(daal_lom::ResultId::mean).get(),
            daal_result.get(daal_lom::ResultId::secondOrderRawMoment).get(),
            daal_result.get(daal_lom::ResultId::variance).get(),
            daal_result.get(daal_lom::ResultId::standardDeviation).get(),
            daal_result.get(daal_lom::ResultId::variation).get(),
            &daal_parameter));

    auto result =
        get_result<Float, task_t>(desc, daal_result).set_result_options(desc.get_result_options());

    return result;
}

template <typename Float, typename Method>
result_t call_daal_kernel_without_weights(const context_cpu& ctx,
                                          const descriptor_t& desc,
                                          const table& data) {
    auto daal_method =
        std::is_same_v<method::dense, Method> ? daal_lom::defaultDense : daal_lom::fastCSR;
    const auto daal_data = interop::convert_to_daal_table<Float>(data);

    auto daal_parameter = daal_lom::Parameter(get_daal_estimates_to_compute(desc));
    auto daal_input = daal_lom::Input();
    auto daal_result = daal_lom::Result();

    daal_input.set(daal_lom::InputId::data, daal_data);

    interop::status_to_exception(
        daal_result.allocate<Float>(&daal_input, &daal_parameter, daal_method));

    interop::status_to_exception(dal::backend::dispatch_by_cpu(ctx, [&](auto cpu) {
        return batch_kernel_t<Float,
                              oneapi::dal::backend::interop::to_daal_cpu_type<decltype(cpu)>::value,
                              Method>()
            .compute(daal_data.get(), &daal_result, &daal_parameter);
    }));

    auto result =
        get_result<Float, task_t>(desc, daal_result).set_result_options(desc.get_result_options());

    return result;
}

template <typename Float, typename Method>
static result_t compute(const context_cpu& ctx, const descriptor_t& desc, const input_t& input) {
    if (input.get_weights().has_data()) {
        return call_daal_kernel_with_weights<Float>(ctx,
                                                    desc,
                                                    input.get_data(),
                                                    input.get_weights());
    }
    else {
        return call_daal_kernel_without_weights<Float, Method>(ctx, desc, input.get_data());
    }
}

template <typename Float, typename Method>
struct compute_kernel_cpu<Float, Method, task_t> {
    result_t operator()(const context_cpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute<Float, Method>(ctx, desc, input);
    }
};

template struct compute_kernel_cpu<float, method::dense, task_t>;
template struct compute_kernel_cpu<double, method::dense, task_t>;
template struct compute_kernel_cpu<float, method::sparse, task_t>;
template struct compute_kernel_cpu<double, method::sparse, task_t>;

} // namespace oneapi::dal::basic_statistics::backend
