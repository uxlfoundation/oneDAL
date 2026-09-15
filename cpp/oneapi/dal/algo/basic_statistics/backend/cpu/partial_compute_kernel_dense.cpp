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
#include "oneapi/dal/algo/basic_statistics/backend/cpu/partial_compute_kernel.hpp"
#include "oneapi/dal/algo/basic_statistics/backend/basic_statistics_interop.hpp"

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include <daal/src/algorithms/low_order_moments/moments_online.h>
#include <daal/src/algorithms/low_order_moments/low_order_moments_kernel.h>

namespace oneapi::dal::basic_statistics::backend {

using dal::backend::context_cpu;
using task_t = task::compute;
using input_t = partial_compute_input<task_t>;
using result_t = partial_compute_result<task_t>;
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
using daal_lom_online_kernel_t =
    daal_lom::internal::LowOrderMomentsOnlineKernel<Float, to_daal_method<Method>::value, Cpu>;

template <typename Float, daal::internal::CpuType Cpu>
using daal_lom_online_dense_kernel_t =
    daal_lom::internal::LowOrderMomentsOnlineKernel<Float, daal_lom::defaultDense, Cpu>;

template <typename Float, typename Task>
inline auto get_partial_result(daal_lom::PartialResult daal_partial_result,
                               const descriptor_t& desc) {
    auto result = partial_compute_result();
    const auto res_op = desc.get_result_options();
    const auto result_ids = get_daal_estimates_to_compute(desc);
    result.set_partial_n_rows(interop::convert_from_daal_homogen_table<Float>(
        daal_partial_result.get(daal_lom::PartialResultId::nObservations)));
    if (result_ids == daal_lom::estimatesMinMax || res_op.test(result_options::min) ||
        res_op.test(result_options::max)) {
        result.set_partial_min(interop::convert_from_daal_homogen_table<Float>(
            daal_partial_result.get(daal_lom::PartialResultId::partialMinimum)));
        result.set_partial_max(interop::convert_from_daal_homogen_table<Float>(
            daal_partial_result.get(daal_lom::PartialResultId::partialMaximum)));
    }
    if (result_ids == daal_lom::estimatesMeanVariance || result_ids == daal_lom::estimatesAll) {
        result.set_partial_sum(interop::convert_from_daal_homogen_table<Float>(
            daal_partial_result.get(daal_lom::PartialResultId::partialSum)));
        result.set_partial_sum_squares(interop::convert_from_daal_homogen_table<Float>(
            daal_partial_result.get(daal_lom::PartialResultId::partialSumSquares)));
        result.set_partial_sum_squares_centered(interop::convert_from_daal_homogen_table<Float>(
            daal_partial_result.get(daal_lom::PartialResultId::partialSumSquaresCentered)));
    }
    return result;
}

template <typename Float, typename Task>
result_t call_daal_kernel_with_weights(const context_cpu& ctx,
                                       const descriptor_t& desc,
                                       const partial_compute_input<Task>& input) {
    auto data = input.get_data();
    auto weights = input.get_weights();
    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(weights.has_data());

    constexpr bool is_online = true;

    ONEDAL_ASSERT(weights.get_row_count() == data.get_row_count());
    ONEDAL_ASSERT(weights.get_column_count() == std::int64_t(1));

    auto daal_input = daal_lom::Input();
    auto daal_partial = daal_lom::PartialResult();

    const auto res_op = desc.get_result_options();

    const auto input_ = input.get_prev();
    row_accessor<const Float> data_accessor(data);
    row_accessor<const Float> weights_accessor(weights);
    const auto result_ids = get_daal_estimates_to_compute(desc);
    const auto daal_parameter = daal_lom::Parameter(result_ids);
    auto weights_arr = weights_accessor.pull();
    auto gen_data_block = data_accessor.pull();
    auto data_arr = copy_immutable(std::move(gen_data_block));
    {
        auto data_ndarr =
            pr::ndarray<Float, 2>::wrap_mutable(data_arr,
                                                { data.get_row_count(), data.get_column_count() });
        auto weights_ndarr = pr::ndarray<Float, 1>::wrap(weights_arr, data.get_row_count());

        apply_weights(ctx, weights_ndarr, data_ndarr);
    }

    const auto daal_data = interop::convert_to_daal_homogen_table<Float>(data_arr,
                                                                         data.get_row_count(),
                                                                         data.get_column_count());

    daal_input.set(daal_lom::InputId::data, daal_data);
    {
        alloc_result<Float>(daal_partial, &daal_input, &daal_parameter, result_ids);
        initialize_result<Float>(daal_partial, &daal_input, &daal_parameter, result_ids);
    }
    const bool has_nobs_data = input_.get_partial_n_rows().has_data();
    if (has_nobs_data) {
        auto daal_nobs = interop::copy_to_daal_homogen_table<Float>(input_.get_partial_n_rows());
        daal_partial.set(daal_lom::PartialResultId::nObservations, daal_nobs);
        if (result_ids == daal_lom::estimatesMinMax || res_op.test(result_options::min) ||
            res_op.test(result_options::max)) {
            auto daal_partial_max =
                interop::copy_to_daal_homogen_table<Float>(input_.get_partial_max());
            auto daal_partial_min =
                interop::copy_to_daal_homogen_table<Float>(input_.get_partial_min());
            daal_partial.set(daal_lom::PartialResultId::partialMaximum, daal_partial_max);
            daal_partial.set(daal_lom::PartialResultId::partialMinimum, daal_partial_min);
        }
        if (result_ids == daal_lom::estimatesMeanVariance || result_ids == daal_lom::estimatesAll) {
            auto daal_partial_sums =
                interop::copy_to_daal_homogen_table<Float>(input_.get_partial_sum());
            auto daal_partial_sum_squares =
                interop::copy_to_daal_homogen_table<Float>(input_.get_partial_sum_squares());
            auto daal_partial_sum_squares_centered = interop::copy_to_daal_homogen_table<Float>(
                input_.get_partial_sum_squares_centered());

            daal_partial.set(daal_lom::PartialResultId::partialSum, daal_partial_sums);
            daal_partial.set(daal_lom::PartialResultId::partialSumSquaresCentered,
                             daal_partial_sum_squares_centered);

            daal_partial.set(daal_lom::PartialResultId::partialSumSquares,
                             daal_partial_sum_squares);
        }
        {
            interop::status_to_exception(
                interop::call_daal_kernel<Float, daal_lom_online_dense_kernel_t>(ctx,
                                                                                 daal_data.get(),
                                                                                 &daal_partial,
                                                                                 &daal_parameter,
                                                                                 is_online));
        }
        auto result = get_partial_result<Float, task_t>(daal_partial, desc);

        return result;
    }
    else {
        {
            interop::status_to_exception(
                interop::call_daal_kernel<Float, daal_lom_online_dense_kernel_t>(ctx,
                                                                                 daal_data.get(),
                                                                                 &daal_partial,
                                                                                 &daal_parameter,
                                                                                 is_online));
        }
        auto result = get_partial_result<Float, task_t>(daal_partial, desc);
        return result;
    }
}

template <typename Float, typename Method, typename Task>
result_t call_daal_kernel_without_weights(const context_cpu& ctx,
                                          const descriptor_t& desc,
                                          const partial_compute_input<Task>& input) {
    auto data = input.get_data();
    ONEDAL_ASSERT(data.has_data());
    constexpr bool is_online = true;

    auto daal_input = daal_lom::Input();
    auto daal_partial = daal_lom::PartialResult();

    const auto input_ = input.get_prev();

    const auto result_ids = get_daal_estimates_to_compute(desc);
    const auto daal_parameter = daal_lom::Parameter(result_ids);

    const auto daal_data = interop::convert_to_daal_table<Float>(data);

    const auto res_op = desc.get_result_options();

    daal_input.set(daal_lom::InputId::data, daal_data);
    const bool has_nobs_data = input_.get_partial_n_rows().has_data();
    {
        alloc_result<Float>(daal_partial, &daal_input, &daal_parameter, result_ids);
        initialize_result<Float>(daal_partial, &daal_input, &daal_parameter, result_ids);
    }
    if (has_nobs_data) {
        auto daal_nobs = interop::copy_to_daal_homogen_table<Float>(input_.get_partial_n_rows());
        daal_partial.set(daal_lom::PartialResultId::nObservations, daal_nobs);
        if (result_ids == daal_lom::estimatesMinMax || res_op.test(result_options::min) ||
            res_op.test(result_options::max)) {
            auto daal_partial_max =
                interop::copy_to_daal_homogen_table<Float>(input_.get_partial_max());
            auto daal_partial_min =
                interop::copy_to_daal_homogen_table<Float>(input_.get_partial_min());
            daal_partial.set(daal_lom::PartialResultId::partialMaximum, daal_partial_max);
            daal_partial.set(daal_lom::PartialResultId::partialMinimum, daal_partial_min);
        }
        if (result_ids == daal_lom::estimatesMeanVariance || result_ids == daal_lom::estimatesAll) {
            auto daal_partial_sums =
                interop::copy_to_daal_homogen_table<Float>(input_.get_partial_sum());
            auto daal_partial_sum_squares =
                interop::copy_to_daal_homogen_table<Float>(input_.get_partial_sum_squares());
            auto daal_partial_sum_squares_centered = interop::copy_to_daal_homogen_table<Float>(
                input_.get_partial_sum_squares_centered());

            daal_partial.set(daal_lom::PartialResultId::partialSum, daal_partial_sums);
            daal_partial.set(daal_lom::PartialResultId::partialSumSquaresCentered,
                             daal_partial_sum_squares_centered);

            daal_partial.set(daal_lom::PartialResultId::partialSumSquares,
                             daal_partial_sum_squares);
        }
        interop::status_to_exception(dal::backend::dispatch_by_cpu(ctx, [&](auto cpu) {
            return daal_lom_online_kernel_t<
                       Float,
                       oneapi::dal::backend::interop::to_daal_cpu_type<decltype(cpu)>::value,
                       Method>()
                .compute(daal_data.get(), &daal_partial, &daal_parameter, is_online);
        }));
        auto result = get_partial_result<Float, task_t>(daal_partial, desc);
        return result;
    }
    else {
        {
            interop::status_to_exception(dal::backend::dispatch_by_cpu(ctx, [&](auto cpu) {
                return daal_lom_online_kernel_t<
                           Float,
                           oneapi::dal::backend::interop::to_daal_cpu_type<decltype(cpu)>::value,
                           Method>()
                    .compute(daal_data.get(), &daal_partial, &daal_parameter, is_online);
            }));
        }
        auto result = get_partial_result<Float, task_t>(daal_partial, desc);
        return result;
    }
}

template <typename Float, typename Method, typename Task>
static partial_compute_result<Task> partial_compute(const context_cpu& ctx,
                                                    const descriptor_t& desc,
                                                    const partial_compute_input<Task>& input) {
    if (input.get_weights().has_data()) {
        return call_daal_kernel_with_weights<Float, Task>(ctx, desc, input);
    }
    else {
        return call_daal_kernel_without_weights<Float, Method, Task>(ctx, desc, input);
    }
}

template <typename Float, typename Method>
struct partial_compute_kernel_cpu<Float, Method, task_t> {
    partial_compute_result<task::compute> operator()(
        const context_cpu& ctx,
        const descriptor_t& desc,
        const partial_compute_input<task::compute>& input) const {
        return partial_compute<Float, Method, task::compute>(ctx, desc, input);
    }
};

template struct partial_compute_kernel_cpu<float, method::dense, task_t>;
template struct partial_compute_kernel_cpu<double, method::dense, task_t>;
template struct partial_compute_kernel_cpu<float, method::sparse, task_t>;
template struct partial_compute_kernel_cpu<double, method::sparse, task_t>;

} // namespace oneapi::dal::basic_statistics::backend
