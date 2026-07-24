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

#include "oneapi/dal/algo/basic_statistics/backend/cpu/finalize_compute_kernel.hpp"
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
using input_t = compute_input<task_t>;
using result_t = compute_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

namespace daal_lom = daal::algorithms::low_order_moments;
namespace interop = dal::backend::interop;
namespace bk = dal::backend;

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

template <typename Float, typename Method, typename Task>
static compute_result<Task> call_daal_kernel_finalize_compute(
    const context_cpu& ctx,
    const descriptor_t& desc,
    const partial_compute_result<Task>& input) {
    const auto result_ids = get_daal_estimates_to_compute(desc);
    const auto daal_parameter = daal_lom::Parameter(result_ids);

    std::int64_t column_count = 0;

    const auto res_op = desc.get_result_options();

    if (result_ids == daal_lom::estimatesMeanVariance || result_ids == daal_lom::estimatesAll) {
        column_count = input.get_partial_sum().get_column_count();
    }
    if (result_ids == daal_lom::estimatesMinMax) {
        column_count = input.get_partial_min().get_column_count();
    }
    ONEDAL_ASSERT(column_count > 0);
    auto daal_partial = daal_lom::PartialResult();
    auto daal_partial_obs = interop::copy_to_daal_homogen_table<Float>(input.get_partial_n_rows());
    auto daal_partial_min = interop::copy_to_daal_homogen_table<Float>(input.get_partial_min());
    auto daal_partial_max = interop::copy_to_daal_homogen_table<Float>(input.get_partial_max());
    auto daal_partial_sums = interop::copy_to_daal_homogen_table<Float>(input.get_partial_sum());
    auto daal_partial_sum_squares =
        interop::copy_to_daal_homogen_table<Float>(input.get_partial_sum_squares());
    auto daal_partial_sum_squares_centered =
        interop::copy_to_daal_homogen_table<Float>(input.get_partial_sum_squares_centered());

    auto daal_means = interop::allocate_daal_homogen_table<Float>(1, column_count);
    auto daal_rawt = interop::allocate_daal_homogen_table<Float>(1, column_count);
    auto daal_variance = interop::allocate_daal_homogen_table<Float>(1, column_count);
    auto daal_stdev = interop::allocate_daal_homogen_table<Float>(1, column_count);
    auto daal_variation = interop::allocate_daal_homogen_table<Float>(1, column_count);
    if (result_ids == daal_lom::estimatesMeanVariance || result_ids == daal_lom::estimatesAll) {
        interop::status_to_exception(dal::backend::dispatch_by_cpu(ctx, [&](auto cpu) {
            return daal_lom_online_kernel_t<
                       Float,
                       oneapi::dal::backend::interop::to_daal_cpu_type<decltype(cpu)>::value,
                       Method>()
                .finalizeCompute(daal_partial_obs.get(),
                                 daal_partial_sums.get(),
                                 daal_partial_sum_squares.get(),
                                 daal_partial_sum_squares_centered.get(),
                                 daal_means.get(),
                                 daal_rawt.get(),
                                 daal_variance.get(),
                                 daal_stdev.get(),
                                 daal_variation.get(),
                                 &daal_parameter);
        }));
    }
    compute_result<Task> res;
    res.set_result_options(desc.get_result_options());
    if (res_op.test(result_options::min)) {
        res.set_min(interop::convert_from_daal_homogen_table<Float>(daal_partial_min));
    }
    if (res_op.test(result_options::max)) {
        res.set_max(interop::convert_from_daal_homogen_table<Float>(daal_partial_max));
    }
    if (res_op.test(result_options::sum)) {
        res.set_sum(interop::convert_from_daal_homogen_table<Float>(daal_partial_sums));
    }
    if (res_op.test(result_options::sum_squares)) {
        res.set_sum_squares(
            interop::convert_from_daal_homogen_table<Float>(daal_partial_sum_squares));
    }
    if (res_op.test(result_options::sum_squares_centered)) {
        res.set_sum_squares_centered(
            interop::convert_from_daal_homogen_table<Float>(daal_partial_sum_squares_centered));
    }
    if (res_op.test(result_options::mean)) {
        res.set_mean(interop::convert_from_daal_homogen_table<Float>(daal_means));
    }
    if (res_op.test(result_options::second_order_raw_moment)) {
        res.set_second_order_raw_moment(interop::convert_from_daal_homogen_table<Float>(daal_rawt));
    }
    if (res_op.test(result_options::variance)) {
        res.set_variance(interop::convert_from_daal_homogen_table<Float>(daal_variance));
    }
    if (res_op.test(result_options::standard_deviation)) {
        res.set_standard_deviation(interop::convert_from_daal_homogen_table<Float>(daal_stdev));
    }
    if (res_op.test(result_options::variation)) {
        res.set_variation(interop::convert_from_daal_homogen_table<Float>(daal_variation));
    }

    return res;
}

template <typename Float, typename Method, typename Task>
static compute_result<Task> finalize_compute(const context_cpu& ctx,
                                             const descriptor_t& desc,
                                             const partial_compute_result<Task>& input) {
    return call_daal_kernel_finalize_compute<Float, Method, Task>(ctx, desc, input);
}

template <typename Float, typename Method>
struct finalize_compute_kernel_cpu<Float, Method, task_t> {
    compute_result<task::compute> operator()(
        const context_cpu& ctx,
        const descriptor_t& desc,
        const partial_compute_result<task::compute>& input) const {
        return finalize_compute<Float, Method, task::compute>(ctx, desc, input);
    }
};

template struct finalize_compute_kernel_cpu<float, method::dense, task_t>;
template struct finalize_compute_kernel_cpu<double, method::dense, task_t>;
template struct finalize_compute_kernel_cpu<float, method::sparse, task_t>;
template struct finalize_compute_kernel_cpu<double, method::sparse, task_t>;

} // namespace oneapi::dal::basic_statistics::backend
