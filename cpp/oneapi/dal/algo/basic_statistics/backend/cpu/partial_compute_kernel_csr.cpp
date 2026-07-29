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

#include "oneapi/dal/algo/basic_statistics/backend/cpu/partial_compute_kernel.hpp"
#include "oneapi/dal/algo/basic_statistics/backend/basic_statistics_interop.hpp"

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/table/csr.hpp"

#include <daal/src/algorithms/low_order_moments/moments_online.h>
#include <daal/src/algorithms/low_order_moments/low_order_moments_kernel.h>

namespace oneapi::dal::basic_statistics::backend {

using dal::backend::context_cpu;
using method_t = method::sparse;
using task_t = task::compute;
using input_t = partial_compute_input<task_t>;
using result_t = partial_compute_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

namespace daal_lom = daal::algorithms::low_order_moments;
namespace interop = dal::backend::interop;

template <typename Float, daal::internal::CpuType Cpu>
using daal_lom_csr_online_kernel_t =
    daal_lom::internal::LowOrderMomentsOnlineKernel<Float, daal_lom::fastCSR, Cpu>;

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
static result_t call_daal_kernel(const context_cpu& ctx,
                                 const descriptor_t& desc,
                                 const partial_compute_input<Task>& input) {
    const auto data = input.get_data();
    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(data.get_kind() == csr_table::kind());

    // DAAL fastCSR online kernel does not take a per-row weights vector.
    if (input.get_weights().has_data()) {
        throw unimplemented(dal::detail::error_messages::method_not_implemented());
    }

    constexpr bool is_online = true;

    auto daal_input = daal_lom::Input();
    auto daal_partial = daal_lom::PartialResult();

    const auto res_op = desc.get_result_options();
    const auto result_ids = get_daal_estimates_to_compute(desc);
    const auto daal_parameter = daal_lom::Parameter(result_ids);

    const auto& prev = input.get_prev();

    const auto daal_data = interop::convert_to_daal_table<Float>(data);
    daal_input.set(daal_lom::InputId::data, daal_data);

    alloc_result<Float>(daal_partial, &daal_input, &daal_parameter, daal_lom::fastCSR);
    initialize_result<Float>(daal_partial, &daal_input, &daal_parameter, daal_lom::fastCSR);

    const bool has_nobs_data = prev.get_partial_n_rows().has_data();
    if (has_nobs_data) {
        auto daal_nobs = interop::copy_to_daal_homogen_table<Float>(prev.get_partial_n_rows());
        daal_partial.set(daal_lom::PartialResultId::nObservations, daal_nobs);
        if (result_ids == daal_lom::estimatesMinMax || res_op.test(result_options::min) ||
            res_op.test(result_options::max)) {
            auto daal_prev_max = interop::copy_to_daal_homogen_table<Float>(prev.get_partial_max());
            auto daal_prev_min = interop::copy_to_daal_homogen_table<Float>(prev.get_partial_min());
            daal_partial.set(daal_lom::PartialResultId::partialMaximum, daal_prev_max);
            daal_partial.set(daal_lom::PartialResultId::partialMinimum, daal_prev_min);
        }
        if (result_ids == daal_lom::estimatesMeanVariance || result_ids == daal_lom::estimatesAll) {
            auto daal_prev_sums =
                interop::copy_to_daal_homogen_table<Float>(prev.get_partial_sum());
            auto daal_prev_sum_squares =
                interop::copy_to_daal_homogen_table<Float>(prev.get_partial_sum_squares());
            auto daal_prev_sum_squares_centered =
                interop::copy_to_daal_homogen_table<Float>(prev.get_partial_sum_squares_centered());

            daal_partial.set(daal_lom::PartialResultId::partialSum, daal_prev_sums);
            daal_partial.set(daal_lom::PartialResultId::partialSumSquares, daal_prev_sum_squares);
            daal_partial.set(daal_lom::PartialResultId::partialSumSquaresCentered,
                             daal_prev_sum_squares_centered);
        }
    }

    interop::status_to_exception(
        interop::call_daal_kernel<Float, daal_lom_csr_online_kernel_t>(ctx,
                                                                       daal_data.get(),
                                                                       &daal_partial,
                                                                       &daal_parameter,
                                                                       is_online));
    return get_partial_result<Float, task_t>(daal_partial, desc);
}

template <typename Float, typename Task>
static partial_compute_result<Task> partial_compute(const context_cpu& ctx,
                                                    const descriptor_t& desc,
                                                    const partial_compute_input<Task>& input) {
    return call_daal_kernel<Float, Task>(ctx, desc, input);
}

template <typename Float>
struct partial_compute_kernel_cpu<Float, method_t, task_t> {
    partial_compute_result<task_t> operator()(const context_cpu& ctx,
                                              const descriptor_t& desc,
                                              const partial_compute_input<task_t>& input) const {
        return partial_compute<Float, task_t>(ctx, desc, input);
    }
};

template struct partial_compute_kernel_cpu<float, method_t, task_t>;
template struct partial_compute_kernel_cpu<double, method_t, task_t>;

} // namespace oneapi::dal::basic_statistics::backend
