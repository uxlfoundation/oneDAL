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

#include <daal/src/algorithms/cordistance/cordistance_kernel.h>

#include "oneapi/dal/algo/correlation_distance/backend/cpu/compute_kernel.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/exceptions.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::correlation_distance::backend {

using dal::backend::context_cpu;
using input_t = compute_input<task::compute>;
using result_t = compute_result<task::compute>;
using descriptor_t = detail::descriptor_base<task::compute>;

namespace daal_correlation = daal::algorithms::correlation_distance;
namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using daal_correlation_t =
    daal_correlation::internal::DistanceKernel<Float, daal_correlation::defaultDense, Cpu>;

template <typename Float>
static result_t call_daal_kernel(const context_cpu& ctx,
                                 const descriptor_t& desc,
                                 const table& x,
                                 const table& y) {
    const std::int64_t row_count_x = x.get_row_count();
    const std::int64_t row_count_y = y.get_row_count();

    dal::detail::check_mul_overflow(row_count_x, row_count_y);
    auto arr_values = array<Float>::empty(row_count_x * row_count_y);

    const auto daal_x = interop::convert_to_daal_table<Float>(x);
    const auto daal_y = interop::convert_to_daal_table<Float>(y);
    const auto daal_values =
        interop::convert_to_daal_homogen_table(arr_values, row_count_x, row_count_y);

    daal::algorithms::Parameter param;
    const daal::data_management::NumericTable* daal_input_tables[2] = { daal_x.get(),
                                                                        daal_y.get() };
    daal::data_management::NumericTable* daal_result_table[1] = { daal_values.get() };

    interop::status_to_exception(
        interop::call_daal_kernel<Float, daal_correlation_t>(ctx,
                                                             2,
                                                             daal_input_tables,
                                                             1,
                                                             daal_result_table,
                                                             &param));

    return result_t().set_values(
        dal::detail::homogen_table_builder{}.reset(arr_values, row_count_x, row_count_y).build());
}

template <typename Float>
static result_t compute(const context_cpu& ctx, const descriptor_t& desc, const input_t& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_x(), input.get_y());
}

template <typename Float>
struct compute_kernel_cpu<Float, method::dense, task::compute> {
    result_t operator()(const context_cpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute<Float>(ctx, desc, input);
    }

#ifdef ONEDAL_DATA_PARALLEL
    void operator()(const context_cpu& ctx,
                    const descriptor_t& desc,
                    const table& x,
                    const table& y,
                    homogen_table& res) const {
        throw unimplemented(dal::detail::error_messages::method_not_implemented());
    }
#endif
};

template struct compute_kernel_cpu<float, method::dense, task::compute>;
template struct compute_kernel_cpu<double, method::dense, task::compute>;

} // namespace oneapi::dal::correlation_distance::backend
