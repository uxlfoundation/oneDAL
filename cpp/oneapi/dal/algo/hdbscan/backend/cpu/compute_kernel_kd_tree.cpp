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

#include "oneapi/dal/algo/hdbscan/backend/cpu/compute_kernel.hpp"
#include "oneapi/dal/algo/hdbscan/common.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include <daal/src/algorithms/hdbscan/hdbscan_kernel.h>

namespace oneapi::dal::hdbscan::backend {

using dal::backend::context_cpu;
using dal::backend::interop::convert_to_daal_table;

using descriptor_t = detail::descriptor_base<task::clustering>;
using result_t = compute_result<task::clustering>;
using input_t = compute_input<task::clustering>;

namespace daal_hdbscan = daal::algorithms::hdbscan;
namespace daal_hdbscan_internal = daal::algorithms::hdbscan::internal;
namespace interop = dal::backend::interop;

template <typename Float, daal::internal::CpuType Cpu>
using daal_hdbscan_kd_tree_t =
    daal_hdbscan_internal::HDBSCANBatchKernel<Float, daal_hdbscan::kdTree, Cpu>;

template <typename Float>
static result_t compute_kernel_kd_tree_impl(const context_cpu& ctx,
                                            const descriptor_t& desc,
                                            const table& data) {
    const std::int64_t row_count = data.get_row_count();
    const std::int64_t min_cluster_size = desc.get_min_cluster_size();
    const std::int64_t min_samples = desc.get_min_samples();

    const auto daal_data = convert_to_daal_table<Float>(data);

    auto daal_assignments = daal::data_management::HomogenNumericTable<int>::create(
        1,
        row_count,
        daal::data_management::NumericTable::doAllocate);
    auto daal_nclusters = daal::data_management::HomogenNumericTable<int>::create(
        1,
        1,
        daal::data_management::NumericTable::doAllocate);

    interop::status_to_exception(interop::call_daal_kernel<Float, daal_hdbscan_kd_tree_t>(
        ctx,
        daal_data.get(),
        daal_assignments.get(),
        daal_nclusters.get(),
        static_cast<size_t>(min_cluster_size),
        static_cast<size_t>(min_samples)));

    daal::data_management::BlockDescriptor<int> nc_block;
    daal_nclusters->getBlockOfRows(0, 1, daal::data_management::readOnly, nc_block);
    const std::int64_t cluster_count = nc_block.getBlockPtr()[0];
    daal_nclusters->releaseBlockOfRows(nc_block);

    auto results =
        result_t().set_cluster_count(cluster_count).set_result_options(desc.get_result_options());

    if (desc.get_result_options().test(result_options::responses)) {
        auto arr_responses = array<std::int32_t>::empty(row_count);
        std::int32_t* resp_ptr = arr_responses.get_mutable_data();

        daal::data_management::BlockDescriptor<int> assign_block;
        daal_assignments->getBlockOfRows(0,
                                         row_count,
                                         daal::data_management::readOnly,
                                         assign_block);
        const int* assign_ptr = assign_block.getBlockPtr();
        for (std::int64_t i = 0; i < row_count; i++) {
            resp_ptr[i] = static_cast<std::int32_t>(assign_ptr[i]);
        }
        daal_assignments->releaseBlockOfRows(assign_block);

        results.set_responses(dal::homogen_table::wrap(arr_responses, row_count, 1));
    }

    return results;
}

template <typename Float>
struct compute_kernel_cpu<Float, method::kd_tree, task::clustering> {
    result_t operator()(const context_cpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute_kernel_kd_tree_impl<Float>(ctx, desc, input.get_data());
    }
};

template struct compute_kernel_cpu<float, method::kd_tree, task::clustering>;
template struct compute_kernel_cpu<double, method::kd_tree, task::clustering>;

} // namespace oneapi::dal::hdbscan::backend
