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
#include "oneapi/dal/algo/hdbscan/backend/cpu/compute_kernel_common.hpp"
#include "oneapi/dal/algo/hdbscan/backend/cpu/cluster_utils.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/table/homogen.hpp"

#include <daal/src/algorithms/hdbscan/hdbscan_kernel.h>

namespace oneapi::dal::hdbscan::backend {

using dal::backend::context_cpu;

using descriptor_t = detail::descriptor_base<task::clustering>;
using result_t = compute_result<task::clustering>;
using input_t = compute_input<task::clustering>;

namespace daal_hdbscan = daal::algorithms::hdbscan::internal;
namespace interop = dal::backend::interop;

template <daal_hdbscan::Method Value>
using daal_method_constant = std::integral_constant<daal_hdbscan::Method, Value>;

template <typename Method>
struct to_daal_method;

template <>
struct to_daal_method<method::brute_force>
        : daal_method_constant<daal_hdbscan::Method::bruteForceDense> {};

template <>
struct to_daal_method<method::kd_tree> : daal_method_constant<daal_hdbscan::Method::kdTree> {};

template <>
struct to_daal_method<method::ball_tree> : daal_method_constant<daal_hdbscan::Method::ballTree> {};

template <typename Method>
struct batch_kernel {
    template <typename Float, daal::internal::CpuType Cpu>
    using type = daal_hdbscan::HDBSCANBatchKernel<Float, to_daal_method<Method>::value, Cpu>;
};

/// Run the HDBSCAN CPU pipeline of `Method` through the DAAL `HDBSCANBatchKernel`.
///
/// @tparam Float  Floating-point type
/// @tparam Method oneAPI method tag
///
/// @param[in] ctx  CPU dispatch context
/// @param[in] desc Algorithm descriptor
/// @param[in] data Input data table of size `n x d`
template <typename Float, typename Method>
static result_t compute_kernel_impl(const context_cpu& ctx,
                                    const descriptor_t& desc,
                                    const table& data) {
    const std::int64_t row_count = data.get_row_count();
    const auto& options = desc.get_result_options();

    const auto daal_data = interop::convert_to_daal_table<Float>(data);

    auto arr_responses = array<std::int32_t>::empty(row_count);
    auto arr_cluster_count = array<std::int32_t>::empty(1);
    const auto daal_responses = interop::convert_to_daal_homogen_table(arr_responses, row_count, 1);
    const auto daal_cluster_count = interop::convert_to_daal_homogen_table(arr_cluster_count, 1, 1);

    // The centers are weighted by membership probability. An empty array becomes a null table,
    // which the kernel skips.
    const bool need_centers = centers_requested(desc);
    const bool need_probabilities = options.test(result_options::probabilities);
    array<Float> arr_probabilities;
    if (need_probabilities || need_centers) {
        arr_probabilities = array<Float>::empty(row_count);
    }
    const auto daal_probabilities =
        interop::convert_to_daal_homogen_table(arr_probabilities, row_count, 1);

    // There is no merge below two rows.
    const std::int64_t edge_count = row_count - 1;
    const bool need_single_linkage_tree =
        options.test(result_options::single_linkage_tree) && edge_count > 0;
    array<Float> arr_single_linkage_tree;
    if (need_single_linkage_tree) {
        arr_single_linkage_tree = array<Float>::empty(edge_count * 4);
    }
    const auto daal_single_linkage_tree =
        interop::convert_to_daal_homogen_table(arr_single_linkage_tree, edge_count, 4);

    interop::status_to_exception(
        interop::call_daal_kernel<Float, batch_kernel<Method>::template type>(
            ctx,
            daal_data.get(),
            daal_responses.get(),
            daal_cluster_count.get(),
            daal_probabilities.get(),
            daal_single_linkage_tree.get(),
            static_cast<size_t>(desc.get_min_cluster_size()),
            static_cast<size_t>(desc.get_min_samples()),
            convert_metric(desc.get_metric()),
            desc.get_degree(),
            desc.get_cluster_selection() == cluster_selection_method::leaf ? 1 : 0,
            desc.get_allow_single_cluster(),
            desc.get_cluster_selection_epsilon(),
            static_cast<size_t>(desc.get_max_cluster_size()),
            desc.get_alpha(),
            static_cast<size_t>(desc.get_leaf_size())));

    const std::int64_t cluster_count = arr_cluster_count.get_data()[0];
    auto results = result_t().set_cluster_count(cluster_count).set_result_options(options);

    if (options.test(result_options::responses)) {
        results.set_responses(homogen_table::wrap(arr_responses, row_count, 1));
        if (need_centers) {
            set_cluster_centers(ctx,
                                desc,
                                data,
                                arr_responses,
                                arr_probabilities,
                                cluster_count,
                                results);
        }
    }
    if (need_probabilities) {
        results.set_probabilities(homogen_table::wrap(arr_probabilities, row_count, 1));
    }
    if (need_single_linkage_tree) {
        results.set_single_linkage_tree(
            homogen_table::wrap(arr_single_linkage_tree, edge_count, 4));
    }
    return results;
}

template <typename Float, typename Method>
struct compute_kernel_cpu<Float, Method, task::clustering> {
    result_t operator()(const context_cpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute_kernel_impl<Float, Method>(ctx, desc, input.get_data());
    }
};

template struct compute_kernel_cpu<float, method::brute_force, task::clustering>;
template struct compute_kernel_cpu<double, method::brute_force, task::clustering>;
template struct compute_kernel_cpu<float, method::kd_tree, task::clustering>;
template struct compute_kernel_cpu<double, method::kd_tree, task::clustering>;
template struct compute_kernel_cpu<float, method::ball_tree, task::clustering>;
template struct compute_kernel_cpu<double, method::ball_tree, task::clustering>;

} // namespace oneapi::dal::hdbscan::backend
