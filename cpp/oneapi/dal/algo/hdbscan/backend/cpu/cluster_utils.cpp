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

#include "oneapi/dal/algo/hdbscan/backend/cpu/cluster_utils.hpp"
#include "oneapi/dal/algo/hdbscan/backend/cpu/compute_kernel_common.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/table/homogen.hpp"

#include <daal/src/algorithms/hdbscan/hdbscan_kernel.h>

namespace oneapi::dal::hdbscan::backend {

namespace interop = dal::backend::interop;

template <typename Float, daal::internal::CpuType Cpu>
using daal_hdbscan_centers_t =
    daal::algorithms::hdbscan::internal::HDBSCANCentersKernel<Float, Cpu>;

template <typename Float, daal::internal::CpuType Cpu>
using daal_hdbscan_extract_t =
    daal::algorithms::hdbscan::internal::HDBSCANExtractClustersKernel<Float, Cpu>;

/// Wrap a host column as a read-only DAAL table, so an immutable array is accepted.
///
/// @param[in] arr       Column data, length `row_count`
/// @param[in] row_count Number of rows; 0 gives a null table
template <typename T>
static daal::data_management::NumericTablePtr wrap_column(const array<T>& arr,
                                                          std::int64_t row_count) {
    if (row_count == 0) {
        return daal::data_management::NumericTablePtr();
    }
    return interop::convert_to_daal_table<T>(homogen_table::wrap(arr, row_count, 1));
}

template <typename Float>
extracted_clusters<Float> extract_clusters_from_mst(
    const dal::backend::context_cpu& ctx,
    const detail::descriptor_base<task::clustering>& desc,
    const array<std::int32_t>& mst_from,
    const array<std::int32_t>& mst_to,
    const array<Float>& mst_weights,
    std::int64_t row_count,
    bool need_probabilities,
    bool need_single_linkage) {
    ONEDAL_ASSERT(row_count > 0);
    const std::int64_t edge_count = row_count - 1;

    extracted_clusters<Float> out;
    out.responses = array<std::int32_t>::empty(row_count);
    if (need_probabilities) {
        out.probabilities = array<Float>::empty(row_count);
    }
    if (need_single_linkage && edge_count > 0) {
        out.single_linkage_tree = array<Float>::empty(4 * edge_count);
    }
    auto cluster_count = array<std::int32_t>::empty(1);

    const auto daal_from = wrap_column(mst_from, edge_count);
    const auto daal_to = wrap_column(mst_to, edge_count);
    const auto daal_weights = wrap_column(mst_weights, edge_count);
    const auto daal_responses = interop::convert_to_daal_homogen_table(out.responses, row_count, 1);
    const auto daal_cluster_count = interop::convert_to_daal_homogen_table(cluster_count, 1, 1);
    const auto daal_probabilities =
        interop::convert_to_daal_homogen_table(out.probabilities, row_count, 1);
    const auto daal_single_linkage_tree =
        interop::convert_to_daal_homogen_table(out.single_linkage_tree, edge_count, 4);

    const bool allow_single_cluster = desc.get_allow_single_cluster();
    const int cluster_selection =
        (desc.get_cluster_selection() == cluster_selection_method::leaf) ? 1 : 0;
    kernel_status_to_exception(interop::call_daal_kernel<Float, daal_hdbscan_extract_t>(
        ctx,
        daal_from.get(),
        daal_to.get(),
        daal_weights.get(),
        static_cast<size_t>(row_count),
        daal_responses.get(),
        daal_cluster_count.get(),
        daal_probabilities.get(),
        daal_single_linkage_tree.get(),
        static_cast<size_t>(desc.get_min_cluster_size()),
        cluster_selection,
        allow_single_cluster,
        desc.get_cluster_selection_epsilon(),
        static_cast<size_t>(desc.get_max_cluster_size())));

    out.cluster_count = cluster_count.get_data()[0];
    return out;
}

template <typename Float>
void set_cluster_centers(const dal::backend::context_cpu& ctx,
                         const detail::descriptor_base<task::clustering>& desc,
                         const table& data,
                         const array<std::int32_t>& labels,
                         const array<Float>& weights,
                         std::int64_t cluster_count,
                         compute_result<task::clustering>& result) {
    const auto store_centers = desc.get_store_centers();
    if (cluster_count <= 0) {
        return;
    }
    const std::int64_t row_count = data.get_row_count();
    const std::int64_t col_count = data.get_column_count();
    const bool need_centroids = store_centers == store_centers_method::centroid ||
                                store_centers == store_centers_method::both;
    const bool need_medoids = store_centers == store_centers_method::medoid ||
                              store_centers == store_centers_method::both;

    // An empty array converts to a null table, which the kernel skips.
    array<Float> centroids;
    array<Float> medoids;
    if (need_centroids) {
        centroids = array<Float>::empty(cluster_count * col_count);
    }
    if (need_medoids) {
        medoids = array<Float>::empty(cluster_count * col_count);
    }

    const auto daal_data = interop::convert_to_daal_table<Float>(data);
    const auto daal_labels = wrap_column(labels, row_count);
    const auto daal_weights = wrap_column(weights, row_count);
    const auto daal_centroids =
        interop::convert_to_daal_homogen_table(centroids, cluster_count, col_count);
    const auto daal_medoids =
        interop::convert_to_daal_homogen_table(medoids, cluster_count, col_count);

    interop::status_to_exception(
        interop::call_daal_kernel<Float, daal_hdbscan_centers_t>(ctx,
                                                                 daal_data.get(),
                                                                 daal_labels.get(),
                                                                 daal_weights.get(),
                                                                 static_cast<size_t>(cluster_count),
                                                                 daal_centroids.get(),
                                                                 daal_medoids.get(),
                                                                 convert_metric(desc.get_metric()),
                                                                 desc.get_degree()));

    if (need_centroids) {
        result.set_cluster_centers(homogen_table::wrap(centroids, cluster_count, col_count));
    }
    if (need_medoids) {
        result.set_medoid_centers(homogen_table::wrap(medoids, cluster_count, col_count));
    }
}

#define INSTANTIATE(F)                                                                  \
    template void set_cluster_centers(const dal::backend::context_cpu&,                 \
                                      const detail::descriptor_base<task::clustering>&, \
                                      const table&,                                     \
                                      const array<std::int32_t>&,                       \
                                      const array<F>&,                                  \
                                      std::int64_t,                                     \
                                      compute_result<task::clustering>&);               \
    template extracted_clusters<F> extract_clusters_from_mst(                           \
        const dal::backend::context_cpu&,                                               \
        const detail::descriptor_base<task::clustering>&,                               \
        const array<std::int32_t>&,                                                     \
        const array<std::int32_t>&,                                                     \
        const array<F>&,                                                                \
        std::int64_t,                                                                   \
        bool,                                                                           \
        bool);

INSTANTIATE(float)
INSTANTIATE(double)

} // namespace oneapi::dal::hdbscan::backend
