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
    auto labels_copy = labels;
    auto weights_copy = weights;
    const auto daal_labels = interop::convert_to_daal_homogen_table(labels_copy, row_count, 1);
    const auto daal_weights = interop::convert_to_daal_homogen_table(weights_copy, row_count, 1);
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
                                      compute_result<task::clustering>&);

INSTANTIATE(float)
INSTANTIATE(double)

} // namespace oneapi::dal::hdbscan::backend
