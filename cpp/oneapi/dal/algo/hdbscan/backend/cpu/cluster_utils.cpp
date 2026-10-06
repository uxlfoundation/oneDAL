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
#include "oneapi/dal/table/homogen.hpp"

namespace oneapi::dal::hdbscan::backend {

template <typename Float>
void set_cluster_centers(const dal::backend::context_cpu& ctx,
                         const detail::descriptor_base<task::clustering>& desc,
                         const Float* data,
                         const std::int32_t* labels,
                         const Float* weights,
                         std::int64_t row_count,
                         std::int64_t col_count,
                         std::int64_t cluster_count,
                         compute_result<task::clustering>& result) {
    const auto store_centers = desc.get_store_centers();
    if (cluster_count <= 0 || store_centers == store_centers_method::none) {
        return;
    }
    if (store_centers == store_centers_method::centroid ||
        store_centers == store_centers_method::both) {
        auto centroids = array<Float>::empty(cluster_count * col_count);
        dal::backend::dispatch_by_cpu(ctx, [&](auto cpu) {
            compute_centroids<decltype(cpu), Float>(data,
                                                    labels,
                                                    weights,
                                                    row_count,
                                                    col_count,
                                                    cluster_count,
                                                    centroids.get_mutable_data());
        });
        result.set_cluster_centers(homogen_table::wrap(centroids, cluster_count, col_count));
    }
    if (store_centers == store_centers_method::medoid ||
        store_centers == store_centers_method::both) {
        auto medoids = array<Float>::empty(cluster_count * col_count);
        dal::backend::dispatch_by_cpu(ctx, [&](auto cpu) {
            compute_medoids<decltype(cpu), Float>(data,
                                                  labels,
                                                  weights,
                                                  row_count,
                                                  col_count,
                                                  cluster_count,
                                                  desc.get_metric(),
                                                  desc.get_degree(),
                                                  medoids.get_mutable_data());
        });
        result.set_medoid_centers(homogen_table::wrap(medoids, cluster_count, col_count));
    }
}

#define INSTANTIATE(F)                                                                  \
    template void set_cluster_centers(const dal::backend::context_cpu&,                 \
                                      const detail::descriptor_base<task::clustering>&, \
                                      const F*,                                         \
                                      const std::int32_t*,                              \
                                      const F*,                                         \
                                      std::int64_t,                                     \
                                      std::int64_t,                                     \
                                      std::int64_t,                                     \
                                      compute_result<task::clustering>&);

INSTANTIATE(float)
INSTANTIATE(double)

} // namespace oneapi::dal::hdbscan::backend
