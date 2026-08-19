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

namespace oneapi::dal::hdbscan::backend {

template <typename Float>
void compute_centroids(const dal::backend::context_cpu& ctx,
                       const Float* data,
                       const std::int32_t* labels,
                       std::int64_t row_count,
                       std::int64_t col_count,
                       std::int64_t cluster_count,
                       Float* centroids) {
    return dal::backend::dispatch_by_cpu(ctx, [&](auto cpu) {
        return compute_centroids<decltype(cpu), Float>(data,
                                                       labels,
                                                       row_count,
                                                       col_count,
                                                       cluster_count,
                                                       centroids);
    });
}

template <typename Float>
void compute_medoids(const dal::backend::context_cpu& ctx,
                     const Float* data,
                     const std::int32_t* labels,
                     std::int64_t row_count,
                     std::int64_t col_count,
                     std::int64_t cluster_count,
                     const Float* centroids,
                     Float* medoids) {
    return dal::backend::dispatch_by_cpu(ctx, [&](auto cpu) {
        return compute_medoids<decltype(cpu), Float>(data,
                                                     labels,
                                                     row_count,
                                                     col_count,
                                                     cluster_count,
                                                     centroids,
                                                     medoids);
    });
}

#define INSTANTIATE(F)                                                \
    template void compute_centroids(const dal::backend::context_cpu&, \
                                    const F*,                         \
                                    const std::int32_t*,              \
                                    std::int64_t,                     \
                                    std::int64_t,                     \
                                    std::int64_t,                     \
                                    F*);                              \
    template void compute_medoids(const dal::backend::context_cpu&,   \
                                  const F*,                           \
                                  const std::int32_t*,                \
                                  std::int64_t,                       \
                                  std::int64_t,                       \
                                  std::int64_t,                       \
                                  const F*,                           \
                                  F*);

INSTANTIATE(float)
INSTANTIATE(double)

} // namespace oneapi::dal::hdbscan::backend
