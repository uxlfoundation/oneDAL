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

#pragma once

#include <cstdint>

#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::hdbscan::backend {

/// Compute the per-cluster centroid (mean point) of a labeled point set on the host.
///
/// This is the CPU / host code path. The GPU backend uses
/// `compute_centroids_gpu` (declared in `backend/gpu/cluster_utils.hpp`) which
/// does the same math directly on the SYCL device via `atomic_ref`, so `data` /
/// `labels` never round-trip through host memory in that path.
///
/// Sums every labeled point into its cluster's row of `centroids`, counts the
/// points per cluster, then divides by the count. Points whose label is
/// negative or out of range are skipped (HDBSCAN noise). Accumulate and
/// normalize inner loops carry `PRAGMA_OMP_SIMD` so the compiler autovectorizes
/// the mul/add.
///
/// Defined in `cluster_utils_cpu.cpp`, which is compiled once per supported
/// instruction set, so `Cpu` selects the vectorization width. Callers on the
/// host path should use the `context_cpu` overload below, which picks the
/// instantiation matching the running machine.
///
/// TODO: explore `mkl::blas::scal` for the normalize loop.
///
/// @tparam Cpu   CPU dispatch tag (`backend::cpu_dispatch_*`)
/// @tparam Float Floating-point type
///
/// @param[in]  data          Row-major input buffer of size `row_count x col_count`
/// @param[in]  labels        Cluster id per point, length `row_count` (-1 = noise)
/// @param[in]  row_count     Number of input points
/// @param[in]  col_count     Number of features per point
/// @param[in]  cluster_count Number of clusters
/// @param[out] centroids     Row-major centroid buffer, size `cluster_count x col_count`
template <typename Cpu, typename Float>
void compute_centroids(const Float* data,
                       const std::int32_t* labels,
                       std::int64_t row_count,
                       std::int64_t col_count,
                       std::int64_t cluster_count,
                       Float* centroids);

/// Compute the per-cluster medoid (closest input point to the centroid) on the host.
///
/// This is the CPU / host code path. The GPU counterpart is
/// `compute_medoids_gpu` (declared in `backend/gpu/cluster_utils.hpp`) which
/// keeps `data` / `labels` on the SYCL device.
///
/// For each labeled point, computes its squared Euclidean distance to the
/// cluster centroid and tracks the minimum per cluster. The chosen medoid is
/// then copied row-by-row into `medoids`. Empty clusters get a zero row.
///
/// Defined in `cluster_utils_cpu.cpp` and dispatched per instruction set, same
/// as `compute_centroids` above.
///
/// @tparam Cpu   CPU dispatch tag (`backend::cpu_dispatch_*`)
/// @tparam Float Floating-point type
///
/// @param[in]  data          Row-major input buffer of size `row_count x col_count`
/// @param[in]  labels        Cluster id per point, length `row_count` (-1 = noise)
/// @param[in]  row_count     Number of input points
/// @param[in]  col_count     Number of features per point
/// @param[in]  cluster_count Number of clusters
/// @param[in]  centroids     Cluster centroids, size `cluster_count x col_count`
/// @param[out] medoids       Output medoid rows, size `cluster_count x col_count`
template <typename Cpu, typename Float>
void compute_medoids(const Float* data,
                     const std::int32_t* labels,
                     std::int64_t row_count,
                     std::int64_t col_count,
                     std::int64_t cluster_count,
                     const Float* centroids,
                     Float* medoids);

/// Dispatch `compute_centroids` to the instantiation matching the running CPU.
///
/// @tparam Float Floating-point type
///
/// @param[in] ctx CPU dispatch context carrying the enabled CPU extensions
template <typename Float>
void compute_centroids(const dal::backend::context_cpu& ctx,
                       const Float* data,
                       const std::int32_t* labels,
                       std::int64_t row_count,
                       std::int64_t col_count,
                       std::int64_t cluster_count,
                       Float* centroids);

/// Dispatch `compute_medoids` to the instantiation matching the running CPU.
///
/// @tparam Float Floating-point type
///
/// @param[in] ctx CPU dispatch context carrying the enabled CPU extensions
template <typename Float>
void compute_medoids(const dal::backend::context_cpu& ctx,
                     const Float* data,
                     const std::int32_t* labels,
                     std::int64_t row_count,
                     std::int64_t col_count,
                     std::int64_t cluster_count,
                     const Float* centroids,
                     Float* medoids);

} // namespace oneapi::dal::hdbscan::backend
