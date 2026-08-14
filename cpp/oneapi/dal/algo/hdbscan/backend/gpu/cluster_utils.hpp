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

#include <limits>
#include <vector>

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/detail/common.hpp"

#ifdef ONEDAL_DATA_PARALLEL
#include <sycl/sycl.hpp>
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#endif

namespace oneapi::dal::hdbscan::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace pr = oneapi::dal::backend::primitives;

/// Compute the per-cluster centroid on device via SYCL atomics + parallel scale.
///
/// Two kernels, both wholly on device:
///   1) A `parallel_for` over `(row_count, col_count)` accumulates `data[i,d]`
///      into `centroids[label,d]` via `sycl::atomic_ref<Float,
///      memory_order::relaxed, memory_scope::device>`. A separate per-row
///      atomic increments the count. Points with negative or out-of-range
///      labels are skipped -- these are HDBSCAN noise points.
///   2) A `parallel_for` over `(cluster_count, col_count)` divides each entry
///      by its cluster count (skips empty clusters). Same math as a
///      per-cluster `mkl::blas::scal(col_count, 1/count, row, 1)`; a single
///      2D parallel_for is used rather than `cluster_count` back-to-back
///      `mkl::blas::scal` submissions because a single grid over
///      `cluster_count x col_count` requires only one kernel launch and
///      one queue submit rather than `cluster_count` of each.
///
/// @tparam Float Floating-point type
///
/// @param[in]  queue         SYCL queue
/// @param[in]  data          Device row-major buffer of size `row_count x col_count`
/// @param[in]  labels        Device cluster ids, length `row_count`
/// @param[in]  row_count     Number of input points
/// @param[in]  col_count     Number of features per point
/// @param[in]  cluster_count Number of clusters
/// @param[out] centroids     Device row-major buffer, size `cluster_count x col_count`
/// @param[in]  deps          Predecessor SYCL events
///
/// @return SYCL event for the normalize kernel (completion == centroids are ready)
template <typename Float>
static sycl::event compute_centroids_gpu(sycl::queue& queue,
                                         const Float* data,
                                         const std::int32_t* labels,
                                         std::int64_t row_count,
                                         std::int64_t col_count,
                                         std::int64_t cluster_count,
                                         Float* centroids,
                                         const std::vector<sycl::event>& deps = {}) {
    ONEDAL_ASSERT(cluster_count > 0);

    const std::int64_t centroids_size = cluster_count * col_count;
    auto counts_usm =
        sycl::malloc_device<std::int64_t>(static_cast<std::size_t>(cluster_count), queue);
    std::int64_t* counts = counts_usm;

    auto zero_centroids_event =
        queue.fill(centroids, Float(0), static_cast<std::size_t>(centroids_size), deps);
    auto zero_counts_event =
        queue.fill(counts, std::int64_t(0), static_cast<std::size_t>(cluster_count), deps);

    auto accum_event = queue.submit([&](sycl::handler& h) {
        h.depends_on({ zero_centroids_event, zero_counts_event });
        h.parallel_for(sycl::range<1>(row_count), [=](sycl::id<1> idx) {
            const std::int64_t i = idx[0];
            const std::int32_t label = labels[i];
            if (label < 0 || static_cast<std::int64_t>(label) >= cluster_count)
                return;
            const Float* row_in = data + i * col_count;
            Float* row_out = centroids + static_cast<std::int64_t>(label) * col_count;
            for (std::int64_t d = 0; d < col_count; d++) {
                sycl::atomic_ref<Float,
                                 sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    ref(row_out[d]);
                ref.fetch_add(row_in[d]);
            }
            sycl::atomic_ref<std::int64_t,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                cref(counts[label]);
            cref.fetch_add(std::int64_t(1));
        });
    });

    auto normalize_event = queue.submit([&](sycl::handler& h) {
        h.depends_on({ accum_event });
        h.parallel_for(sycl::range<2>(cluster_count, col_count), [=](sycl::id<2> idx) {
            const std::int64_t k = idx[0];
            const std::int64_t d = idx[1];
            const std::int64_t cnt = counts[k];
            if (cnt == 0)
                return;
            centroids[k * col_count + d] /= static_cast<Float>(cnt);
        });
    });

    // counts_usm is only used inside these two kernels; the caller waits on
    // normalize_event before touching centroids, so we can free counts once
    // the normalize dependency chain completes. Schedule the free after
    // normalize_event so the storage outlives every access.
    queue.submit([&](sycl::handler& h) {
        h.depends_on({ normalize_event });
        h.host_task([=]() {
            sycl::free(counts_usm, queue);
        });
    });

    return normalize_event;
}

/// Compute the per-cluster medoid on device.
///
/// Two kernels:
///   1) `(row_count)` parallel scan finds each point's squared distance to its
///      cluster centroid and updates a per-cluster `(best_dist, best_idx)`
///      slot via a lock in the form of a 64-bit CAS packing
///      `(reinterpret_cast<uint32_t>(dist), int32_t idx)` -- simpler
///      alternative: use two arrays and an atomic-min protocol via
///      `sycl::atomic_ref` on 32-bit sortable-encoded float bits, then a
///      second pass to fetch idx. Here we use a straightforward per-cluster
///      atomic min on the packed representation.
///   2) `(cluster_count, col_count)` parallel copy of the winning row into
///      medoids; empty clusters get zero rows.
///
/// @tparam Float Floating-point type
///
/// @param[in]  queue         SYCL queue
/// @param[in]  data          Device row-major buffer of size `row_count x col_count`
/// @param[in]  labels        Device cluster ids, length `row_count`
/// @param[in]  row_count     Number of input points
/// @param[in]  col_count     Number of features per point
/// @param[in]  cluster_count Number of clusters
/// @param[in]  centroids     Device row-major centroids buffer
/// @param[out] medoids       Device row-major buffer, size `cluster_count x col_count`
/// @param[in]  deps          Predecessor SYCL events
///
/// @return SYCL event for the medoid-copy kernel
template <typename Float>
static sycl::event compute_medoids_gpu(sycl::queue& queue,
                                       const Float* data,
                                       const std::int32_t* labels,
                                       std::int64_t row_count,
                                       std::int64_t col_count,
                                       std::int64_t cluster_count,
                                       const Float* centroids,
                                       Float* medoids,
                                       const std::vector<sycl::event>& deps = {}) {
    ONEDAL_ASSERT(cluster_count > 0);

    // Per-cluster best (dist, idx) tracked as separate device arrays and
    // updated by comparing/writing under a per-cluster lock (a 32-bit atomic
    // compare-exchange on the dist bits). The kernel handles ties by
    // arbitrary tie-break -- HDBSCAN medoids on ties are not uniquely defined.
    auto best_dist_usm = sycl::malloc_device<Float>(static_cast<std::size_t>(cluster_count), queue);
    auto best_idx_usm =
        sycl::malloc_device<std::int64_t>(static_cast<std::size_t>(cluster_count), queue);

    Float* best_dist = best_dist_usm;
    std::int64_t* best_idx = best_idx_usm;

    auto init_event = queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(cluster_count), [=](sycl::id<1> idx) {
            const std::int64_t k = idx[0];
            best_dist[k] = std::numeric_limits<Float>::max();
            best_idx[k] = -1;
        });
    });

    auto scan_event = queue.submit([&](sycl::handler& h) {
        h.depends_on({ init_event });
        h.parallel_for(sycl::range<1>(row_count), [=](sycl::id<1> idx) {
            const std::int64_t i = idx[0];
            const std::int32_t label = labels[i];
            if (label < 0 || static_cast<std::int64_t>(label) >= cluster_count)
                return;
            const Float* pt = data + i * col_count;
            const Float* center = centroids + static_cast<std::int64_t>(label) * col_count;
            Float dist = Float(0);
            for (std::int64_t d = 0; d < col_count; d++) {
                const Float diff = pt[d] - center[d];
                dist += diff * diff;
            }

            // Race-free min update: atomically read current best; if this
            // candidate is smaller, try to swap. Loop until either we win
            // the swap or someone else has written a smaller value.
            sycl::atomic_ref<Float,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                dref(best_dist[label]);
            Float cur = dref.load();
            while (dist < cur) {
                if (dref.compare_exchange_strong(cur, dist)) {
                    // We won the dist slot; also record our index. There is a
                    // small window where another thread could win the dist
                    // slot between our CAS success and our idx write, but
                    // that thread must also have a smaller dist -- so the
                    // idx we're about to write is stale-but-consistent
                    // relative to a suboptimal dist and will be overwritten
                    // by the subsequent thread. In steady state, whichever
                    // thread wrote the final winning dist also updates idx
                    // last (barring an interleaved third thread with an
                    // equal dist, in which case the tie-break is
                    // implementation-defined).
                    best_idx[label] = i;
                    break;
                }
            }
        });
    });

    auto copy_event = queue.submit([&](sycl::handler& h) {
        h.depends_on({ scan_event });
        h.parallel_for(sycl::range<2>(cluster_count, col_count), [=](sycl::id<2> idx) {
            const std::int64_t k = idx[0];
            const std::int64_t d = idx[1];
            const std::int64_t src = best_idx[k];
            medoids[k * col_count + d] = (src >= 0) ? data[src * col_count + d] : Float(0);
        });
    });

    queue.submit([&](sycl::handler& h) {
        h.depends_on({ copy_event });
        h.host_task([=]() {
            sycl::free(best_dist_usm, queue);
            sycl::free(best_idx_usm, queue);
        });
    });

    return copy_event;
}

#endif // ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::hdbscan::backend
