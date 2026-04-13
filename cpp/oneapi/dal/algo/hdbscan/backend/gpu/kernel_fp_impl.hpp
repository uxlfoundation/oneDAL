/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

#include "oneapi/dal/algo/hdbscan/backend/gpu/kernels_fp.hpp"
#include "oneapi/dal/detail/profiler.hpp"

namespace oneapi::dal::hdbscan::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

// Helper: Euclidean distance between two rows of a host matrix
template <typename Float>
static inline Float euclidean_dist(const Float* data,
                                   std::int64_t i,
                                   std::int64_t j,
                                   std::int64_t cols) {
    Float sum = Float(0);
    const Float* ri = data + i * cols;
    const Float* rj = data + j * cols;
    for (std::int64_t d = 0; d < cols; d++) {
        Float v = ri[d] - rj[d];
        sum += v * v;
    }
    return static_cast<Float>(std::sqrt(static_cast<double>(sum)));
}

/// Compute core distances on host. No O(n^2) distance matrix needed.
/// Core distance of point i = distance to (min_samples-1)-th nearest OTHER point.
template <typename Float>
sycl::event kernels_fp<Float>::compute_core_distances(sycl::queue& queue,
                                                      const pr::ndview<Float, 2>& data,
                                                      pr::ndview<Float, 1>& core_distances,
                                                      std::int64_t min_samples,
                                                      const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(compute_core_distances, queue);

    const std::int64_t n = data.get_dimension(0);
    const std::int64_t cols = data.get_dimension(1);

    ONEDAL_ASSERT(n > 0);
    ONEDAL_ASSERT(cols > 0);
    ONEDAL_ASSERT(core_distances.get_dimension(0) == n);
    ONEDAL_ASSERT(min_samples >= 1);
    ONEDAL_ASSERT(min_samples <= n);

    auto data_host = data.to_host(queue, deps);
    const Float* dp = data_host.get_data();

    auto core_host = pr::ndarray<Float, 1>::empty(n);
    Float* cp = core_host.get_mutable_data();

    const std::int64_t k = min_samples - 1;
    std::vector<Float> dists(n);

    for (std::int64_t i = 0; i < n; i++) {
        for (std::int64_t j = 0; j < n; j++) {
            dists[j] = (j == i) ? Float(0) : euclidean_dist(dp, i, j, cols);
        }
        std::int64_t target = k + 1; // +1 to skip self-distance at position 0
        if (target >= n)
            target = n - 1;
        std::nth_element(dists.begin(), dists.begin() + target, dists.end());
        cp[i] = dists[target];
    }

    queue.memcpy(core_distances.get_mutable_data(), cp, n * sizeof(Float)).wait_and_throw();
    return sycl::event{};
}

/// Build MST using Prim's algorithm on host. Distances computed on-the-fly.
template <typename Float>
sycl::event kernels_fp<Float>::build_mst(sycl::queue& queue,
                                         const pr::ndview<Float, 2>& data,
                                         const pr::ndview<Float, 1>& core_distances,
                                         pr::ndview<std::int32_t, 1>& mst_from,
                                         pr::ndview<std::int32_t, 1>& mst_to,
                                         pr::ndview<Float, 1>& mst_weights,
                                         const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(build_mst, queue);

    const std::int64_t n = data.get_dimension(0);
    const std::int64_t cols = data.get_dimension(1);
    const std::int64_t edge_count = n - 1;

    ONEDAL_ASSERT(n > 1);
    ONEDAL_ASSERT(mst_from.get_dimension(0) == edge_count);
    ONEDAL_ASSERT(mst_to.get_dimension(0) == edge_count);
    ONEDAL_ASSERT(mst_weights.get_dimension(0) == edge_count);

    auto data_host = data.to_host(queue, deps);
    auto cd_host = core_distances.to_host(queue, deps);
    const Float* dp = data_host.get_data();
    const Float* cd = cd_host.get_data();

    auto mf = pr::ndarray<std::int32_t, 1>::empty(edge_count);
    auto mt = pr::ndarray<std::int32_t, 1>::empty(edge_count);
    auto mw = pr::ndarray<Float, 1>::empty(edge_count);
    auto* fp = mf.get_mutable_data();
    auto* tp = mt.get_mutable_data();
    auto* wp = mw.get_mutable_data();

    std::vector<Float> min_edge(n, std::numeric_limits<Float>::max());
    std::vector<std::int32_t> min_from(n, 0);
    std::vector<bool> in_mst(n, false);

    in_mst[0] = true;
    for (std::int64_t j = 1; j < n; j++) {
        Float dist = euclidean_dist(dp, 0, j, cols);
        Float mrd = std::max({ cd[0], cd[j], dist });
        min_edge[j] = mrd;
    }

    for (std::int64_t e = 0; e < edge_count; e++) {
        std::int32_t best = -1;
        Float best_w = std::numeric_limits<Float>::max();
        for (std::int64_t j = 0; j < n; j++) {
            if (!in_mst[j] && min_edge[j] < best_w) {
                best_w = min_edge[j];
                best = static_cast<std::int32_t>(j);
            }
        }
        ONEDAL_ASSERT(best >= 0);

        fp[e] = min_from[best];
        tp[e] = best;
        wp[e] = best_w;
        in_mst[best] = true;

        for (std::int64_t j = 0; j < n; j++) {
            if (in_mst[j])
                continue;
            Float dist = euclidean_dist(dp, best, j, cols);
            Float mrd = std::max({ cd[best], cd[j], dist });
            if (mrd < min_edge[j]) {
                min_edge[j] = mrd;
                min_from[j] = best;
            }
        }
    }

    queue.memcpy(mst_from.get_mutable_data(), fp, edge_count * sizeof(std::int32_t))
        .wait_and_throw();
    queue.memcpy(mst_to.get_mutable_data(), tp, edge_count * sizeof(std::int32_t)).wait_and_throw();
    queue.memcpy(mst_weights.get_mutable_data(), wp, edge_count * sizeof(Float)).wait_and_throw();

    return sycl::event{};
}

/// Sort MST edges by weight ascending on host.
template <typename Float>
sycl::event kernels_fp<Float>::sort_mst_by_weight(sycl::queue& queue,
                                                  pr::ndview<std::int32_t, 1>& mst_from,
                                                  pr::ndview<std::int32_t, 1>& mst_to,
                                                  pr::ndview<Float, 1>& mst_weights,
                                                  std::int64_t edge_count,
                                                  const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(sort_mst_by_weight, queue);

    auto fh = mst_from.to_host(queue, deps);
    auto th = mst_to.to_host(queue, deps);
    auto wh = mst_weights.to_host(queue, deps);

    auto* fp = fh.get_mutable_data();
    auto* tp = th.get_mutable_data();
    auto* wp = wh.get_mutable_data();

    std::vector<std::int64_t> order(edge_count);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::int64_t a, std::int64_t b) {
        return wp[a] < wp[b];
    });

    auto fs = pr::ndarray<std::int32_t, 1>::empty(edge_count);
    auto ts = pr::ndarray<std::int32_t, 1>::empty(edge_count);
    auto ws = pr::ndarray<Float, 1>::empty(edge_count);
    for (std::int64_t i = 0; i < edge_count; i++) {
        fs.get_mutable_data()[i] = fp[order[i]];
        ts.get_mutable_data()[i] = tp[order[i]];
        ws.get_mutable_data()[i] = wp[order[i]];
    }

    queue.memcpy(mst_from.get_mutable_data(), fs.get_data(), edge_count * sizeof(std::int32_t))
        .wait_and_throw();
    queue.memcpy(mst_to.get_mutable_data(), ts.get_data(), edge_count * sizeof(std::int32_t))
        .wait_and_throw();
    queue.memcpy(mst_weights.get_mutable_data(), ws.get_data(), edge_count * sizeof(Float))
        .wait_and_throw();

    return sycl::event{};
}

/// Extract flat clusters from the sorted MST.
template <typename Float>
sycl::event kernels_fp<Float>::extract_clusters(sycl::queue& queue,
                                                const pr::ndview<std::int32_t, 1>& mst_from,
                                                const pr::ndview<std::int32_t, 1>& mst_to,
                                                const pr::ndview<Float, 1>& mst_weights,
                                                pr::ndview<std::int32_t, 1>& responses,
                                                std::int64_t row_count,
                                                std::int64_t min_cluster_size,
                                                const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(extract_clusters, queue);

    const std::int64_t edge_count = row_count - 1;
    const std::int64_t n_dendro_nodes = row_count - 1;

    auto fh = mst_from.to_host(queue, deps);
    auto th = mst_to.to_host(queue, deps);
    auto wh = mst_weights.to_host(queue, deps);

    const auto* from_ptr = fh.get_data();
    const auto* to_ptr = th.get_data();
    const auto* weights_ptr = wh.get_data();

    // Step 1: Kruskal dendrogram
    std::vector<std::int32_t> uf_parent(row_count);
    std::vector<std::int32_t> comp_size(row_count, 1);
    std::iota(uf_parent.begin(), uf_parent.end(), 0);

    auto uf_find = [&](std::int32_t x) -> std::int32_t {
        while (uf_parent[x] != x) {
            uf_parent[x] = uf_parent[uf_parent[x]];
            x = uf_parent[x];
        }
        return x;
    };

    struct DendroNode {
        std::int32_t left;
        std::int32_t right;
        Float weight;
        std::int32_t size;
    };
    std::vector<DendroNode> dendro(n_dendro_nodes);

    std::vector<std::int32_t> comp_to_node(row_count);
    std::iota(comp_to_node.begin(), comp_to_node.end(), 0);

    for (std::int64_t e = 0; e < edge_count; e++) {
        std::int32_t ru = uf_find(from_ptr[e]);
        std::int32_t rv = uf_find(to_ptr[e]);
        if (ru == rv)
            continue;

        std::int32_t node_id = static_cast<std::int32_t>(row_count + e);
        std::int32_t left_node = comp_to_node[ru];
        std::int32_t right_node = comp_to_node[rv];
        std::int32_t new_size = comp_size[ru] + comp_size[rv];

        dendro[e] = { left_node, right_node, weights_ptr[e], new_size };

        if (comp_size[ru] < comp_size[rv]) {
            uf_parent[ru] = rv;
            comp_size[rv] = new_size;
            comp_to_node[rv] = node_id;
        }
        else {
            uf_parent[rv] = ru;
            comp_size[ru] = new_size;
            comp_to_node[ru] = node_id;
        }
    }

    // Step 2: Condensed tree
    const std::int64_t total_nodes = 2 * row_count - 1;

    std::vector<std::int32_t> node_size(total_nodes, 0);
    for (std::int64_t i = 0; i < row_count; i++)
        node_size[i] = 1;
    for (std::int64_t e = 0; e < n_dendro_nodes; e++)
        node_size[row_count + e] = dendro[e].size;

    std::vector<std::int32_t> left_child(total_nodes, -1);
    std::vector<std::int32_t> right_child(total_nodes, -1);
    std::vector<Float> node_weight(total_nodes, Float(0));
    for (std::int64_t e = 0; e < n_dendro_nodes; e++) {
        std::int64_t nid = row_count + e;
        left_child[nid] = dendro[e].left;
        right_child[nid] = dendro[e].right;
        node_weight[nid] = dendro[e].weight;
    }

    struct CondensedEdge {
        std::int32_t parent;
        std::int32_t child;
        Float lambda_val;
        std::int32_t child_size;
    };
    std::vector<CondensedEdge> condensed;
    condensed.reserve(2 * row_count);

    std::int32_t root = -1;
    for (std::int64_t e = n_dendro_nodes - 1; e >= 0; e--) {
        if (dendro[e].size > 0) {
            root = static_cast<std::int32_t>(row_count + e);
            break;
        }
    }
    if (root < 0) {
        auto rh = pr::ndarray<std::int32_t, 1>::empty(row_count);
        for (std::int64_t i = 0; i < row_count; i++)
            rh.get_mutable_data()[i] = -1;
        queue.memcpy(responses.get_mutable_data(), rh.get_data(), row_count * sizeof(std::int32_t))
            .wait_and_throw();
        return sycl::event{};
    }

    std::int32_t next_cid = static_cast<std::int32_t>(row_count);
    std::vector<std::int32_t> dendro_to_cluster(total_nodes, -1);
    dendro_to_cluster[root] = next_cid++;

    std::function<void(std::int32_t, std::vector<std::int32_t>&)> collect_leaves;
    collect_leaves = [&](std::int32_t nid, std::vector<std::int32_t>& out) {
        if (nid < row_count) {
            out.push_back(nid);
            return;
        }
        collect_leaves(left_child[nid], out);
        collect_leaves(right_child[nid], out);
    };

    struct StackItem {
        std::int32_t node;
        std::int32_t cluster;
    };
    std::vector<StackItem> stack;
    stack.push_back({ root, dendro_to_cluster[root] });

    while (!stack.empty()) {
        auto [nid, parent_cid] = stack.back();
        stack.pop_back();

        if (nid < row_count)
            continue;

        std::int32_t lc = left_child[nid];
        std::int32_t rc = right_child[nid];
        if (lc < 0 || rc < 0)
            continue;

        std::int32_t ls = node_size[lc];
        std::int32_t rs = node_size[rc];
        Float lambda = (node_weight[nid] > Float(0)) ? Float(1) / node_weight[nid]
                                                     : std::numeric_limits<Float>::max();

        bool l_big = ls >= min_cluster_size;
        bool r_big = rs >= min_cluster_size;

        if (l_big && r_big) {
            std::int32_t lcid = next_cid++;
            std::int32_t rcid = next_cid++;
            dendro_to_cluster[lc] = lcid;
            dendro_to_cluster[rc] = rcid;
            condensed.push_back({ parent_cid, lcid, lambda, ls });
            condensed.push_back({ parent_cid, rcid, lambda, rs });
            stack.push_back({ lc, lcid });
            stack.push_back({ rc, rcid });
        }
        else if (l_big) {
            dendro_to_cluster[lc] = parent_cid;
            std::vector<std::int32_t> fallen;
            collect_leaves(rc, fallen);
            for (auto pt : fallen)
                condensed.push_back({ parent_cid, pt, lambda, 1 });
            stack.push_back({ lc, parent_cid });
        }
        else if (r_big) {
            dendro_to_cluster[rc] = parent_cid;
            std::vector<std::int32_t> fallen;
            collect_leaves(lc, fallen);
            for (auto pt : fallen)
                condensed.push_back({ parent_cid, pt, lambda, 1 });
            stack.push_back({ rc, parent_cid });
        }
        else {
            std::vector<std::int32_t> fallen;
            collect_leaves(lc, fallen);
            collect_leaves(rc, fallen);
            for (auto pt : fallen)
                condensed.push_back({ parent_cid, pt, lambda, 1 });
        }
    }

    // Step 3: Stability & EOM
    const std::int32_t n_clusters = next_cid;

    std::vector<Float> stability(n_clusters, Float(0));
    std::vector<Float> lambda_birth(n_clusters, Float(0));
    std::vector<bool> is_leaf_cluster(n_clusters, true);
    std::vector<std::vector<std::int32_t>> child_clusters(n_clusters);

    for (const auto& e : condensed) {
        if (e.child >= row_count) {
            lambda_birth[e.child] = e.lambda_val;
            is_leaf_cluster[e.parent] = false;
            child_clusters[e.parent].push_back(e.child);
        }
    }

    for (const auto& e : condensed) {
        Float birth = lambda_birth[e.parent];
        Float contrib = (e.lambda_val - birth) * static_cast<Float>(e.child_size);
        if (contrib > Float(0)) {
            stability[e.parent] += contrib;
        }
    }

    std::vector<bool> is_selected(n_clusters, true);

    for (std::int32_t c = n_clusters - 1; c >= static_cast<std::int32_t>(row_count); c--) {
        if (is_leaf_cluster[c])
            continue;

        Float child_sum = Float(0);
        for (auto cc : child_clusters[c])
            child_sum += stability[cc];

        if (child_sum > stability[c]) {
            is_selected[c] = false;
            stability[c] = child_sum;
        }
        else {
            std::vector<std::int32_t> desc_stack;
            for (auto cc : child_clusters[c])
                desc_stack.push_back(cc);
            while (!desc_stack.empty()) {
                auto d = desc_stack.back();
                desc_stack.pop_back();
                is_selected[d] = false;
                for (auto dd : child_clusters[d])
                    desc_stack.push_back(dd);
            }
        }
    }

    // Step 4: Label points
    std::int32_t label_counter = 0;
    std::vector<std::int32_t> cluster_label(n_clusters, -1);
    for (std::int32_t c = static_cast<std::int32_t>(row_count); c < n_clusters; c++) {
        if (is_selected[c])
            cluster_label[c] = label_counter++;
    }

    std::vector<std::int32_t> cluster_parent(n_clusters, -1);
    for (const auto& e : condensed) {
        if (e.child >= row_count)
            cluster_parent[e.child] = e.parent;
    }

    std::vector<std::int32_t> point_fell_from(row_count, -1);
    for (const auto& e : condensed) {
        if (e.child < row_count)
            point_fell_from[e.child] = e.parent;
    }

    auto rh = pr::ndarray<std::int32_t, 1>::empty(row_count);
    auto* rp = rh.get_mutable_data();

    for (std::int64_t i = 0; i < row_count; i++) {
        rp[i] = -1;
        std::int32_t c = point_fell_from[i];
        while (c >= static_cast<std::int32_t>(row_count) && c < n_clusters) {
            if (is_selected[c]) {
                rp[i] = cluster_label[c];
                break;
            }
            c = cluster_parent[c];
        }
    }

    // Points never ejected
    std::vector<std::int32_t> dendro_parent(total_nodes, -1);
    for (std::int64_t e = 0; e < n_dendro_nodes; e++) {
        std::int64_t nid = row_count + e;
        if (dendro[e].size > 0) {
            dendro_parent[dendro[e].left] = static_cast<std::int32_t>(nid);
            dendro_parent[dendro[e].right] = static_cast<std::int32_t>(nid);
        }
    }

    for (std::int64_t i = 0; i < row_count; i++) {
        if (point_fell_from[i] >= 0)
            continue;

        std::int32_t nid = static_cast<std::int32_t>(i);
        while (nid >= 0 && nid < static_cast<std::int32_t>(total_nodes)) {
            std::int32_t cid = dendro_to_cluster[nid];
            if (cid >= static_cast<std::int32_t>(row_count)) {
                std::int32_t c = cid;
                while (c >= static_cast<std::int32_t>(row_count) && c < n_clusters) {
                    if (is_selected[c]) {
                        rp[i] = cluster_label[c];
                        break;
                    }
                    c = cluster_parent[c];
                }
                break;
            }
            nid = dendro_parent[nid];
        }
    }

    queue.memcpy(responses.get_mutable_data(), rp, row_count * sizeof(std::int32_t))
        .wait_and_throw();
    return sycl::event{};
}

#endif

} // namespace oneapi::dal::hdbscan::backend
