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
#include <limits>
#include <numeric>
#include <vector>

#include "oneapi/dal/algo/hdbscan/backend/gpu/kernels_fp.hpp"
#include "oneapi/dal/detail/profiler.hpp"

namespace oneapi::dal::hdbscan::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

/// Compute core distances on host.
/// Core distance of point i = Euclidean distance to its (min_samples)-th nearest neighbor.
/// sklearn convention: min_samples includes the point itself, so core_dist = dist to
/// (min_samples - 1)-th nearest *other* point.
template <typename Float>
sycl::event kernels_fp<Float>::compute_core_distances(sycl::queue& queue,
                                                      const pr::ndview<Float, 2>& data,
                                                      pr::ndview<Float, 1>& core_distances,
                                                      std::int64_t min_samples,
                                                      const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(compute_core_distances, queue);

    const std::int64_t row_count = data.get_dimension(0);
    const std::int64_t column_count = data.get_dimension(1);

    ONEDAL_ASSERT(row_count > 0);
    ONEDAL_ASSERT(column_count > 0);
    ONEDAL_ASSERT(core_distances.get_dimension(0) == row_count);
    ONEDAL_ASSERT(min_samples >= 1);
    ONEDAL_ASSERT(min_samples <= row_count);

    // Transfer data to host
    auto data_host = pr::ndarray<Float, 2>::empty({ row_count, column_count });
    queue
        .memcpy(data_host.get_mutable_data(),
                data.get_data(),
                row_count * column_count * sizeof(Float),
                deps)
        .wait_and_throw();

    const Float* dp = data_host.get_data();
    auto core_host = pr::ndarray<Float, 1>::empty(row_count);
    Float* cp = core_host.get_mutable_data();

    // k-th neighbor index (0-based, excluding self).
    // sklearn: min_samples includes self, so the core distance is the distance
    // to the (min_samples - 1)-th nearest OTHER point.
    const std::int64_t k = min_samples - 1; // 0-based index into sorted distances

    std::vector<Float> dists(row_count);

    for (std::int64_t i = 0; i < row_count; i++) {
        // Compute Euclidean distances from point i to all points
        for (std::int64_t j = 0; j < row_count; j++) {
            if (j == i) {
                dists[j] = Float(0);
                continue;
            }
            Float sum = Float(0);
            for (std::int64_t d = 0; d < column_count; d++) {
                Float val = dp[i * column_count + d] - dp[j * column_count + d];
                sum += val * val;
            }
            dists[j] = static_cast<Float>(std::sqrt(static_cast<double>(sum)));
        }

        // Partial sort to find the (k+1)-th smallest (index k+1 because index 0 is self=0)
        // We need the k-th nearest OTHER point, which is at position k+1 in all-distances
        // (since distance to self = 0 is at position 0 after sort).
        std::int64_t target = k + 1; // +1 to skip self
        if (target >= row_count) {
            target = row_count - 1;
        }
        std::nth_element(dists.begin(), dists.begin() + target, dists.end());
        cp[i] = dists[target];
    }

    // Copy to device
    queue.memcpy(core_distances.get_mutable_data(), cp, row_count * sizeof(Float)).wait_and_throw();

    return sycl::event{};
}

/// Build MST using Prim's algorithm with mutual reachability distance.
/// mutual_reachability_dist(i, j) = max(core_dist[i], core_dist[j], euclidean_dist(i, j))
template <typename Float>
sycl::event kernels_fp<Float>::build_mst(sycl::queue& queue,
                                         const pr::ndview<Float, 2>& data,
                                         const pr::ndview<Float, 1>& core_distances,
                                         pr::ndview<std::int32_t, 1>& mst_from,
                                         pr::ndview<std::int32_t, 1>& mst_to,
                                         pr::ndview<Float, 1>& mst_weights,
                                         const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(build_mst, queue);

    const std::int64_t row_count = data.get_dimension(0);
    const std::int64_t column_count = data.get_dimension(1);
    const std::int64_t edge_count = row_count - 1;

    ONEDAL_ASSERT(row_count > 1);
    ONEDAL_ASSERT(mst_from.get_dimension(0) == edge_count);
    ONEDAL_ASSERT(mst_to.get_dimension(0) == edge_count);
    ONEDAL_ASSERT(mst_weights.get_dimension(0) == edge_count);

    // Transfer to host
    auto data_host = pr::ndarray<Float, 2>::empty({ row_count, column_count });
    auto cdist_host = pr::ndarray<Float, 1>::empty(row_count);

    queue
        .memcpy(data_host.get_mutable_data(),
                data.get_data(),
                row_count * column_count * sizeof(Float),
                deps)
        .wait_and_throw();
    queue
        .memcpy(cdist_host.get_mutable_data(), core_distances.get_data(), row_count * sizeof(Float))
        .wait_and_throw();

    const Float* dp = data_host.get_data();
    const Float* cd = cdist_host.get_data();

    auto mf = pr::ndarray<std::int32_t, 1>::empty(edge_count);
    auto mt = pr::ndarray<std::int32_t, 1>::empty(edge_count);
    auto mw = pr::ndarray<Float, 1>::empty(edge_count);
    auto* fp = mf.get_mutable_data();
    auto* tp = mt.get_mutable_data();
    auto* wp = mw.get_mutable_data();

    // Prim's algorithm
    std::vector<Float> min_edge(row_count, std::numeric_limits<Float>::max());
    std::vector<std::int32_t> min_from(row_count, 0);
    std::vector<bool> in_mst(row_count, false);

    in_mst[0] = true;
    for (std::int64_t j = 1; j < row_count; j++) {
        Float sum = Float(0);
        for (std::int64_t d = 0; d < column_count; d++) {
            Float val = dp[d] - dp[j * column_count + d];
            sum += val * val;
        }
        Float dist = static_cast<Float>(std::sqrt(static_cast<double>(sum)));
        Float mrd = std::max({ cd[0], cd[j], dist });
        min_edge[j] = mrd;
    }

    for (std::int64_t e = 0; e < edge_count; e++) {
        std::int32_t best = -1;
        Float best_w = std::numeric_limits<Float>::max();
        for (std::int64_t j = 0; j < row_count; j++) {
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

        for (std::int64_t j = 0; j < row_count; j++) {
            if (in_mst[j])
                continue;
            Float sum = Float(0);
            for (std::int64_t d = 0; d < column_count; d++) {
                Float val = dp[best * column_count + d] - dp[j * column_count + d];
                sum += val * val;
            }
            Float dist = static_cast<Float>(std::sqrt(static_cast<double>(sum)));
            Float mrd = std::max({ cd[best], cd[j], dist });
            if (mrd < min_edge[j]) {
                min_edge[j] = mrd;
                min_from[j] = best;
            }
        }
    }

    // Copy to device
    queue.memcpy(mst_from.get_mutable_data(), fp, edge_count * sizeof(std::int32_t))
        .wait_and_throw();
    queue.memcpy(mst_to.get_mutable_data(), tp, edge_count * sizeof(std::int32_t)).wait_and_throw();
    queue.memcpy(mst_weights.get_mutable_data(), wp, edge_count * sizeof(Float)).wait_and_throw();

    return sycl::event{};
}

/// Sort MST edges by weight (ascending).
template <typename Float>
sycl::event kernels_fp<Float>::sort_mst_by_weight(sycl::queue& queue,
                                                  pr::ndview<std::int32_t, 1>& mst_from,
                                                  pr::ndview<std::int32_t, 1>& mst_to,
                                                  pr::ndview<Float, 1>& mst_weights,
                                                  std::int64_t edge_count,
                                                  const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(sort_mst_by_weight, queue);

    auto fh = pr::ndarray<std::int32_t, 1>::empty(edge_count);
    auto th = pr::ndarray<std::int32_t, 1>::empty(edge_count);
    auto wh = pr::ndarray<Float, 1>::empty(edge_count);

    queue
        .memcpy(fh.get_mutable_data(), mst_from.get_data(), edge_count * sizeof(std::int32_t), deps)
        .wait_and_throw();
    queue.memcpy(th.get_mutable_data(), mst_to.get_data(), edge_count * sizeof(std::int32_t))
        .wait_and_throw();
    queue.memcpy(wh.get_mutable_data(), mst_weights.get_data(), edge_count * sizeof(Float))
        .wait_and_throw();

    auto* fp = fh.get_mutable_data();
    auto* tp = th.get_mutable_data();
    auto* wp = wh.get_mutable_data();

    // Sort by weight using index permutation
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
///
/// Correct HDBSCAN algorithm:
/// 1. Build single-linkage dendrogram (Kruskal: process MST edges smallest→largest,
///    merge components via union-find, each merge creates a dendrogram node).
/// 2. Traverse dendrogram top→down to build the condensed tree:
///    at each split, if both children have >= min_cluster_size points they become
///    child clusters; otherwise the small side's points fall out as noise.
/// 3. Compute cluster stability and apply Excess-of-Mass (EOM) selection.
/// 4. Label every point with its selected cluster (−1 = noise).
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
    const std::int64_t n_dendro_nodes = row_count - 1; // one node per merge

    // Transfer MST to host
    auto fh = pr::ndarray<std::int32_t, 1>::empty(edge_count);
    auto th = pr::ndarray<std::int32_t, 1>::empty(edge_count);
    auto wh = pr::ndarray<Float, 1>::empty(edge_count);

    queue
        .memcpy(fh.get_mutable_data(), mst_from.get_data(), edge_count * sizeof(std::int32_t), deps)
        .wait_and_throw();
    queue.memcpy(th.get_mutable_data(), mst_to.get_data(), edge_count * sizeof(std::int32_t))
        .wait_and_throw();
    queue.memcpy(wh.get_mutable_data(), mst_weights.get_data(), edge_count * sizeof(Float))
        .wait_and_throw();

    const auto* from_ptr = fh.get_data();
    const auto* to_ptr = th.get_data();
    const auto* weights_ptr = wh.get_data();

    // ---------------------------------------------------------------
    // Step 1: Build single-linkage dendrogram via Kruskal (union-find)
    // Edges are already sorted ascending by weight.
    // Each merge creates dendrogram node with id = row_count + merge_index
    // ---------------------------------------------------------------

    // Union-Find: parent[] and comp_size[] for components 0..row_count-1
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

    // Dendrogram node: left child, right child, weight, size
    struct DendroNode {
        std::int32_t left;
        std::int32_t right;
        Float weight;
        std::int32_t size;
    };
    std::vector<DendroNode> dendro(n_dendro_nodes);

    // Map: component root → current dendrogram node id
    // Initially each point is its own "leaf" with id = point_index
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

        // Union: merge smaller into larger
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

    // ---------------------------------------------------------------
    // Step 2: Build condensed tree by traversing dendrogram top→down
    // ---------------------------------------------------------------
    // Total nodes in the id space: leaves [0, row_count) and internal [row_count, 2*row_count-1)

    const std::int64_t total_nodes = 2 * row_count - 1;

    // Sizes of each node (leaves = 1, internal = stored in dendro)
    std::vector<std::int32_t> node_size(total_nodes, 0);
    for (std::int64_t i = 0; i < row_count; i++)
        node_size[i] = 1;
    for (std::int64_t e = 0; e < n_dendro_nodes; e++)
        node_size[row_count + e] = dendro[e].size;

    // Children of each internal node
    std::vector<std::int32_t> left_child(total_nodes, -1);
    std::vector<std::int32_t> right_child(total_nodes, -1);
    std::vector<Float> node_weight(total_nodes, Float(0));
    for (std::int64_t e = 0; e < n_dendro_nodes; e++) {
        std::int64_t nid = row_count + e;
        left_child[nid] = dendro[e].left;
        right_child[nid] = dendro[e].right;
        node_weight[nid] = dendro[e].weight;
    }

    // Condensed tree edges: (parent_cluster, child, lambda_val, child_size)
    struct CondensedEdge {
        std::int32_t parent;
        std::int32_t child;
        Float lambda_val;
        std::int32_t child_size;
    };
    std::vector<CondensedEdge> condensed;
    condensed.reserve(2 * row_count);

    // BFS / DFS from root.  The root is the last dendrogram node created.
    // Find root = the node with the largest id that has size > 0.
    std::int32_t root = -1;
    for (std::int64_t e = n_dendro_nodes - 1; e >= 0; e--) {
        if (dendro[e].size > 0) {
            root = static_cast<std::int32_t>(row_count + e);
            break;
        }
    }
    if (root < 0) {
        // Degenerate: no merges happened → all points noise
        auto rh = pr::ndarray<std::int32_t, 1>::empty(row_count);
        for (std::int64_t i = 0; i < row_count; i++)
            rh.get_mutable_data()[i] = -1;
        queue.memcpy(responses.get_mutable_data(), rh.get_data(), row_count * sizeof(std::int32_t))
            .wait_and_throw();
        return sycl::event{};
    }

    // Next available cluster id for the condensed tree
    std::int32_t next_cid = static_cast<std::int32_t>(row_count);

    // Map from dendrogram node id → condensed cluster id (-1 if not a cluster)
    std::vector<std::int32_t> dendro_to_cluster(total_nodes, -1);

    // Assign the root a cluster id
    dendro_to_cluster[root] = next_cid++;

    // Collect all leaf-point ids that belong to a given dendrogram subtree
    // (needed when a small branch falls out)
    std::function<void(std::int32_t, std::vector<std::int32_t>&)> collect_leaves;
    collect_leaves = [&](std::int32_t nid, std::vector<std::int32_t>& out) {
        if (nid < row_count) {
            out.push_back(nid);
            return;
        }
        collect_leaves(left_child[nid], out);
        collect_leaves(right_child[nid], out);
    };

    // DFS stack: (dendrogram_node_id, parent_condensed_cluster_id)
    struct StackItem {
        std::int32_t node;
        std::int32_t cluster;
    };
    std::vector<StackItem> stack;
    stack.push_back({ root, dendro_to_cluster[root] });

    while (!stack.empty()) {
        auto [nid, parent_cid] = stack.back();
        stack.pop_back();

        if (nid < row_count) {
            // Leaf: this single point belongs to parent_cid
            // (will be used for labeling later)
            continue;
        }

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
            // True split: both children become new clusters
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
            // Right side is too small → its points fall out of parent_cid
            // Left side continues as parent_cid (no new cluster)
            dendro_to_cluster[lc] = parent_cid;

            // Record small-side points falling out
            std::vector<std::int32_t> fallen;
            collect_leaves(rc, fallen);
            for (auto pt : fallen) {
                condensed.push_back({ parent_cid, pt, lambda, 1 });
            }

            stack.push_back({ lc, parent_cid });
        }
        else if (r_big) {
            // Left side too small
            dendro_to_cluster[rc] = parent_cid;

            std::vector<std::int32_t> fallen;
            collect_leaves(lc, fallen);
            for (auto pt : fallen) {
                condensed.push_back({ parent_cid, pt, lambda, 1 });
            }

            stack.push_back({ rc, parent_cid });
        }
        else {
            // Both too small → all points fall out of parent_cid
            std::vector<std::int32_t> fallen;
            collect_leaves(lc, fallen);
            collect_leaves(rc, fallen);
            for (auto pt : fallen) {
                condensed.push_back({ parent_cid, pt, lambda, 1 });
            }
        }
    }

    // ---------------------------------------------------------------
    // Step 3: Compute stability & EOM cluster selection
    // ---------------------------------------------------------------
    const std::int32_t n_clusters = next_cid; // total cluster ids allocated

    std::vector<Float> stability(n_clusters, Float(0));
    std::vector<Float> lambda_birth(n_clusters, Float(0));
    std::vector<bool> is_leaf_cluster(n_clusters, true); // leaf = no child clusters
    std::vector<std::vector<std::int32_t>> child_clusters(n_clusters);

    // lambda_birth[c] = the lambda at which cluster c first appeared
    for (const auto& e : condensed) {
        if (e.child >= row_count) {
            // child is a cluster
            lambda_birth[e.child] = e.lambda_val;
            is_leaf_cluster[e.parent] = false;
            child_clusters[e.parent].push_back(e.child);
        }
    }

    // Stability = sum over ALL condensed tree edges where parent=c:
    //   (lambda_val - lambda_birth(c)) * child_size
    // This includes both individual point fallouts (child_size=1) and
    // cluster splits (child_size = size of child cluster).
    for (const auto& e : condensed) {
        Float birth = lambda_birth[e.parent];
        Float contrib = (e.lambda_val - birth) * static_cast<Float>(e.child_size);
        if (contrib > Float(0)) {
            stability[e.parent] += contrib;
        }
    }

    // EOM selection: bottom-up.
    // If sum of children's stability > parent's stability → deselect parent, keep children.
    // Otherwise → keep parent, deselect all descendants.
    std::vector<bool> is_selected(n_clusters, true);

    // Process in reverse id order (children have higher ids than parents)
    for (std::int32_t c = n_clusters - 1; c >= static_cast<std::int32_t>(row_count); c--) {
        if (is_leaf_cluster[c])
            continue; // leaf clusters are always selected initially

        Float child_sum = Float(0);
        for (auto cc : child_clusters[c]) {
            child_sum += stability[cc];
        }

        if (child_sum > stability[c]) {
            // Children win → deselect parent
            is_selected[c] = false;
            stability[c] = child_sum; // propagate upward
        }
        else {
            // Parent wins → deselect all descendant clusters
            std::vector<std::int32_t> desc_stack;
            for (auto cc : child_clusters[c]) {
                desc_stack.push_back(cc);
            }
            while (!desc_stack.empty()) {
                auto d = desc_stack.back();
                desc_stack.pop_back();
                is_selected[d] = false;
                for (auto dd : child_clusters[d]) {
                    desc_stack.push_back(dd);
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // Step 4: Label each point
    // ---------------------------------------------------------------
    // Assign sequential labels 0,1,2,... to selected clusters
    std::int32_t label_counter = 0;
    std::vector<std::int32_t> cluster_label(n_clusters, -1);
    for (std::int32_t c = static_cast<std::int32_t>(row_count); c < n_clusters; c++) {
        if (is_selected[c]) {
            cluster_label[c] = label_counter++;
        }
    }

    // For each point, find which selected cluster it belongs to.
    // Walk the condensed tree: find the deepest selected cluster containing the point.
    // Build point → cluster mapping from condensed edges.

    // First, for each cluster, find its parent cluster (walk condensed edges)
    std::vector<std::int32_t> cluster_parent(n_clusters, -1);
    for (const auto& e : condensed) {
        if (e.child >= row_count) {
            cluster_parent[e.child] = e.parent;
        }
    }

    // For each point, find the cluster it directly fell out of.
    // Then walk up until we find a selected cluster.
    std::vector<std::int32_t> point_fell_from(row_count, -1);
    for (const auto& e : condensed) {
        if (e.child < row_count) {
            point_fell_from[e.child] = e.parent;
        }
    }

    auto rh = pr::ndarray<std::int32_t, 1>::empty(row_count);
    auto* rp = rh.get_mutable_data();

    for (std::int64_t i = 0; i < row_count; i++) {
        rp[i] = -1; // default: noise

        std::int32_t c = point_fell_from[i];
        // Walk up from c to find the nearest selected ancestor
        while (c >= static_cast<std::int32_t>(row_count) && c < n_clusters) {
            if (is_selected[c]) {
                rp[i] = cluster_label[c];
                break;
            }
            c = cluster_parent[c];
        }
    }

    // Points that never appeared in any condensed edge (they are still inside
    // a leaf cluster that was never split). Walk the dendrogram to find them.
    for (std::int64_t i = 0; i < row_count; i++) {
        if (point_fell_from[i] >= 0)
            continue; // already handled

        // This point was never individually ejected.
        // It belongs to the deepest cluster that contains it.
        // Walk dendrogram: find which condensed cluster the dendrogram leaf belongs to.
        // We trace: the point is in some dendrogram internal node → check if that node
        // maps to a condensed cluster → walk up to find selected cluster.

        // Since this point was never ejected, it's part of a leaf cluster in the condensed tree.
        // Will be handled below using the dendrogram parent map.
    }

    // Build dendrogram parent map (internal node → parent internal node)
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
            continue; // already labeled

        // Walk up dendrogram from leaf i to find a node with a condensed cluster id
        std::int32_t nid = static_cast<std::int32_t>(i);
        while (nid >= 0 && nid < static_cast<std::int32_t>(total_nodes)) {
            std::int32_t cid = dendro_to_cluster[nid];
            if (cid >= static_cast<std::int32_t>(row_count)) {
                // Found a condensed cluster. Walk up to find selected one.
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

    // Copy to device
    queue.memcpy(responses.get_mutable_data(), rp, row_count * sizeof(std::int32_t))
        .wait_and_throw();

    return sycl::event{};
}

#endif

} // namespace oneapi::dal::hdbscan::backend
