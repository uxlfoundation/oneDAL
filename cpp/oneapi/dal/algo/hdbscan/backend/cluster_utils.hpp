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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/backend/common.hpp"

extern "C" {
void dgemm_(const char*,
            const char*,
            const std::int64_t*,
            const std::int64_t*,
            const std::int64_t*,
            const double*,
            const double*,
            const std::int64_t*,
            const double*,
            const std::int64_t*,
            const double*,
            double*,
            const std::int64_t*);
void sgemm_(const char*,
            const char*,
            const std::int64_t*,
            const std::int64_t*,
            const std::int64_t*,
            const float*,
            const float*,
            const std::int64_t*,
            const float*,
            const std::int64_t*,
            const float*,
            float*,
            const std::int64_t*);
}

namespace oneapi::dal::hdbscan::backend {

namespace de = oneapi::dal::detail;
namespace bk = oneapi::dal::backend;

namespace {

inline void gemm_call(const char* ta,
                      const char* tb,
                      const std::int64_t* m,
                      const std::int64_t* n,
                      const std::int64_t* k,
                      const float* alpha,
                      const float* a,
                      const std::int64_t* lda,
                      const float* b,
                      const std::int64_t* ldb,
                      const float* beta,
                      float* c,
                      const std::int64_t* ldc) {
    sgemm_(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void gemm_call(const char* ta,
                      const char* tb,
                      const std::int64_t* m,
                      const std::int64_t* n,
                      const std::int64_t* k,
                      const double* alpha,
                      const double* a,
                      const std::int64_t* lda,
                      const double* b,
                      const std::int64_t* ldb,
                      const double* beta,
                      double* c,
                      const std::int64_t* ldc) {
    dgemm_(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

} // anonymous namespace

/// Compute squared Euclidean distance matrix using GEMM.
/// Formula: dist²[i,j] = ||a_i||² + ||a_j||² - 2 * (A * A^T)[i,j]
/// Output: row-major n x n matrix of Euclidean distances (not squared).
template <typename Float>
static void compute_distance_matrix(const Float* data,
                                    Float* dist_matrix,
                                    std::int64_t row_count,
                                    std::int64_t col_count) {
    ONEDAL_ASSERT(row_count > 0);
    ONEDAL_ASSERT(col_count > 0);

    // Step 1: Compute row norms: norm[i] = sum(data[i,:]^2)
    std::vector<Float> norms(row_count);

    de::threader_for_int64(row_count, [&](std::int64_t i) {
        Float sum = Float(0);
        const Float* row = data + i * col_count;
        PRAGMA_OMP_SIMD_ARGS(reduction(+ : sum))
        for (std::int64_t d = 0; d < col_count; d++) {
            sum += row[d] * row[d];
        }
        norms[i] = sum;
    });

    // Step 2: Compute A * A^T via GEMM
    // Data is row-major: A[row_count x col_count]
    // Fortran GEMM sees column-major: A_f[col_count x row_count]
    // We want C_rm[row_count x row_count] = A_rm * A_rm^T
    // In Fortran terms: C_f = A_f^T * A_f  =>  transa='t', transb='n'
    {
        const char transa = 't';
        const char transb = 'n';
        const std::int64_t m = row_count;
        const std::int64_t n = row_count;
        const std::int64_t k = col_count;
        const Float alpha = Float(1);
        const Float beta = Float(0);
        const std::int64_t lda = col_count;
        const std::int64_t ldb = col_count;
        const std::int64_t ldc = row_count;

        gemm_call(&transa,
                  &transb,
                  &m,
                  &n,
                  &k,
                  &alpha,
                  data,
                  &lda,
                  data,
                  &ldb,
                  &beta,
                  dist_matrix,
                  &ldc);
    }

    // Step 3: dist²[i,j] = norm[i] + norm[j] - 2 * dot(i,j), then sqrt
    // Parallelized over row blocks
    const std::int64_t block_size = 256;
    const bk::uniform_blocking blocking(row_count, block_size);
    const std::int64_t block_count = blocking.get_block_count();

    de::threader_for_int64(block_count, [&](std::int64_t b) {
        const std::int64_t i_begin = blocking.get_block_start_index(b);
        const std::int64_t i_end = blocking.get_block_end_index(b);

        for (std::int64_t i = i_begin; i < i_end; i++) {
            Float* row = dist_matrix + i * row_count;
            const Float ni = norms[i];
            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for (std::int64_t j = 0; j < row_count; j++) {
                Float d2 = ni + norms[j] - Float(2) * row[j];
                // Clamp to zero (numerical noise can make it slightly negative)
                if (d2 < Float(0))
                    d2 = Float(0);
                row[j] = static_cast<Float>(std::sqrt(static_cast<double>(d2)));
            }
            row[i] = Float(0); // Self-distance is exactly zero
        }
    });
}

/// Compute core distances from a precomputed distance matrix.
/// core_distance[i] = distance to the (min_samples)-th nearest neighbor (including self).
/// Uses nth_element for O(n) partial sort per row.
template <typename Float>
static void compute_core_distances_on_host(const Float* dist_matrix,
                                           Float* core_distances,
                                           std::int64_t row_count,
                                           std::int64_t min_samples) {
    ONEDAL_ASSERT(row_count > 0);
    ONEDAL_ASSERT(min_samples >= 1);
    ONEDAL_ASSERT(min_samples <= row_count);

    const std::int64_t target = min_samples - 1;

    de::threader_for_int64(row_count, [&](std::int64_t i) {
        // Copy row from distance matrix (avoid mutating the original)
        std::vector<Float> dists(row_count);
        const Float* row = dist_matrix + i * row_count;
        std::memcpy(dists.data(), row, row_count * sizeof(Float));

        std::int64_t t = (target >= row_count) ? row_count - 1 : target;
        std::nth_element(dists.begin(), dists.begin() + t, dists.end());
        core_distances[i] = dists[t];
    });
}

/// Build MST using Prim's algorithm with precomputed distance matrix.
/// Uses Mutual Reachability Distance: MRD(i,j) = max(core[i], core[j], dist[i,j])
template <typename Float>
static void build_mst_on_host(const Float* dist_matrix,
                              const Float* core_distances,
                              std::int32_t* mst_from,
                              std::int32_t* mst_to,
                              Float* mst_weights,
                              std::int64_t row_count) {
    ONEDAL_ASSERT(row_count > 1);

    const std::int64_t edge_count = row_count - 1;

    std::vector<Float> min_edge(row_count, std::numeric_limits<Float>::max());
    std::vector<std::int32_t> min_from(row_count, 0);
    std::vector<bool> in_mst(row_count, false);

    // Initialize from node 0
    in_mst[0] = true;
    const Float* row0 = dist_matrix;
    const Float core0 = core_distances[0];
    for (std::int64_t j = 1; j < row_count; j++) {
        const Float dist = row0[j];
        min_edge[j] = std::max({ core0, core_distances[j], dist });
    }

    for (std::int64_t e = 0; e < edge_count; e++) {
        // Find minimum edge to non-MST node
        std::int32_t best = -1;
        Float best_w = std::numeric_limits<Float>::max();
        for (std::int64_t j = 0; j < row_count; j++) {
            if (!in_mst[j] && min_edge[j] < best_w) {
                best_w = min_edge[j];
                best = static_cast<std::int32_t>(j);
            }
        }
        ONEDAL_ASSERT(best >= 0);

        mst_from[e] = min_from[best];
        mst_to[e] = best;
        mst_weights[e] = best_w;
        in_mst[best] = true;

        // Update min_edge for remaining nodes
        const Float* best_row = dist_matrix + static_cast<std::int64_t>(best) * row_count;
        const Float core_best = core_distances[best];
        for (std::int64_t j = 0; j < row_count; j++) {
            if (in_mst[j])
                continue;
            const Float dist = best_row[j];
            const Float mrd = std::max({ core_best, core_distances[j], dist });
            if (mrd < min_edge[j]) {
                min_edge[j] = mrd;
                min_from[j] = best;
            }
        }
    }
}

/// Legacy interface for GPU backend compatibility (computes distances on-the-fly)
template <typename Float>
static inline Float euclidean_dist(const Float* data,
                                   std::int64_t i,
                                   std::int64_t j,
                                   std::int64_t cols) {
    Float sum = Float(0);
    const Float* ri = data + i * cols;
    const Float* rj = data + j * cols;
    for (std::int64_t d = 0; d < cols; d++) {
        const Float v = ri[d] - rj[d];
        sum += v * v;
    }
    return static_cast<Float>(std::sqrt(static_cast<double>(sum)));
}

/// Legacy build_mst interface (for GPU backend that computes MRD on-the-fly)
template <typename Float>
static void build_mst_on_host(const Float* data,
                              const Float* core_distances,
                              std::int32_t* mst_from,
                              std::int32_t* mst_to,
                              Float* mst_weights,
                              std::int64_t row_count,
                              std::int64_t col_count) {
    ONEDAL_ASSERT(row_count > 1);

    const std::int64_t edge_count = row_count - 1;

    std::vector<Float> min_edge(row_count, std::numeric_limits<Float>::max());
    std::vector<std::int32_t> min_from(row_count, 0);
    std::vector<bool> in_mst(row_count, false);

    in_mst[0] = true;
    for (std::int64_t j = 1; j < row_count; j++) {
        const Float dist = euclidean_dist(data, 0, j, col_count);
        min_edge[j] = std::max({ core_distances[0], core_distances[j], dist });
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

        mst_from[e] = min_from[best];
        mst_to[e] = best;
        mst_weights[e] = best_w;
        in_mst[best] = true;

        for (std::int64_t j = 0; j < row_count; j++) {
            if (in_mst[j])
                continue;
            const Float dist = euclidean_dist(data, best, j, col_count);
            const Float mrd = std::max({ core_distances[best], core_distances[j], dist });
            if (mrd < min_edge[j]) {
                min_edge[j] = mrd;
                min_from[j] = best;
            }
        }
    }
}

template <typename Float>
static void sort_mst_by_weight_on_host(std::int32_t* mst_from,
                                       std::int32_t* mst_to,
                                       Float* mst_weights,
                                       std::int64_t edge_count) {
    ONEDAL_ASSERT(edge_count > 0);

    std::vector<std::int64_t> order(edge_count);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::int64_t a, std::int64_t b) {
        if (mst_weights[a] != mst_weights[b])
            return mst_weights[a] < mst_weights[b];
        const auto a_lo = std::min(mst_from[a], mst_to[a]);
        const auto a_hi = std::max(mst_from[a], mst_to[a]);
        const auto b_lo = std::min(mst_from[b], mst_to[b]);
        const auto b_hi = std::max(mst_from[b], mst_to[b]);
        if (a_lo != b_lo)
            return a_lo < b_lo;
        return a_hi < b_hi;
    });

    std::vector<std::int32_t> sorted_from(edge_count);
    std::vector<std::int32_t> sorted_to(edge_count);
    std::vector<Float> sorted_weights(edge_count);
    for (std::int64_t i = 0; i < edge_count; i++) {
        sorted_from[i] = mst_from[order[i]];
        sorted_to[i] = mst_to[order[i]];
        sorted_weights[i] = mst_weights[order[i]];
    }

    std::copy(sorted_from.begin(), sorted_from.end(), mst_from);
    std::copy(sorted_to.begin(), sorted_to.end(), mst_to);
    std::copy(sorted_weights.begin(), sorted_weights.end(), mst_weights);
}

/// Extract flat cluster labels from sorted MST edges.
/// Implements the full HDBSCAN pipeline: dendrogram -> condensed tree -> cluster selection -> labels.
/// cluster_selection: 0 = EOM (default), 1 = leaf
/// Returns the number of clusters found. Noise points are labeled -1.
template <typename Float>
static std::int64_t extract_clusters_from_mst(const std::int32_t* from_ptr,
                                              const std::int32_t* to_ptr,
                                              const Float* weights_ptr,
                                              std::int32_t* responses,
                                              std::int64_t row_count,
                                              std::int64_t min_cluster_size,
                                              int cluster_selection = 0,
                                              bool allow_single_cluster = false) {
    ONEDAL_ASSERT(row_count > 0);
    ONEDAL_ASSERT(min_cluster_size >= 2);

    const std::int64_t edge_count = row_count - 1;
    const std::int64_t n_dendro_nodes = row_count - 1;
    const std::int64_t total_nodes = 2 * row_count - 1;

    // --- Phase 1: Build Kruskal dendrogram via union-find ---

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

    struct dendro_node {
        std::int32_t left;
        std::int32_t right;
        Float weight;
        std::int32_t size;
    };
    std::vector<dendro_node> dendro(n_dendro_nodes);

    std::vector<std::int32_t> comp_to_node(row_count);
    std::iota(comp_to_node.begin(), comp_to_node.end(), 0);

    for (std::int64_t e = 0; e < edge_count; e++) {
        const std::int32_t ru = uf_find(from_ptr[e]);
        const std::int32_t rv = uf_find(to_ptr[e]);
        if (ru == rv)
            continue;

        const std::int32_t left_node = comp_to_node[ru];
        const std::int32_t right_node = comp_to_node[rv];
        const std::int32_t new_size = comp_size[ru] + comp_size[rv];
        const std::int32_t node_id = static_cast<std::int32_t>(row_count + e);

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

    // --- Phase 2: Build condensed tree ---

    std::vector<std::int32_t> node_size(total_nodes, 0);
    for (std::int64_t i = 0; i < row_count; i++)
        node_size[i] = 1;
    for (std::int64_t e = 0; e < n_dendro_nodes; e++)
        node_size[row_count + e] = dendro[e].size;

    std::vector<std::int32_t> left_child(total_nodes, -1);
    std::vector<std::int32_t> right_child(total_nodes, -1);
    std::vector<Float> node_weight(total_nodes, Float(0));
    for (std::int64_t e = 0; e < n_dendro_nodes; e++) {
        const std::int64_t nid = row_count + e;
        left_child[nid] = dendro[e].left;
        right_child[nid] = dendro[e].right;
        node_weight[nid] = dendro[e].weight;
    }

    struct condensed_edge {
        std::int32_t parent;
        std::int32_t child;
        Float lambda_val;
        std::int32_t child_size;
    };
    std::vector<condensed_edge> condensed;
    condensed.reserve(2 * row_count);

    // Find the root of the dendrogram
    std::int32_t root = -1;
    for (std::int64_t e = n_dendro_nodes - 1; e >= 0; e--) {
        if (dendro[e].size > 0) {
            root = static_cast<std::int32_t>(row_count + e);
            break;
        }
    }
    if (root < 0) {
        std::fill(responses, responses + row_count, -1);
        return 0;
    }

    std::int32_t next_cid = static_cast<std::int32_t>(row_count);
    std::vector<std::int32_t> dendro_to_cluster(total_nodes, -1);
    dendro_to_cluster[root] = next_cid++;

    // Collect all leaf indices under a dendrogram subtree
    std::function<void(std::int32_t, std::vector<std::int32_t>&)> collect_leaves;
    collect_leaves = [&](std::int32_t nid, std::vector<std::int32_t>& out) {
        if (nid < row_count) {
            out.push_back(nid);
            return;
        }
        collect_leaves(left_child[nid], out);
        collect_leaves(right_child[nid], out);
    };

    struct stack_item {
        std::int32_t node;
        std::int32_t cluster;
    };
    std::vector<stack_item> stack;
    stack.push_back({ root, dendro_to_cluster[root] });

    while (!stack.empty()) {
        auto [nid, parent_cid] = stack.back();
        stack.pop_back();

        if (nid < row_count)
            continue;

        const std::int32_t lc = left_child[nid];
        const std::int32_t rc = right_child[nid];
        if (lc < 0 || rc < 0)
            continue;

        const std::int32_t ls = node_size[lc];
        const std::int32_t rs = node_size[rc];
        const Float lambda = (node_weight[nid] > Float(0)) ? Float(1) / node_weight[nid]
                                                           : std::numeric_limits<Float>::max();

        const bool l_big = ls >= min_cluster_size;
        const bool r_big = rs >= min_cluster_size;

        if (l_big && r_big) {
            // Both children are large: split into two new clusters
            const std::int32_t lcid = next_cid++;
            const std::int32_t rcid = next_cid++;
            dendro_to_cluster[lc] = lcid;
            dendro_to_cluster[rc] = rcid;
            condensed.push_back({ parent_cid, lcid, lambda, ls });
            condensed.push_back({ parent_cid, rcid, lambda, rs });
            stack.push_back({ lc, lcid });
            stack.push_back({ rc, rcid });
        }
        else if (l_big) {
            // Left is large: continues the parent cluster, right falls out
            dendro_to_cluster[lc] = parent_cid;
            std::vector<std::int32_t> fallen;
            collect_leaves(rc, fallen);
            for (auto pt : fallen)
                condensed.push_back({ parent_cid, pt, lambda, 1 });
            stack.push_back({ lc, parent_cid });
        }
        else if (r_big) {
            // Right is large: continues the parent cluster, left falls out
            dendro_to_cluster[rc] = parent_cid;
            std::vector<std::int32_t> fallen;
            collect_leaves(lc, fallen);
            for (auto pt : fallen)
                condensed.push_back({ parent_cid, pt, lambda, 1 });
            stack.push_back({ rc, parent_cid });
        }
        else {
            // Both children too small: all points fall out as noise
            std::vector<std::int32_t> fallen;
            collect_leaves(lc, fallen);
            collect_leaves(rc, fallen);
            for (auto pt : fallen)
                condensed.push_back({ parent_cid, pt, lambda, 1 });
        }
    }

    // --- Phase 3: Compute stability and select clusters via EOM ---

    const std::int32_t n_clusters = next_cid;
    const std::int32_t root_cid = static_cast<std::int32_t>(row_count);

    std::vector<Float> stability(n_clusters, Float(0));
    std::vector<Float> lambda_birth(n_clusters, Float(0));
    std::vector<bool> is_leaf_cluster(n_clusters, true);
    std::vector<std::vector<std::int32_t>> child_clusters(n_clusters);
    std::vector<std::int32_t> cluster_size(n_clusters, 0);
    cluster_size[root_cid] = static_cast<std::int32_t>(row_count);

    for (const auto& e : condensed) {
        if (e.child >= row_count) {
            lambda_birth[e.child] = e.lambda_val;
            is_leaf_cluster[e.parent] = false;
            child_clusters[e.parent].push_back(e.child);
            cluster_size[e.child] = e.child_size;
        }
    }

    for (const auto& e : condensed) {
        const Float birth = lambda_birth[e.parent];
        const Float contrib = (e.lambda_val - birth) * static_cast<Float>(e.child_size);
        if (contrib > Float(0)) {
            stability[e.parent] += contrib;
        }
    }

    std::vector<bool> is_selected(n_clusters, true);

    // Clusters smaller than min_cluster_size cannot be selected
    for (std::int32_t c = root_cid; c < n_clusters; c++) {
        if (cluster_size[c] < min_cluster_size) {
            is_selected[c] = false;
        }
    }

    if (cluster_selection == 1) {
        // Leaf selection: select all leaf clusters in the condensed tree
        for (std::int32_t c = root_cid; c < n_clusters; c++) {
            is_selected[c] = is_leaf_cluster[c] && (cluster_size[c] >= min_cluster_size);
        }
    }
    else {
        // Bottom-up EOM: compare parent stability vs sum of children
        for (std::int32_t c = n_clusters - 1; c >= root_cid; c--) {
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
                // Parent wins: deselect all descendants
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
    }

    // Phase 3b: allow_single_cluster enforcement.
    if (!allow_single_cluster) {
        std::int32_t sel_count = 0;
        for (std::int32_t c = root_cid; c < n_clusters; c++) {
            if (is_selected[c])
                sel_count++;
        }
        if (sel_count == 1 && is_selected[root_cid] && !is_leaf_cluster[root_cid]) {
            is_selected[root_cid] = false;
            for (auto cc : child_clusters[root_cid])
                is_selected[cc] = true;
        }
    }

    // --- Phase 4: Label each point ---

    std::int32_t label_counter = 0;
    std::vector<std::int32_t> cluster_label(n_clusters, -1);
    for (std::int32_t c = root_cid; c < n_clusters; c++) {
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

    // Label points that fell out of a cluster: walk up the cluster tree
    for (std::int64_t i = 0; i < row_count; i++) {
        responses[i] = -1;
        std::int32_t c = point_fell_from[i];
        while (c >= root_cid && c < n_clusters) {
            if (is_selected[c]) {
                responses[i] = cluster_label[c];
                break;
            }
            c = cluster_parent[c];
        }
    }

    // Label points that were never ejected: walk up the dendrogram tree
    std::vector<std::int32_t> dendro_parent(total_nodes, -1);
    for (std::int64_t e = 0; e < n_dendro_nodes; e++) {
        const std::int64_t nid = row_count + e;
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
            const std::int32_t cid = dendro_to_cluster[nid];
            if (cid >= root_cid) {
                std::int32_t c = cid;
                while (c >= root_cid && c < n_clusters) {
                    if (is_selected[c]) {
                        responses[i] = cluster_label[c];
                        break;
                    }
                    c = cluster_parent[c];
                }
                break;
            }
            nid = dendro_parent[nid];
        }
    }

    return label_counter;
}

} // namespace oneapi::dal::hdbscan::backend
