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

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/backend/common.hpp"

#ifndef ONEDAL_DATA_PARALLEL
#include "src/externals/service_blas.h"
#endif

namespace oneapi::dal::hdbscan::backend {

namespace de = oneapi::dal::detail;
namespace bk = oneapi::dal::backend;

#ifndef ONEDAL_DATA_PARALLEL

template <typename Float>
static void compute_distance_matrix(const Float* data,
                                    Float* dist_matrix,
                                    std::int64_t row_count,
                                    std::int64_t col_count) {
    ONEDAL_ASSERT(row_count > 0);
    ONEDAL_ASSERT(col_count > 0);

    auto norms_arr = dal::array<Float>::empty(row_count);
    Float* norms = norms_arr.get_mutable_data();

    de::threader_for_int64(row_count, [&](std::int64_t i) {
        Float sum = Float(0);
        const Float* row = data + i * col_count;
        PRAGMA_OMP_SIMD_ARGS(reduction(+ : sum))
        for (std::int64_t d = 0; d < col_count; d++) {
            sum += row[d] * row[d];
        }
        norms[i] = sum;
    });

    {
        const char transa = 't';
        const char transb = 'n';
        const DAAL_INT m = static_cast<DAAL_INT>(row_count);
        const DAAL_INT n = static_cast<DAAL_INT>(row_count);
        const DAAL_INT k = static_cast<DAAL_INT>(col_count);
        const Float alpha_val = Float(1);
        const Float beta_val = Float(0);
        const DAAL_INT lda = static_cast<DAAL_INT>(col_count);
        const DAAL_INT ldb = static_cast<DAAL_INT>(col_count);
        const DAAL_INT ldc = static_cast<DAAL_INT>(row_count);

        daal::internal::BlasInst<Float, daal::internal::CpuType::sse2>::xgemm(&transa,
                                                                              &transb,
                                                                              &m,
                                                                              &n,
                                                                              &k,
                                                                              &alpha_val,
                                                                              data,
                                                                              &lda,
                                                                              data,
                                                                              &ldb,
                                                                              &beta_val,
                                                                              dist_matrix,
                                                                              &ldc);
    }

    const std::int64_t block_size = 256;
    const bk::uniform_blocking blocking(row_count, block_size);
    const std::int64_t block_count = blocking.get_block_count();

    de::threader_for_int64(block_count, [&](std::int64_t b) {
        const std::int64_t i_begin = blocking.get_block_start_index(b);
        const std::int64_t i_end = blocking.get_block_end_index(b);

        for (std::int64_t i = i_begin; i < i_end; i++) {
            Float* row = dist_matrix + i * row_count;
            const Float ni = norms[i];
            PRAGMA_VECTOR_ALWAYS
            for (std::int64_t j = 0; j < row_count; j++) {
                Float d2 = ni + norms[j] - Float(2) * row[j];
                if (d2 < Float(0))
                    d2 = Float(0);
                row[j] = static_cast<Float>(std::sqrt(static_cast<double>(d2)));
            }
            row[i] = Float(0);
        }
    });
}

#endif // !ONEDAL_DATA_PARALLEL

template <typename Float>
static void compute_core_distances(const Float* dist_matrix,
                                   Float* core_distances,
                                   std::int64_t row_count,
                                   std::int64_t min_samples) {
    ONEDAL_ASSERT(row_count > 0);
    ONEDAL_ASSERT(min_samples >= 1);
    ONEDAL_ASSERT(min_samples <= row_count);

    const std::int64_t target = min_samples - 1;
    const std::int64_t t = (target >= row_count) ? row_count - 1 : target;

    const std::int64_t num_threads = de::threader_get_max_threads();
    auto scratch = dal::array<Float>::empty(row_count * num_threads);
    Float* scratch_ptr = scratch.get_mutable_data();

    de::threader_for_int64(row_count, [&](std::int64_t i) {
        const std::int64_t tid = de::threader_get_current_thread_index();
        Float* dists = scratch_ptr + tid * row_count;
        const Float* row = dist_matrix + i * row_count;
        std::memcpy(dists, row, row_count * sizeof(Float));

        std::nth_element(dists, dists + t, dists + row_count);
        core_distances[i] = dists[t];
    });
}

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

template <typename Float>
static void build_mst(const Float* dist_matrix,
                      const Float* core_distances,
                      std::int32_t* mst_from,
                      std::int32_t* mst_to,
                      Float* mst_weights,
                      std::int64_t row_count) {
    ONEDAL_ASSERT(row_count > 1);

    const std::int64_t n = row_count;

    auto uf_parent_arr = dal::array<std::int32_t>::empty(n);
    auto uf_rank_arr = dal::array<std::int32_t>::zeros(n);
    auto comp_of_arr = dal::array<std::int32_t>::empty(n);
    std::int32_t* uf_parent = uf_parent_arr.get_mutable_data();
    std::int32_t* uf_rank = uf_rank_arr.get_mutable_data();
    std::int32_t* comp_of = comp_of_arr.get_mutable_data();
    for (std::int64_t i = 0; i < n; i++) {
        uf_parent[i] = static_cast<std::int32_t>(i);
        comp_of[i] = static_cast<std::int32_t>(i);
    }

    auto uf_find = [&](std::int32_t x) -> std::int32_t {
        while (uf_parent[x] != x) {
            uf_parent[x] = uf_parent[uf_parent[x]];
            x = uf_parent[x];
        }
        return x;
    };
    auto uf_union = [&](std::int32_t rx, std::int32_t ry) {
        if (uf_rank[rx] < uf_rank[ry])
            uf_parent[rx] = ry;
        else if (uf_rank[rx] > uf_rank[ry])
            uf_parent[ry] = rx;
        else {
            uf_parent[ry] = rx;
            uf_rank[rx]++;
        }
    };

    auto pt_best_mrd_arr = dal::array<Float>::empty(n);
    auto pt_best_idx_arr = dal::array<std::int32_t>::empty(n);
    auto comp_best_mrd_arr = dal::array<Float>::empty(n);
    auto comp_best_from_arr = dal::array<std::int32_t>::empty(n);
    auto comp_best_to_arr = dal::array<std::int32_t>::empty(n);
    Float* pt_best_mrd = pt_best_mrd_arr.get_mutable_data();
    std::int32_t* pt_best_idx = pt_best_idx_arr.get_mutable_data();
    Float* comp_best_mrd = comp_best_mrd_arr.get_mutable_data();
    std::int32_t* comp_best_from = comp_best_from_arr.get_mutable_data();
    std::int32_t* comp_best_to = comp_best_to_arr.get_mutable_data();

    std::int64_t edges_added = 0;
    std::int64_t num_components = n;

    while (num_components > 1) {
        for (std::int64_t i = 0; i < n; i++) {
            const std::int32_t my_comp = comp_of[i];
            const Float* row = dist_matrix + i * n;
            Float best_m = std::numeric_limits<Float>::max();
            std::int32_t best_j = -1;
            for (std::int64_t j = 0; j < n; j++) {
                if (comp_of[j] == my_comp)
                    continue;
                const Float mrd = std::max({ core_distances[i], core_distances[j], row[j] });
                if (mrd < best_m) {
                    best_m = mrd;
                    best_j = static_cast<std::int32_t>(j);
                }
            }
            pt_best_mrd[i] = best_m;
            pt_best_idx[i] = best_j;
        }

        for (std::int64_t c = 0; c < n; c++) {
            comp_best_mrd[c] = std::numeric_limits<Float>::max();
            comp_best_from[c] = -1;
            comp_best_to[c] = -1;
        }
        for (std::int64_t i = 0; i < n; i++) {
            if (pt_best_idx[i] < 0)
                continue;
            const std::int32_t c = comp_of[i];
            if (pt_best_mrd[i] < comp_best_mrd[c]) {
                comp_best_mrd[c] = pt_best_mrd[i];
                comp_best_from[c] = static_cast<std::int32_t>(i);
                comp_best_to[c] = pt_best_idx[i];
            }
        }

        std::int64_t added = 0;
        for (std::int64_t c = 0; c < n; c++) {
            if (comp_best_from[c] < 0)
                continue;
            const std::int32_t u = comp_best_from[c], v = comp_best_to[c];
            const std::int32_t ru = uf_find(u), rv = uf_find(v);
            if (ru == rv)
                continue;
            mst_from[edges_added] = u;
            mst_to[edges_added] = v;
            mst_weights[edges_added] = comp_best_mrd[c];
            edges_added++;
            added++;
            uf_union(ru, rv);
            num_components--;
        }
        if (added == 0)
            break;

        for (std::int64_t i = 0; i < n; i++)
            comp_of[i] = uf_find(static_cast<std::int32_t>(i));
    }
}

template <typename Float>
static void build_mst(const Float* data,
                      const Float* core_distances,
                      std::int32_t* mst_from,
                      std::int32_t* mst_to,
                      Float* mst_weights,
                      std::int64_t row_count,
                      std::int64_t col_count) {
    ONEDAL_ASSERT(row_count > 1);

    const std::int64_t n = row_count;

    auto uf_parent_arr = dal::array<std::int32_t>::empty(n);
    auto uf_rank_arr = dal::array<std::int32_t>::zeros(n);
    auto comp_of_arr = dal::array<std::int32_t>::empty(n);
    std::int32_t* uf_parent = uf_parent_arr.get_mutable_data();
    std::int32_t* uf_rank = uf_rank_arr.get_mutable_data();
    std::int32_t* comp_of = comp_of_arr.get_mutable_data();
    for (std::int64_t i = 0; i < n; i++) {
        uf_parent[i] = static_cast<std::int32_t>(i);
        comp_of[i] = static_cast<std::int32_t>(i);
    }

    auto uf_find = [&](std::int32_t x) -> std::int32_t {
        while (uf_parent[x] != x) {
            uf_parent[x] = uf_parent[uf_parent[x]];
            x = uf_parent[x];
        }
        return x;
    };
    auto uf_union = [&](std::int32_t rx, std::int32_t ry) {
        if (uf_rank[rx] < uf_rank[ry])
            uf_parent[rx] = ry;
        else if (uf_rank[rx] > uf_rank[ry])
            uf_parent[ry] = rx;
        else {
            uf_parent[ry] = rx;
            uf_rank[rx]++;
        }
    };

    auto pt_best_mrd_arr = dal::array<Float>::empty(n);
    auto pt_best_idx_arr = dal::array<std::int32_t>::empty(n);
    auto comp_best_mrd_arr = dal::array<Float>::empty(n);
    auto comp_best_from_arr = dal::array<std::int32_t>::empty(n);
    auto comp_best_to_arr = dal::array<std::int32_t>::empty(n);
    Float* pt_best_mrd = pt_best_mrd_arr.get_mutable_data();
    std::int32_t* pt_best_idx = pt_best_idx_arr.get_mutable_data();
    Float* comp_best_mrd = comp_best_mrd_arr.get_mutable_data();
    std::int32_t* comp_best_from = comp_best_from_arr.get_mutable_data();
    std::int32_t* comp_best_to = comp_best_to_arr.get_mutable_data();

    std::int64_t edges_added = 0;
    std::int64_t num_components = n;

    while (num_components > 1) {
        for (std::int64_t i = 0; i < n; i++) {
            const std::int32_t my_comp = comp_of[i];
            Float best_m = std::numeric_limits<Float>::max();
            std::int32_t best_j = -1;
            for (std::int64_t j = 0; j < n; j++) {
                if (comp_of[j] == my_comp)
                    continue;
                const Float dist = euclidean_dist(data, i, j, col_count);
                const Float mrd = std::max({ core_distances[i], core_distances[j], dist });
                if (mrd < best_m) {
                    best_m = mrd;
                    best_j = static_cast<std::int32_t>(j);
                }
            }
            pt_best_mrd[i] = best_m;
            pt_best_idx[i] = best_j;
        }

        for (std::int64_t c = 0; c < n; c++) {
            comp_best_mrd[c] = std::numeric_limits<Float>::max();
            comp_best_from[c] = -1;
            comp_best_to[c] = -1;
        }
        for (std::int64_t i = 0; i < n; i++) {
            if (pt_best_idx[i] < 0)
                continue;
            const std::int32_t c = comp_of[i];
            if (pt_best_mrd[i] < comp_best_mrd[c]) {
                comp_best_mrd[c] = pt_best_mrd[i];
                comp_best_from[c] = static_cast<std::int32_t>(i);
                comp_best_to[c] = pt_best_idx[i];
            }
        }

        std::int64_t added = 0;
        for (std::int64_t c = 0; c < n; c++) {
            if (comp_best_from[c] < 0)
                continue;
            const std::int32_t u = comp_best_from[c], v = comp_best_to[c];
            const std::int32_t ru = uf_find(u), rv = uf_find(v);
            if (ru == rv)
                continue;
            mst_from[edges_added] = u;
            mst_to[edges_added] = v;
            mst_weights[edges_added] = comp_best_mrd[c];
            edges_added++;
            added++;
            uf_union(ru, rv);
            num_components--;
        }
        if (added == 0)
            break;

        for (std::int64_t i = 0; i < n; i++)
            comp_of[i] = uf_find(static_cast<std::int32_t>(i));
    }
}

template <typename Float>
static void sort_mst_by_weight(std::int32_t* mst_from,
                               std::int32_t* mst_to,
                               Float* mst_weights,
                               std::int64_t edge_count) {
    ONEDAL_ASSERT(edge_count > 0);

    auto order_arr = dal::array<std::int64_t>::empty(edge_count);
    std::int64_t* order = order_arr.get_mutable_data();
    for (std::int64_t i = 0; i < edge_count; i++)
        order[i] = i;

    std::sort(order, order + edge_count, [&](std::int64_t a, std::int64_t b) {
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

    auto sorted_from_arr = dal::array<std::int32_t>::empty(edge_count);
    auto sorted_to_arr = dal::array<std::int32_t>::empty(edge_count);
    auto sorted_weights_arr = dal::array<Float>::empty(edge_count);
    std::int32_t* sorted_from = sorted_from_arr.get_mutable_data();
    std::int32_t* sorted_to = sorted_to_arr.get_mutable_data();
    Float* sorted_weights = sorted_weights_arr.get_mutable_data();

    for (std::int64_t i = 0; i < edge_count; i++) {
        sorted_from[i] = mst_from[order[i]];
        sorted_to[i] = mst_to[order[i]];
        sorted_weights[i] = mst_weights[order[i]];
    }

    std::copy(sorted_from, sorted_from + edge_count, mst_from);
    std::copy(sorted_to, sorted_to + edge_count, mst_to);
    std::copy(sorted_weights, sorted_weights + edge_count, mst_weights);
}

template <typename Float>
static std::int64_t extract_clusters_from_mst(const std::int32_t* from_ptr,
                                              const std::int32_t* to_ptr,
                                              const Float* weights_ptr,
                                              std::int32_t* responses,
                                              std::int64_t row_count,
                                              std::int64_t min_cluster_size,
                                              int cluster_selection = 0,
                                              bool allow_single_cluster = false,
                                              double cluster_selection_epsilon = 0.0,
                                              std::int64_t max_cluster_size = 0) {
    ONEDAL_ASSERT(row_count > 0);
    ONEDAL_ASSERT(min_cluster_size >= 2);

    const std::int64_t edge_count = row_count - 1;
    const std::int64_t n_dendro_nodes = row_count - 1;
    const std::int64_t total_nodes = 2 * row_count - 1;

    // --- Phase 1: Build Kruskal dendrogram via union-find ---

    auto uf_parent_arr = dal::array<std::int32_t>::empty(row_count);
    auto comp_size_arr = dal::array<std::int32_t>::empty(row_count);
    std::int32_t* uf_parent = uf_parent_arr.get_mutable_data();
    std::int32_t* comp_size = comp_size_arr.get_mutable_data();
    for (std::int64_t i = 0; i < row_count; i++) {
        uf_parent[i] = static_cast<std::int32_t>(i);
        comp_size[i] = 1;
    }

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

    auto comp_to_node_arr = dal::array<std::int32_t>::empty(row_count);
    std::int32_t* comp_to_node = comp_to_node_arr.get_mutable_data();
    for (std::int64_t i = 0; i < row_count; i++)
        comp_to_node[i] = static_cast<std::int32_t>(i);

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

    auto node_size_arr = dal::array<std::int32_t>::zeros(total_nodes);
    auto left_child_arr = dal::array<std::int32_t>::empty(total_nodes);
    auto right_child_arr = dal::array<std::int32_t>::empty(total_nodes);
    auto node_weight_arr = dal::array<Float>::zeros(total_nodes);
    std::int32_t* node_size = node_size_arr.get_mutable_data();
    std::int32_t* left_child = left_child_arr.get_mutable_data();
    std::int32_t* right_child = right_child_arr.get_mutable_data();
    Float* node_weight = node_weight_arr.get_mutable_data();

    for (std::int64_t i = 0; i < row_count; i++)
        node_size[i] = 1;
    for (std::int64_t i = 0; i < total_nodes; i++) {
        left_child[i] = -1;
        right_child[i] = -1;
    }
    for (std::int64_t e = 0; e < n_dendro_nodes; e++) {
        node_size[row_count + e] = dendro[e].size;
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
    auto dendro_to_cluster_arr = dal::array<std::int32_t>::empty(total_nodes);
    std::int32_t* dendro_to_cluster = dendro_to_cluster_arr.get_mutable_data();
    std::fill(dendro_to_cluster, dendro_to_cluster + total_nodes, std::int32_t(-1));
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

    // --- Phase 3: Compute stability and select clusters via EOM ---

    const std::int32_t n_clusters = next_cid;
    const std::int32_t root_cid = static_cast<std::int32_t>(row_count);

    auto stability_arr = dal::array<Float>::zeros(n_clusters);
    auto lambda_birth_arr = dal::array<Float>::zeros(n_clusters);
    Float* stability = stability_arr.get_mutable_data();
    Float* lambda_birth = lambda_birth_arr.get_mutable_data();

    std::vector<bool> is_leaf_cluster(n_clusters, true);
    std::vector<std::vector<std::int32_t>> child_clusters(n_clusters);
    auto cluster_size_arr = dal::array<std::int32_t>::zeros(n_clusters);
    std::int32_t* cluster_size = cluster_size_arr.get_mutable_data();
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

    for (std::int32_t c = root_cid; c < n_clusters; c++) {
        if (cluster_size[c] < min_cluster_size) {
            is_selected[c] = false;
        }
    }

    const std::int32_t mcs_max = (max_cluster_size > 0)
                                     ? static_cast<std::int32_t>(max_cluster_size)
                                     : std::numeric_limits<std::int32_t>::max();

    if (cluster_selection == 1) {
        for (std::int32_t c = root_cid; c < n_clusters; c++) {
            is_selected[c] = is_leaf_cluster[c] && (cluster_size[c] >= min_cluster_size);
        }
    }
    else {
        for (std::int32_t c = n_clusters - 1; c >= root_cid; c--) {
            if (is_leaf_cluster[c])
                continue;

            Float child_sum = Float(0);
            for (auto cc : child_clusters[c])
                child_sum += stability[cc];

            const Float parent_stab = (cluster_size[c] > mcs_max) ? Float(0) : stability[c];

            if (child_sum > parent_stab) {
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
    }

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

    auto cluster_parent_arr = dal::array<std::int32_t>::empty(n_clusters);
    std::int32_t* cluster_parent = cluster_parent_arr.get_mutable_data();
    std::fill(cluster_parent, cluster_parent + n_clusters, std::int32_t(-1));
    for (const auto& e : condensed) {
        if (e.child >= row_count)
            cluster_parent[e.child] = e.parent;
    }

    if (cluster_selection_epsilon > 0.0) {
        const Float eps = static_cast<Float>(cluster_selection_epsilon);
        bool changed = true;
        while (changed) {
            changed = false;
            for (std::int32_t c = root_cid + 1; c < n_clusters; c++) {
                if (!is_selected[c])
                    continue;
                const Float birth_dist =
                    (lambda_birth[c] > Float(0)) ? Float(1) / lambda_birth[c] : Float(0);
                if (birth_dist < eps) {
                    const std::int32_t parent = cluster_parent[c];
                    if (parent >= root_cid && parent < n_clusters) {
                        is_selected[c] = false;
                        is_selected[parent] = true;
                        changed = true;
                    }
                }
            }
        }
    }

    // --- Phase 4: Label each point ---

    std::int32_t label_counter = 0;
    auto cluster_label_arr = dal::array<std::int32_t>::empty(n_clusters);
    std::int32_t* cluster_label = cluster_label_arr.get_mutable_data();
    std::fill(cluster_label, cluster_label + n_clusters, std::int32_t(-1));
    for (std::int32_t c = root_cid; c < n_clusters; c++) {
        if (is_selected[c])
            cluster_label[c] = label_counter++;
    }

    auto point_fell_from_arr = dal::array<std::int32_t>::empty(row_count);
    std::int32_t* point_fell_from = point_fell_from_arr.get_mutable_data();
    std::fill(point_fell_from, point_fell_from + row_count, std::int32_t(-1));
    for (const auto& e : condensed) {
        if (e.child < row_count)
            point_fell_from[e.child] = e.parent;
    }

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

    auto dendro_parent_arr = dal::array<std::int32_t>::empty(total_nodes);
    std::int32_t* dendro_parent = dendro_parent_arr.get_mutable_data();
    std::fill(dendro_parent, dendro_parent + total_nodes, std::int32_t(-1));
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

template <typename Float>
static void compute_centroids(const Float* data,
                              const std::int32_t* labels,
                              std::int64_t row_count,
                              std::int64_t col_count,
                              std::int64_t cluster_count,
                              Float* centroids) {
    ONEDAL_ASSERT(cluster_count > 0);

    auto counts_arr = dal::array<std::int64_t>::zeros(cluster_count);
    std::int64_t* counts = counts_arr.get_mutable_data();
    std::memset(centroids, 0, cluster_count * col_count * sizeof(Float));

    for (std::int64_t i = 0; i < row_count; i++) {
        const std::int32_t label = labels[i];
        if (label < 0 || label >= cluster_count)
            continue;
        counts[label]++;
        Float* row_out = centroids + label * col_count;
        const Float* row_in = data + i * col_count;
        for (std::int64_t d = 0; d < col_count; d++) {
            row_out[d] += row_in[d];
        }
    }

    for (std::int64_t k = 0; k < cluster_count; k++) {
        if (counts[k] == 0)
            continue;
        Float* row = centroids + k * col_count;
        const Float inv = Float(1) / static_cast<Float>(counts[k]);
        for (std::int64_t d = 0; d < col_count; d++) {
            row[d] *= inv;
        }
    }
}

template <typename Float>
static void compute_medoids(const Float* data,
                            const std::int32_t* labels,
                            std::int64_t row_count,
                            std::int64_t col_count,
                            std::int64_t cluster_count,
                            const Float* centroids,
                            Float* medoids) {
    ONEDAL_ASSERT(cluster_count > 0);

    auto best_dist_arr = dal::array<Float>::empty(cluster_count);
    auto best_idx_arr = dal::array<std::int64_t>::empty(cluster_count);
    Float* best_dist = best_dist_arr.get_mutable_data();
    std::int64_t* best_idx = best_idx_arr.get_mutable_data();
    for (std::int64_t k = 0; k < cluster_count; k++) {
        best_dist[k] = std::numeric_limits<Float>::max();
        best_idx[k] = -1;
    }

    for (std::int64_t i = 0; i < row_count; i++) {
        const std::int32_t label = labels[i];
        if (label < 0 || label >= cluster_count)
            continue;
        const Float* pt = data + i * col_count;
        const Float* center = centroids + label * col_count;
        Float dist = Float(0);
        for (std::int64_t d = 0; d < col_count; d++) {
            const Float diff = pt[d] - center[d];
            dist += diff * diff;
        }
        if (dist < best_dist[label]) {
            best_dist[label] = dist;
            best_idx[label] = i;
        }
    }

    for (std::int64_t k = 0; k < cluster_count; k++) {
        Float* row = medoids + k * col_count;
        if (best_idx[k] >= 0) {
            std::memcpy(row, data + best_idx[k] * col_count, col_count * sizeof(Float));
        }
        else {
            std::memset(row, 0, col_count * sizeof(Float));
        }
    }
}

} // namespace oneapi::dal::hdbscan::backend
