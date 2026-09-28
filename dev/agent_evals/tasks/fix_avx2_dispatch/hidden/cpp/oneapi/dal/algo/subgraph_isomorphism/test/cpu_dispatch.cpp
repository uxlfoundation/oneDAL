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

// Subgraph isomorphism must give the same answer on every CPU code path. The kernel is
// run with the dispatch restricted to a given ISA through host_policy, so the AVX2 and
// SSE2 paths are exercised on any x86 host that supports them (AVX-512 servers included).
// The avx2 code path only exists when the library is built with it (e.g. --cpu=all).

#include <memory>
#include <utility>
#include <vector>

#include "oneapi/dal/algo/subgraph_isomorphism/graph_matching.hpp"
#include "oneapi/dal/backend/dispatcher_cpu.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::algo::subgraph_isomorphism::test {

namespace si = dal::preview::subgraph_isomorphism;
using graph_t = dal::preview::undirected_adjacency_vector_graph<std::int32_t>;
using cpu_ext = dal::detail::cpu_extension;

/// Undirected graph from an adjacency predicate on vertices [0, n)
template <typename Adjacent>
graph_t make_graph(std::int32_t n, Adjacent&& adjacent) {
    std::vector<std::vector<std::int32_t>> nbrs(n);
    std::int64_t edge_count = 0;
    for (std::int32_t u = 0; u < n; ++u) {
        for (std::int32_t v = 0; v < n; ++v) {
            if (u != v && adjacent(u, v)) {
                nbrs[u].push_back(v);
                if (u < v)
                    ++edge_count;
            }
        }
    }
    graph_t g;
    auto& impl = oneapi::dal::detail::get_impl(g);
    auto& va = impl._vertex_allocator;
    auto& ea = impl._edge_allocator;
    using i32_traits = std::allocator_traits<std::allocator<char>>::rebind_traits<std::int32_t>;
    using i64_traits = std::allocator_traits<std::allocator<char>>::rebind_traits<std::int64_t>;
    const std::int64_t cols_count = 2 * edge_count;
    const std::int64_t rows_count = n + 1;
    std::int32_t* degrees = i32_traits::allocate(va, n);
    std::int32_t* cols = i32_traits::allocate(va, cols_count);
    std::int64_t* rows = i64_traits::allocate(ea, rows_count);
    std::int32_t* rows_vertex = i32_traits::allocate(va, rows_count);
    std::int64_t pos = 0;
    for (std::int32_t u = 0; u < n; ++u) {
        rows[u] = pos;
        rows_vertex[u] = static_cast<std::int32_t>(pos);
        degrees[u] = static_cast<std::int32_t>(nbrs[u].size());
        for (auto v : nbrs[u])
            cols[pos++] = v;
    }
    rows[n] = pos;
    rows_vertex[n] = static_cast<std::int32_t>(pos);
    impl.set_topology(n, edge_count, rows, cols, cols_count, degrees);
    impl.get_topology()._rows_vertex =
        oneapi::dal::preview::detail::container<std::int32_t>::wrap(rows_vertex, rows_count);
    return g;
}

graph_t complete_graph(std::int32_t n) {
    return make_graph(n, [](std::int32_t, std::int32_t) {
        return true;
    });
}

/// Circulant graph: u ~ v iff their cyclic distance is in [1, k]
graph_t circulant_graph(std::int32_t n, std::int32_t k) {
    return make_graph(n, [=](std::int32_t u, std::int32_t v) {
        std::int32_t d = u > v ? u - v : v - u;
        d = std::min(d, n - d);
        return d >= 1 && d <= k;
    });
}

graph_t path_graph(std::int32_t n) {
    return make_graph(n, [](std::int32_t u, std::int32_t v) {
        return u - v == 1 || v - u == 1;
    });
}

std::int64_t match_count(const graph_t& target,
                         const graph_t& pattern,
                         si::kind kind,
                         const cpu_ext* forced) {
    const auto desc =
        si::descriptor<float, si::method::by_default, si::task::by_default, std::allocator<char>>(
            std::allocator<char>())
            .set_kind(kind)
            .set_max_match_count(0)
            .set_semantic_match(false);
    if (!forced) {
        return dal::preview::graph_matching(desc, target, pattern).get_match_count();
    }
    dal::detail::host_policy policy;
    policy.set_enabled_cpu_extensions(*forced);
    using ops_t = si::detail::graph_matching_ops<std::decay_t<decltype(desc)>, graph_t>;
    typename ops_t::input_t input{ target, pattern };
    return ops_t()(policy, desc, input).get_match_count();
}

void check_all_paths(const graph_t& target,
                     const graph_t& pattern,
                     si::kind kind,
                     std::int64_t expected) {
    const cpu_ext sse2 = cpu_ext::sse2;
    const cpu_ext avx2 = cpu_ext::avx2;
    CAPTURE(expected);
    REQUIRE(match_count(target, pattern, kind, nullptr) == expected);
    REQUIRE(match_count(target, pattern, kind, &sse2) == expected);
    REQUIRE(match_count(target, pattern, kind, &avx2) == expected);
}

TEST("subgraph isomorphism: avx2 code path is built", "[subgraph_isomorphism][cpu]") {
#if defined(TARGET_X86_64) && !defined(ONEDAL_CPU_DISPATCH_AVX2)
    FAIL("library built without the avx2 code path; build with --cpu=all");
#endif
    SUCCEED();
}

TEST("subgraph isomorphism: same match count on every CPU path, complete target",
     "[subgraph_isomorphism][cpu]") {
    const auto target = complete_graph(70);
    const auto triangle = complete_graph(3);
    // every ordered triple of distinct vertices is an embedding of a triangle into K_70
    check_all_paths(target, triangle, si::kind::non_induced, 70LL * 69 * 68);
    check_all_paths(target, triangle, si::kind::induced, 70LL * 69 * 68);
}

TEST("subgraph isomorphism: same match count on every CPU path, circulant target",
     "[subgraph_isomorphism][cpu]") {
    // C_96(1..6): 6-regular each side, 576 edges, density ~0.13 so the bit representation is used
    const auto target = circulant_graph(96, 6);
    const auto p3 = path_graph(3);
    // non-induced ordered paths u-v-w: 96 centers * 12 * 11
    check_all_paths(target, p3, si::kind::non_induced, 96LL * 12 * 11);
    const std::int64_t induced = match_count(target, p3, si::kind::induced, nullptr);
    REQUIRE(induced > 0);
    check_all_paths(target, p3, si::kind::induced, induced);
}

} // namespace oneapi::dal::algo::subgraph_isomorphism::test
