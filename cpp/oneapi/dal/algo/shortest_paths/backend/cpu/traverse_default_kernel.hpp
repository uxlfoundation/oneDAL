/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/algo/shortest_paths/backend/cpu/service.hpp"
#include "oneapi/dal/algo/shortest_paths/common.hpp"
#include "oneapi/dal/algo/shortest_paths/traverse_types.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/graph/detail/container.hpp"

namespace oneapi::dal::preview::shortest_paths::backend {
using namespace oneapi::dal::preview::detail;
using namespace oneapi::dal::preview::backend;

template <typename Vertex, typename EdgeValue, typename BinsVector>
inline void update_bins(const Vertex& v,
                        const EdgeValue& new_dist,
                        const EdgeValue& delta,
                        BinsVector& local_bins) {
    ONEDAL_ASSERT(new_dist > 0);
    ONEDAL_ASSERT(delta > 0);
    ONEDAL_ASSERT(new_dist / delta <= std::numeric_limits<EdgeValue>::max());
    const std::int64_t dest_bin = static_cast<std::int64_t>(new_dist / delta);
    ONEDAL_ASSERT(dest_bin >= 0);
    if (dest_bin >= local_bins.size()) {
        local_bins.resize(dest_bin + 1);
    }
    local_bins[dest_bin].push_back(v);
}

template <typename Mode>
struct relax_edges {};

template <>
struct relax_edges<mode::distances> {
    template <typename Topology, typename EdgeValue, typename BinsVector, typename DataToRelax>
    inline void operator()(const Topology& t,
                           const EdgeValue* vals,
                           typename Topology::vertex_type u,
                           EdgeValue delta,
                           DataToRelax& dist,
                           BinsVector& local_bins) {
        for (std::int64_t v_ = t._rows_ptr[u]; v_ < t._rows_ptr[u + 1]; v_++) {
            const auto v = t._cols_ptr[v_];
            const auto v_w = vals[v_];
            EdgeValue old_dist = dist[v];
            const EdgeValue new_dist = dist[u] + v_w;
            while (new_dist < old_dist) {
                if (dist.compare_exchange_strong(v, old_dist, new_dist)) {
                    update_bins(v, new_dist, delta, local_bins);
                    break;
                }
                old_dist = dist[v];
            }
        }
    }
};

template <>
struct relax_edges<mode::distances_predecessors> {
    template <typename Topology, typename EdgeValue, typename BinsVector, typename DataToRelax>
    inline void operator()(const Topology& t,
                           const EdgeValue* vals,
                           typename Topology::vertex_type u,
                           EdgeValue delta,
                           DataToRelax& dp,
                           BinsVector& local_bins) {
        for (std::int64_t v_ = t._rows_ptr[u]; v_ < t._rows_ptr[u + 1]; v_++) {
            const auto v = t._cols_ptr[v_];
            const auto v_w = vals[v_];
            auto old_dp = dp[v];
            EdgeValue old_dist = old_dp.dist;
            const EdgeValue new_dist = dp[u].dist + v_w;
            while (new_dist < old_dist) {
                if (dp.compare_exchange_strong(
                        v,
                        old_dp,
                        dist_pred<EdgeValue, typename Topology::vertex_type>(new_dist, u))) {
                    update_bins(v, new_dist, delta, local_bins);
                    break;
                }
                old_dp = dp[v];
                old_dist = old_dp.dist;
            }
        }
    }
};

template <typename BinsVector>
inline void find_next_bin_index(std::int64_t& curr_bin_index, const BinsVector& local_bins) {
    const std::int64_t max_bin_count = std::numeric_limits<std::int64_t>::max() / 2;
    std::atomic<std::int64_t> min_nonempty_bucket_index;
    min_nonempty_bucket_index.store(max_bin_count);

    dal::detail::threader_for(local_bins.size(), 1, [&](std::int64_t i) {
        const auto& local_bin = local_bins[i];
        for (std::int64_t bucket_index = 0; bucket_index < local_bin.size(); bucket_index++) {
            if (!local_bin[bucket_index].empty()) {
                const std::int64_t new_min_index =
                    std::min(min_nonempty_bucket_index.load(), bucket_index);
                std::int64_t old_min_index = min_nonempty_bucket_index.load();
                while (new_min_index < old_min_index) {
                    if (min_nonempty_bucket_index.compare_exchange_strong(old_min_index,
                                                                          new_min_index)) {
                        break;
                    }
                    old_min_index = min_nonempty_bucket_index.load();
                }
                break;
            }
        }
    });

    curr_bin_index = min_nonempty_bucket_index.load();
}

template <typename SharedBinContainer, typename BinsVector>
inline std::int64_t reduce_to_common_bin(const std::int64_t& curr_bin_index,
                                         BinsVector& local_bins,
                                         SharedBinContainer& shared_bin) {
    std::atomic<std::int64_t> vertex_count_in_shared_bin = 0;
    dal::detail::threader_for(local_bins.size(), 1, [&](std::int64_t i) {
        auto& local_bin = local_bins[i];
        if (curr_bin_index < local_bin.size()) {
            auto& bucket = local_bin[curr_bin_index];
            if (bucket.size() != 0) {
                std::int64_t copy_start = vertex_count_in_shared_bin.fetch_add(bucket.size());
                copy(bucket.begin(), bucket.end(), shared_bin.get_mutable_data() + copy_start);
                bucket.resize(0);
            }
        }
    });
    return vertex_count_in_shared_bin.load();
}

template <typename Mode, typename VertexType, typename EdgeValue>
struct get_result_from_ralaxing_data {};

template <typename VertexType, typename EdgeValue>
struct get_result_from_ralaxing_data<mode::distances, VertexType, EdgeValue> {
    template <typename DataToRelax>
    inline traverse_result<task::one_to_all> operator()(
        const detail::descriptor_base<task::one_to_all>& desc,
        std::int64_t vertex_count,
        const DataToRelax& data) {
        using value_type = EdgeValue;
        auto dist_arr = array<value_type>::empty(vertex_count);
        value_type* dist_ = dist_arr.get_mutable_data();
        const auto computed_dist = data.get_distances_ptr();

        dal::detail::threader_for(vertex_count, 4096, [&](std::int64_t i) {
            dist_[i] = computed_dist[i];
        });

        return traverse_result<task::one_to_all>().set_distances(
            dal::detail::homogen_table_builder{}.reset(dist_arr, vertex_count, 1).build());
    }
};

template <typename VertexType, typename EdgeValue>
struct get_result_from_ralaxing_data<mode::distances_predecessors, VertexType, EdgeValue> {
    template <typename DataToRelax>
    inline traverse_result<task::one_to_all> operator()(
        const detail::descriptor_base<task::one_to_all>& desc,
        std::int64_t vertex_count,
        const DataToRelax& data) {
        using value_type = EdgeValue;
        using vertex_type = VertexType;
        if (desc.get_optional_results() & optional_results::distances) {
            auto dist_arr = array<value_type>::empty(vertex_count);
            auto pred_arr = array<vertex_type>::empty(vertex_count);
            value_type* dist_ = dist_arr.get_mutable_data();
            vertex_type* pred_ = pred_arr.get_mutable_data();

            dal::detail::threader_for(vertex_count, 4096, [&](std::int64_t i) {
                dist_[i] = data.get_distance(i);
                pred_[i] = data.get_predecessor(i);
            });

            return traverse_result<task::one_to_all>()
                .set_distances(
                    dal::detail::homogen_table_builder{}.reset(dist_arr, vertex_count, 1).build())
                .set_predecessors(
                    dal::detail::homogen_table_builder{}.reset(pred_arr, vertex_count, 1).build());
        }
        else {
            auto pred_arr = array<vertex_type>::empty(vertex_count);
            vertex_type* pred_ = pred_arr.get_mutable_data();

            dal::detail::threader_for(vertex_count, 4096, [&](std::int64_t i) {
                pred_[i] = data.get_predecessor(i);
            });

            return traverse_result<task::one_to_all>().set_predecessors(
                dal::detail::homogen_table_builder{}.reset(pred_arr, vertex_count, 1).build());
        }
    }
};

template <typename Cpu, typename EdgeValue, typename Mode>
struct delta_stepping {
    traverse_result<task::one_to_all> operator()(
        const detail::descriptor_base<task::one_to_all>& desc,
        const dal::preview::detail::topology<std::int32_t>& t,
        const EdgeValue* vals,
        byte_alloc_iface* alloc_ptr) {
        using value_type = EdgeValue;
        using vertex_type = std::int32_t;
        using vertex_allocator_type = inner_alloc<vertex_type>;

        vertex_allocator_type vertex_allocator(alloc_ptr);
        const auto source = dal::detail::integral_cast<std::int32_t>(desc.get_source());
        const value_type delta = desc.get_delta();
        const std::int64_t max_bin_count = std::numeric_limits<std::int64_t>::max() / 2;
        const std::int64_t max_elements_in_bin = 1000;
        const auto vertex_count = t.get_vertex_count();
        const value_type max_dist = std::numeric_limits<value_type>::max();
        using relaxing_data_t =
            typename relaxing_data_type<Mode, vertex_type, value_type>::value_type;
        data_to_relax<Cpu, Mode, vertex_type, relaxing_data_t> dist(vertex_count,
                                                                    source,
                                                                    max_dist,
                                                                    alloc_ptr);

        const std::int64_t thread_cnt = dal::detail::threader_get_max_threads();

        using v1v_t = vector_container<vertex_type, vertex_allocator_type>;
        using v1a_t = inner_alloc<v1v_t>;
        using v2v_t = vector_container<v1v_t, v1a_t>;
        using v2a_t = inner_alloc<v2v_t>;
        using v3v_t = vector_container<v2v_t, v2a_t>;
        v2a_t v2a(alloc_ptr);
        v3v_t local_bins(thread_cnt, v2a);
        local_bins[0].reserve(t.get_vertex_degree(source));

        v1v_t shared_bin(t.get_edge_count(), vertex_allocator);
        shared_bin[0] = source;
        std::int64_t curr_bin_index = 0;
        std::int64_t vertex_count_in_shared_bin = 1;

        vertex_type* local_processing_bins =
            allocate(vertex_allocator, max_elements_in_bin * thread_cnt);

        dal::detail::shared<vertex_type> local_processing_bins_shared(
            local_processing_bins,
            destroy_delete<vertex_type, vertex_allocator_type>(max_elements_in_bin * thread_cnt,
                                                               vertex_allocator));

        while (curr_bin_index != max_bin_count) {
            dal::detail::threader_for(
                vertex_count_in_shared_bin,
                1,
                [&](std::int64_t i) {
                    const vertex_type u = shared_bin[i];
                    if (dist.get_distance(u) >= delta * static_cast<value_type>(curr_bin_index)) {
                        relax_edges<Mode>()(
                            t,
                            vals,
                            u,
                            delta,
                            dist,
                            local_bins[dal::detail::threader_get_current_thread_index()]);
                    }
                });

            dal::detail::threader_for(thread_cnt, 1, [&](std::int64_t i) {
                const std::int64_t thread_id = i;
                auto& local_bin = local_bins[thread_id];
                while (curr_bin_index < local_bin.size() && !local_bin[curr_bin_index].empty() &&
                       local_bin[curr_bin_index].size() < max_elements_in_bin) {
                    auto copy_begin = local_processing_bins + max_elements_in_bin * thread_id;
                    auto& bucket = local_bin[curr_bin_index];
                    copy(bucket.begin(), bucket.end(), copy_begin);
                    const std::int64_t copy_count = bucket.size();
                    bucket.resize(0);
                    for (std::int64_t j = 0; j < copy_count; ++j) {
                        relax_edges<Mode>()(t, vals, copy_begin[j], delta, dist, local_bin);
                    }
                }
            });

            find_next_bin_index(curr_bin_index, local_bins);

            vertex_count_in_shared_bin =
                reduce_to_common_bin(curr_bin_index, local_bins, shared_bin);
        }
        return get_result_from_ralaxing_data<Mode, vertex_type, value_type>()(desc,
                                                                              vertex_count,
                                                                              dist);
    }
};

} // namespace oneapi::dal::preview::shortest_paths::backend
