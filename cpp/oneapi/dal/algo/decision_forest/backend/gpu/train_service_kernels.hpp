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

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/algo/decision_forest/train_types.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_misc_structs.hpp"

namespace oneapi::dal::decision_forest::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

template <typename Float,
          typename Bin = std::uint32_t,
          typename Index = std::int32_t,
          typename Task = task::by_default>
class train_service_kernels {
    using impl_const_t = impl_const<Index, Task>;
    using context_t = train_context<Float, Index, Task>;

public:
    train_service_kernels(sycl::queue& q) : queue_(q){};
    ~train_service_kernels() = default;

    std::uint64_t get_oob_rows_required_mem_size(Index row_count,
                                                 Index tree_count,
                                                 double observations_per_tree_fraction);

    sycl::event get_split_node_count(const pr::ndarray<Index, 1>& node_list,
                                     Index node_count,
                                     Index& split_node_count,
                                     const bk::event_vector& deps = {});

    sycl::event calculate_left_child_row_count_on_local_data(
        const context_t& ctx,
        const pr::ndarray<Bin, 2>& data,
        const pr::ndarray<Index, 1>& node_list,
        const pr::ndarray<Index, 1>& tree_order,
        Index column_count,
        Index node_count,
        const bk::event_vector& deps);

    sycl::event do_level_partition_by_groups(const context_t& ctx,
                                             const pr::ndarray<Bin, 2>& data,
                                             const pr::ndarray<Index, 1>& node_list,
                                             pr::ndarray<Index, 1>& tree_order,
                                             pr::ndarray<Index, 1>& tree_order_buf,
                                             Index data_row_count,
                                             Index data_selected_row_count,
                                             Index data_column_count,
                                             Index node_count,
                                             Index tree_count,
                                             const bk::event_vector& deps = {});

    sycl::event initialize_tree_order(pr::ndarray<Index, 1>& tree_order,
                                      Index tree_count,
                                      Index row_count,
                                      Index stride,
                                      const bk::event_vector& deps = {});

    sycl::event update_mdi_var_importance(const pr::ndarray<Index, 1>& node_list,
                                          const pr::ndarray<Float, 1>& node_imp_decrease_list,
                                          pr::ndarray<Float, 1>& res_var_imp,
                                          Index column_count,
                                          Index node_count,
                                          const bk::event_vector& deps = {});

    sycl::event mark_present_rows(const pr::ndarray<Index, 1>& row_list,
                                  pr::ndarray<Index, 1>& row_buffer,
                                  Index global_row_count,
                                  Index block_row_count,
                                  Index node_row_count,
                                  Index node_count,
                                  Index node_idx,
                                  Index krn_local_size,
                                  Index sbg_sum_count,
                                  const bk::event_vector& deps = {});

    sycl::event count_absent_rows_for_blocks(const pr::ndarray<Index, 1>& row_buffer,
                                             pr::ndarray<Index, 1>& part_sum_list,
                                             Index block_row_count,
                                             Index node_count,
                                             Index node_idx,
                                             Index krn_local_size,
                                             Index sbg_sum_count,
                                             const bk::event_vector& deps = {});

    sycl::event count_absent_rows_total(const pr::ndarray<Index, 1>& part_sum_list,
                                        pr::ndarray<Index, 1>& part_pref_sum_list,
                                        pr::ndarray<Index, 1>& oob_rows_num_list,
                                        Index node_count,
                                        Index node_idx,
                                        Index krn_local_size,
                                        Index sbg_sum_count,
                                        const bk::event_vector& deps = {});

    sycl::event fill_oob_rows_list_by_blocks(const pr::ndarray<Index, 1>& row_buffer,
                                             const pr::ndarray<Index, 1>& part_pref_sum_list,
                                             const pr::ndarray<Index, 1>& oob_row_num_list,
                                             pr::ndarray<Index, 1>& oob_row_list,
                                             Index block_row_count,
                                             Index node_count,
                                             Index node_idx,
                                             Index total_oob_row_num,
                                             Index krn_local_size,
                                             Index sbg_sum_count,
                                             const bk::event_vector& deps = {});

    sycl::event get_oob_row_list(const pr::ndarray<Index, 1>& row_list,
                                 const pr::ndarray<Index, 1>& node_list,
                                 pr::ndarray<Index, 1>& oob_row_count_list,
                                 pr::ndarray<Index, 1>& oob_row_list,
                                 Index global_row_count,
                                 Index block_row_count,
                                 Index node_count,
                                 const bk::event_vector& deps = {});

private:
    sycl::queue queue_;

    static constexpr inline double aproximate_oob_rows_fraction_ = 0.6;
    static constexpr inline Index partition_min_block_size_ = 128;
    // max blocks number for one node
    static constexpr inline Index partition_max_block_count_ = 256;

    static constexpr inline Index max_local_sums_ = 256;

    static constexpr inline Index preferable_group_size_ = 256;
    static constexpr inline Index preferable_partition_group_size_ = 128; // it showed best perf
    static constexpr inline Index preferable_partition_groups_count_ = 8192;
    // auxilliary buffer for nodes partitioning
    static constexpr inline Index aux_node_buffer_prop_count_ = 2;

    static constexpr inline Index preferable_sbg_size_ = 16;
    static constexpr inline Index max_sbg_count_per_group_ = 16;
};

#endif

} // namespace oneapi::dal::decision_forest::backend
