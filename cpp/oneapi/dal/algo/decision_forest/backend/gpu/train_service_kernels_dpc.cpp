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

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_service_kernels.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_helpers.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::decision_forest::backend {

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

using sycl::ext::oneapi::plus;
using sycl::ext::oneapi::minimum;
using sycl::ext::oneapi::maximum;

using alloc = sycl::usm::alloc;
using address = sycl::access::address_space;

template <typename Float, typename Bin, typename Index, typename Task>
std::uint64_t train_service_kernels<Float, Bin, Index, Task>::get_oob_rows_required_mem_size(
    Index row_count,
    Index tree_count,
    double observations_per_tree_fraction) {
    // mem size occupied on GPU for storing OOB rows indices
    const std::uint64_t oob_rows_aprox_count =
        row_count * (Float(1) - observations_per_tree_fraction) +
        row_count * observations_per_tree_fraction * aproximate_oob_rows_fraction_;
    return sizeof(Index) * oob_rows_aprox_count * tree_count;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_service_kernels<Float, Bin, Index, Task>::initialize_tree_order(
    pr::ndarray<Index, 1>& tree_order,
    Index tree_count,
    Index row_count,
    Index stride,
    const bk::event_vector& deps) {
    ONEDAL_ASSERT(tree_order.get_count() == tree_count * stride);

    Index* tree_order_ptr = tree_order.get_mutable_data();
    const sycl::range<2> range{ de::integral_cast<std::size_t>(row_count),
                                de::integral_cast<std::size_t>(tree_count) };

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::id<2> id) {
            tree_order_ptr[id[1] * stride + id[0]] = id[0];
        });
    });

    event.wait_and_throw();
    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_service_kernels<Float, Bin, Index, Task>::get_split_node_count(
    const pr::ndarray<Index, 1>& node_list,
    Index node_count,
    Index& split_node_count,
    const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(get_split_node_count, queue_);
    ONEDAL_ASSERT(node_list.get_count() == node_count * impl_const_t::node_prop_count_);

    const Index node_prop_count =
        impl_const_t::node_prop_count_; // num of split attributes for node
    const Index bad_val = impl_const_t::bad_val_;

    const Index* node_list_ptr = node_list.get_data();
    auto split_node_count_buf = pr::ndarray<Index, 1>::empty(queue_, { 1 }, alloc::device);
    Index* split_node_count_ptr = split_node_count_buf.get_mutable_data();

    auto krn_local_size = preferable_sbg_size_;
    const sycl::nd_range<1> nd_range =
        bk::make_multiple_nd_range_1d(krn_local_size, krn_local_size);

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();
            if (sbg.get_group_id() > 0) {
                return;
            }

            const Index local_id = sbg.get_local_id();
            const Index local_size = sbg.get_local_range()[0];

            Index sum = 0;
            for (Index i = local_id; i < node_count; i += local_size) {
                sum += Index(node_list_ptr[i * node_prop_count + impl_const_t::ind_fid] != bad_val);
            }

            sum = sycl::reduce_over_group(sbg, sum, plus<Index>());

            if (local_id == 0) {
                split_node_count_ptr[0] = sum;
            }
        });
    });

    auto split_node_count_host = split_node_count_buf.to_host(queue_, { event });
    split_node_count = split_node_count_host.get_data()[0];

    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event
train_service_kernels<Float, Bin, Index, Task>::calculate_left_child_row_count_on_local_data(
    const context_t& ctx,
    const pr::ndarray<Bin, 2>& data,
    const pr::ndarray<Index, 1>& node_list,
    const pr::ndarray<Index, 1>& tree_order,
    Index column_count,
    Index node_count,
    const bk::event_vector& deps) {
    // this function is used for ditributed mode only, because for batch it is claculated during bs
    ONEDAL_ASSERT(ctx.distr_mode_);

    ONEDAL_ASSERT(data.get_count() == ctx.row_count_ * ctx.column_count_);
    ONEDAL_ASSERT(node_list.get_count() == node_count * impl_const_t::node_prop_count_);
    ONEDAL_ASSERT(tree_order.get_count() == ctx.tree_in_block_ * ctx.selected_row_total_count_);

    const Index total_block_count = de::check_mul_overflow(node_count, partition_max_block_count_);

    const Index node_prop_count =
        impl_const_t::node_prop_count_; // num of split attributes for node
    const Index leaf_mark = impl_const_t::leaf_mark_;
    const Index max_block_count = partition_max_block_count_;
    const Index min_block_size = partition_min_block_size_;

    const Bin* data_ptr = data.get_data();
    Index* node_list_ptr = node_list.get_mutable_data();
    const Index* tree_order_ptr = tree_order.get_data();

    auto krn_local_size = preferable_partition_group_size_;
    const sycl::nd_range<1> nd_range =
        bk::make_multiple_nd_range_1d(preferable_partition_groups_count_ * krn_local_size,
                                      krn_local_size);

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();
            const Index sub_group_size = sbg.get_local_range()[0];
            const Index work_group_size = item.get_local_range()[0];
            const Index sub_groups_in_work_group_num =
                work_group_size / sub_group_size; // num of subgroups for current node processing

            const Index sub_group_local_id = sbg.get_local_id();
            const Index work_group_local_id = item.get_local_id()[0];

            const Index sub_group_id =
                item.get_group().get_group_id(0) * sub_groups_in_work_group_num +
                work_group_local_id / sub_group_size;
            const Index sub_groups_num =
                item.get_group_range(0) *
                sub_groups_in_work_group_num; // num of subgroups for current node processing

            for (Index block_ind_glob = sub_group_id; block_ind_glob < total_block_count;
                 block_ind_glob += sub_groups_num) {
                const Index node_id = block_ind_glob / max_block_count;
                const Index block_ind = block_ind_glob % max_block_count;

                Index* node = node_list_ptr + node_id * node_prop_count;
                const Index offset = node[impl_const_t::ind_ofs];
                const Index row_count = node[impl_const_t::ind_lrc];
                const Index feat_id = node[impl_const_t::ind_fid];
                const Index split_val = node[impl_const_t::ind_bin];

                Index node_block_count =
                    row_count / min_block_size
                        ? sycl::min(row_count / min_block_size, max_block_count)
                        : 1;

                // if block_ind assigned for this sbg less than current node's block count -> sbg will just go to the next node
                if (feat_id != leaf_mark && block_ind < node_block_count) // split node
                {
                    const Index block_size =
                        node_block_count > 1
                            ? row_count / node_block_count + bool(row_count % node_block_count)
                            : row_count;

                    const Index ind_end = sycl::min((block_ind + 1) * block_size, row_count);
                    const Index ind_start = sycl::min(block_ind * block_size, ind_end);
                    const Index group_row_count = ind_end - ind_start;

                    if (group_row_count > 0) {
                        Index group_row_to_right_count = 0;
                        for (Index i = ind_start + sub_group_local_id; i < ind_end;
                             i += sub_group_size) {
                            const Index id = tree_order_ptr[offset + i];
                            const Index to_right =
                                Index(static_cast<Index>(data_ptr[id * column_count + feat_id]) >
                                      split_val);
                            group_row_to_right_count +=
                                sycl::reduce_over_group(sbg, to_right, plus<Index>());
                        }

                        if (0 == sub_group_local_id) {
                            bk::atomic_global_add(node + impl_const_t::ind_lch_lrc,
                                                  group_row_count - group_row_to_right_count);
                        }
                    }
                }
            }
        });
    });

    event.wait_and_throw();

    return event;
}
template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_service_kernels<Float, Bin, Index, Task>::do_level_partition_by_groups(
    const context_t& ctx,
    const pr::ndarray<Bin, 2>& data,
    const pr::ndarray<Index, 1>& node_list,
    pr::ndarray<Index, 1>& tree_order,
    pr::ndarray<Index, 1>& tree_order_buf,
    Index data_row_count,
    Index data_selected_row_count,
    Index data_column_count,
    Index node_count,
    Index tree_count,
    const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(do_level_partition, queue_);

    ONEDAL_ASSERT(data.get_count() == data_row_count * data_column_count);
    ONEDAL_ASSERT(node_list.get_count() == node_count * impl_const_t::node_prop_count_);
    ONEDAL_ASSERT(tree_order.get_count() == data_selected_row_count * tree_count);
    ONEDAL_ASSERT(tree_order_buf.get_count() == data_selected_row_count * tree_count);

    const Index total_block_count = de::check_mul_overflow(node_count, partition_max_block_count_);

    // node_aux_list is auxilliary buffer for synchronization of left and right boundaries of blocks (elems_to_left_count, elems_to_right_count)
    // processed by subgroups in the same node
    // no mul overflow check is required due to there is already buffer of size node_count * impl_const_t::node_prop_count_
    ONEDAL_ASSERT(aux_node_buffer_prop_count_ <= impl_const_t::node_prop_count_);

    auto [node_aux_list, last_event] =
        pr::ndarray<Index, 1>::zeros(queue_,
                                     { node_count * aux_node_buffer_prop_count_ },
                                     alloc::device);
    last_event.wait_and_throw();

    const Index node_prop_count =
        impl_const_t::node_prop_count_; // num of split attributes for node
    const Index leaf_mark = impl_const_t::leaf_mark_;
    const Index aux_node_buffer_prop_count =
        aux_node_buffer_prop_count_; // num of auxilliary attributes for node
    const Index max_block_count = partition_max_block_count_;
    const Index min_block_size = partition_min_block_size_;

    const Bin* data_ptr = data.get_data();
    const Index* node_list_ptr = node_list.get_data();
    Index* node_aux_list_ptr = node_aux_list.get_mutable_data();
    const Index* tree_order_ptr = tree_order.get_data();
    Index* tree_order_buf_ptr = tree_order_buf.get_mutable_data();

    bool distr_mode = ctx.distr_mode_;

    auto krn_local_size = preferable_partition_group_size_;
    const sycl::nd_range<1> nd_range =
        bk::make_multiple_nd_range_1d(preferable_partition_groups_count_ * krn_local_size,
                                      krn_local_size);

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();
            const Index sub_group_size = sbg.get_local_range()[0];
            const Index work_group_size = item.get_local_range()[0];
            const Index sub_groups_in_work_group_num =
                work_group_size / sub_group_size; // num of subgroups for current node processing

            const Index sub_group_local_id = sbg.get_local_id();
            const Index work_group_local_id = item.get_local_id()[0];

            const Index sub_group_id =
                item.get_group().get_group_id(0) * sub_groups_in_work_group_num +
                work_group_local_id / sub_group_size;
            const Index sub_groups_num =
                item.get_group_range(0) *
                sub_groups_in_work_group_num; // num of subgroups for current node processing

            for (Index block_ind_glob = sub_group_id; block_ind_glob < total_block_count;
                 block_ind_glob += sub_groups_num) {
                const Index node_id = block_ind_glob / max_block_count;
                const Index block_ind = block_ind_glob % max_block_count;

                const Index* node = node_list_ptr + node_id * node_prop_count;
                const Index offset = node[impl_const_t::ind_ofs];
                const Index row_count =
                    distr_mode ? node[impl_const_t::ind_lrc] : node[impl_const_t::ind_grc];
                const Index feat_id = node[impl_const_t::ind_fid];
                const Index feat_bin = node[impl_const_t::ind_bin];
                const Index left_ch_row_count =
                    distr_mode
                        ? node[impl_const_t::ind_lch_lrc]
                        : node[impl_const_t::ind_lch_grc]; // num of items in the Left part of node

                Index node_block_count =
                    row_count / min_block_size
                        ? sycl::min(row_count / min_block_size, max_block_count)
                        : 1;

                // if block_ind assigned for this sbg less than current node's block count -> sbg will just go to the next node
                if (feat_id != leaf_mark && block_ind < node_block_count) // split node
                {
                    Index* node_aux = node_aux_list_ptr + node_id * aux_node_buffer_prop_count;

                    const Index block_size =
                        node_block_count > 1
                            ? row_count / node_block_count + bool(row_count % node_block_count)
                            : row_count;

                    const Index ind_end = sycl::min((block_ind + 1) * block_size, row_count);
                    const Index ind_start = sycl::min(block_ind * block_size, ind_end);
                    const Index group_row_count = ind_end - ind_start;

                    Index group_left_boundary = 0;
                    Index group_right_boundary = 0;

                    if (node_block_count > 1 && group_row_count > 0) {
                        Index group_row_to_right_count = 0;
                        for (Index i = ind_start + sub_group_local_id; i < ind_end;
                             i += sub_group_size) {
                            const Index id = tree_order_ptr[offset + i];
                            const Index to_right =
                                Index(static_cast<Index>(
                                          data_ptr[id * data_column_count + feat_id]) > feat_bin);
                            group_row_to_right_count +=
                                sycl::reduce_over_group(sbg, to_right, plus<Index>());
                        }

                        if (0 == sub_group_local_id) {
                            group_left_boundary =
                                bk::atomic_global_add(node_aux + 0,
                                                      group_row_count - group_row_to_right_count);
                            group_right_boundary =
                                bk::atomic_global_add(node_aux + 1, group_row_to_right_count);
                        }
                        group_left_boundary = sycl::group_broadcast(sbg, group_left_boundary, 0);
                        group_right_boundary = sycl::group_broadcast(sbg, group_right_boundary, 0);
                    }

                    Index group_row_to_right_count = 0;
                    for (Index i = ind_start + sub_group_local_id; i < ind_end;
                         i += sub_group_size) {
                        const Index id = tree_order_ptr[offset + i];
                        const Index to_right =
                            Index(static_cast<Index>(data_ptr[id * data_column_count + feat_id]) >
                                  feat_bin);
                        const Index boundary =
                            group_row_to_right_count +
                            sycl::exclusive_scan_over_group(sbg, to_right, plus<Index>());
                        const Index pos_new =
                            (to_right ? left_ch_row_count + group_right_boundary + boundary
                                      : group_left_boundary + i - ind_start - boundary);
                        tree_order_buf_ptr[offset + pos_new] = id;
                        group_row_to_right_count +=
                            sycl::reduce_over_group(sbg, to_right, plus<Index>());
                    }
                }
            }
        });
    });

    event.wait_and_throw();

    std::swap(tree_order, tree_order_buf);
    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_service_kernels<Float, Bin, Index, Task>::update_mdi_var_importance(
    const pr::ndarray<Index, 1>& node_list,
    const pr::ndarray<Float, 1>& node_imp_decrease_list,
    pr::ndarray<Float, 1>& res_var_imp,
    Index data_column_count,
    Index node_count,
    const bk::event_vector& deps) {
    ONEDAL_ASSERT(node_list.get_count() == node_count * impl_const_t::node_prop_count_);
    ONEDAL_ASSERT(node_imp_decrease_list.get_count() == node_count);
    ONEDAL_ASSERT(res_var_imp.get_count() == data_column_count);

    const Index krn_local_size = bk::down_pow2(std::min(preferable_group_size_, node_count));

    const Index* node_list_ptr = node_list.get_data();
    const Float* node_imp_decrease_list_ptr = node_imp_decrease_list.get_data();
    Float* res_var_imp_ptr = res_var_imp.get_mutable_data();

    const sycl::nd_range<2> nd_range =
        bk::make_multiple_nd_range_2d({ krn_local_size, data_column_count }, { krn_local_size, 1 });

    const Index node_prop_count =
        impl_const_t::node_prop_count_; // num of split attributes for node
    const Index leaf_mark = impl_const_t::leaf_mark_;
    const Index max_sub_groups_num =
        max_sbg_count_per_group_; //need to calculate it via device info

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        sycl::local_accessor<Float, 1> buf(max_sub_groups_num, cgh);

        cgh.parallel_for(nd_range, [=](sycl::nd_item<2> item) {
            auto sbg = item.get_sub_group();

            const Index local_id = item.get_local_id()[0];
            const Index sub_group_local_id = sbg.get_local_id();
            const Index sub_group_size = sbg.get_local_range()[0];
            const Index local_size = item.get_local_range()[0];
            const Index n_sub_groups =
                local_size / sub_group_size; // num of subgroups for current node processing
            const Index sub_group_id = local_id / sub_group_size;

            const Index buf_idx =
                item.get_global_id()[1] %
                (max_sub_groups_num / n_sub_groups); // local buffer is shared between 16 sub groups
            const Index ftr_id = item.get_global_id()[1];

            const Index sbg_elem_count =
                node_count / n_sub_groups + bool(node_count % n_sub_groups);

            const Index ind_start = sub_group_id * sbg_elem_count;
            const Index ind_end = sycl::min((sub_group_id + 1) * sbg_elem_count, node_count);

            Float ftr_imp = Float(0);

            for (Index node_idx = ind_start + sub_group_local_id; node_idx < ind_end;
                 node_idx += sub_group_size) {
                Index split_ftr_id =
                    node_list_ptr[node_idx * node_prop_count + impl_const_t::ind_fid];
                ftr_imp +=
                    sycl::reduce_over_group(sbg,
                                            ((split_ftr_id != leaf_mark && ftr_id == split_ftr_id)
                                                 ? node_imp_decrease_list_ptr[node_idx]
                                                 : Float(0)),
                                            plus<Float>());
            }

            if (0 == sub_group_local_id) {
                if (1 == n_sub_groups) {
                    res_var_imp_ptr[ftr_id] += ftr_imp;
                }
                else {
                    buf[buf_idx + sub_group_id] = ftr_imp;
                }
            }

            item.barrier(sycl::access::fence_space::local_space);
            if (1 < n_sub_groups && 0 == sub_group_id) {
                // first sub group for current node reduces over local buffer if required
                Float ftr_imp = (sub_group_local_id < n_sub_groups)
                                    ? buf[buf_idx + sub_group_local_id]
                                    : (Float)0;
                Float total_ftr_imp = sycl::reduce_over_group(sbg, ftr_imp, plus<Float>());

                if (0 == local_id) {
                    res_var_imp_ptr[ftr_id] += total_ftr_imp;
                }
            }
        });
    });

    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_service_kernels<Float, Bin, Index, Task>::mark_present_rows(
    const pr::ndarray<Index, 1>& row_list,
    pr::ndarray<Index, 1>& row_buffer,
    Index global_row_count,
    Index block_row_count,
    Index node_row_count,
    Index node_count,
    Index node_idx,
    Index krn_local_size,
    Index sbg_sum_count,
    const bk::event_vector& deps) {
    ONEDAL_ASSERT(row_list.get_count() == de::check_mul_overflow(global_row_count, node_count));
    ONEDAL_ASSERT(row_buffer.get_count() == de::check_mul_overflow(block_row_count, node_count));

    const Index* rows_list_ptr = row_list.get_data();
    Index* rows_buffer_ptr = row_buffer.get_mutable_data();
    const Index item_present_mark = 1;

    const sycl::nd_range<1> nd_range =
        bk::make_multiple_nd_range_1d(krn_local_size * sbg_sum_count, krn_local_size);

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();

            const Index n_groups = item.get_group_range(0);
            const Index n_sub_groups = sbg.get_group_range()[0];
            const Index n_total_sub_groups = n_sub_groups * n_groups;
            const Index elems_for_sbg =
                node_row_count / n_total_sub_groups + bool(node_row_count % n_total_sub_groups);

            const Index local_size = sbg.get_local_range()[0];

            const Index local_id = sbg.get_local_id();
            const Index sub_group_id = sbg.get_group_id();
            const Index group_id = item.get_group().get_group_id(0) * n_sub_groups + sub_group_id;

            const Index ind_start = group_id * elems_for_sbg;
            const Index ind_end = sycl::min((group_id + 1) * elems_for_sbg, node_row_count);

            for (Index i = ind_start + local_id; i < ind_end; i += local_size) {
                rows_buffer_ptr[block_row_count * node_idx +
                                rows_list_ptr[global_row_count * node_idx + i]] = item_present_mark;
            }
        });
    });

    event.wait_and_throw();

    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_service_kernels<Float, Bin, Index, Task>::count_absent_rows_for_blocks(
    const pr::ndarray<Index, 1>& row_buffer,
    pr::ndarray<Index, 1>& part_sum_list,
    Index block_row_count,
    Index node_count,
    Index node_idx,
    Index krn_local_size,
    Index sbg_sum_count,
    const bk::event_vector& deps) {
    ONEDAL_ASSERT(row_buffer.get_count() == block_row_count * node_count);
    ONEDAL_ASSERT(part_sum_list.get_count() == sbg_sum_count);

    const Index* rows_buffer_ptr = row_buffer.get_data();
    Index* part_sum_list_ptr = part_sum_list.get_mutable_data();
    const Index item_absent_mark = -1;

    const sycl::nd_range<1> nd_range =
        bk::make_multiple_nd_range_1d(krn_local_size * sbg_sum_count, krn_local_size);

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();

            const Index n_groups = item.get_group_range(0);
            const Index n_sub_groups = sbg.get_group_range()[0];
            const Index n_total_sub_groups = n_sub_groups * n_groups;
            const Index elems_for_sbg =
                block_row_count / n_total_sub_groups + bool(block_row_count % n_total_sub_groups);

            const Index local_size = sbg.get_local_range()[0];

            const Index local_id = sbg.get_local_id();
            const Index sub_group_id = sbg.get_group_id();
            const Index group_id = item.get_group().get_group_id(0) * n_sub_groups + sub_group_id;

            const Index ind_start = group_id * elems_for_sbg;
            const Index ind_end = sycl::min((group_id + 1) * elems_for_sbg, block_row_count);

            Index sub_sum = 0;

            for (Index i = ind_start + local_id; i < ind_end; i += local_size) {
                sub_sum +=
                    Index(item_absent_mark == rows_buffer_ptr[block_row_count * node_idx + i]);
            }

            Index sum = sycl::reduce_over_group(sbg, sub_sum, plus<Index>());

            if (local_id == 0) {
                part_sum_list_ptr[group_id] = sum;
            }
        });
    });

    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_service_kernels<Float, Bin, Index, Task>::count_absent_rows_total(
    const pr::ndarray<Index, 1>& part_sum_list,
    pr::ndarray<Index, 1>& part_pref_sum_list,
    pr::ndarray<Index, 1>& oob_rows_num_list,
    Index tree_count,
    Index tree_idx,
    Index krn_local_size,
    Index sbg_sum_count,
    const bk::event_vector& deps) {
    ONEDAL_ASSERT(part_sum_list.get_count() == sbg_sum_count);
    ONEDAL_ASSERT(part_pref_sum_list.get_count() == sbg_sum_count * tree_count);
    ONEDAL_ASSERT(oob_rows_num_list.get_count() == tree_count + 1);

    const Index* part_sum_list_ptr = part_sum_list.get_data();
    Index* part_pref_sum_list_ptr = part_pref_sum_list.get_mutable_data();
    Index* total_sum_ptr = oob_rows_num_list.get_mutable_data();

    const sycl::nd_range<1> nd_range =
        bk::make_multiple_nd_range_1d(krn_local_size * sbg_sum_count, krn_local_size);

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();

            if (sbg.get_group_id() > 0)
                return;
            const Index local_size = sbg.get_local_range()[0];
            const Index local_id = sbg.get_local_id();

            Index sum = 0;

            for (Index i = local_id; i < sbg_sum_count; i += local_size) {
                Index value = part_sum_list_ptr[i];
                Index boundary = sycl::exclusive_scan_over_group(sbg, value, plus<Index>());
                part_pref_sum_list_ptr[sbg_sum_count * tree_idx + i] = sum + boundary;
                sum += sycl::reduce_over_group(sbg, value, plus<Index>());
            }

            if (local_id == 0) {
                total_sum_ptr[tree_idx + 1] = total_sum_ptr[tree_idx] + sum;
            }
        });
    });

    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_service_kernels<Float, Bin, Index, Task>::fill_oob_rows_list_by_blocks(
    const pr::ndarray<Index, 1>& row_buffer,
    const pr::ndarray<Index, 1>& part_pref_sum_list,
    const pr::ndarray<Index, 1>& oob_row_num_list,
    pr::ndarray<Index, 1>& oob_row_list,
    Index block_row_count,
    Index node_count,
    Index node_idx,
    Index total_oob_row_num,
    Index krn_local_size,
    Index sbg_sum_count,
    const bk::event_vector& deps) {
    ONEDAL_ASSERT(row_buffer.get_count() == block_row_count * node_count);
    ONEDAL_ASSERT(part_pref_sum_list.get_count() == sbg_sum_count * node_count);
    ONEDAL_ASSERT(oob_row_num_list.get_count() == node_count + 1);
    ONEDAL_ASSERT(oob_row_list.get_count() == total_oob_row_num);

    const Index* rows_buffer_ptr = row_buffer.get_data();
    const Index* part_pref_sum_list_ptr = part_pref_sum_list.get_data();
    const Index* oob_row_num_list_ptr = oob_row_num_list.get_data();
    Index* oob_row_list_ptr = oob_row_list.get_mutable_data();

    const Index item_absent_mark = -1;

    const sycl::nd_range<1> nd_range =
        bk::make_multiple_nd_range_1d(krn_local_size * sbg_sum_count, krn_local_size);

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();

            const Index n_groups = item.get_group_range(0);
            const Index n_sub_groups = sbg.get_group_range()[0];
            const Index n_total_sub_groups = n_sub_groups * n_groups;
            const Index elems_for_sbg =
                block_row_count / n_total_sub_groups + bool(block_row_count % n_total_sub_groups);

            const Index local_size = sbg.get_local_range()[0];

            const Index local_id = sbg.get_local_id();
            const Index sub_group_id = sbg.get_group_id();
            const Index group_id = item.get_group().get_group_id(0) * n_sub_groups + sub_group_id;

            const Index ind_start = group_id * elems_for_sbg;
            const Index ind_end = sycl::min((group_id + 1) * elems_for_sbg, block_row_count);

            const Index oob_row_list_offset = oob_row_num_list_ptr[node_idx];

            Index group_offset = part_pref_sum_list_ptr[n_groups * node_idx + group_id];
            Index sum = 0;

            for (Index i = ind_start + local_id; i < ind_end; i += local_size) {
                Index oob_row =
                    Index(item_absent_mark == rows_buffer_ptr[block_row_count * node_idx + i]);
                Index pos = group_offset + sum +
                            sycl::exclusive_scan_over_group(sbg, oob_row, plus<Index>());
                if (oob_row) {
                    oob_row_list_ptr[oob_row_list_offset + pos] = i;
                }
                sum += sycl::reduce_over_group(sbg, oob_row, plus<Index>());
            }
        });
    });

    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_service_kernels<Float, Bin, Index, Task>::get_oob_row_list(
    const pr::ndarray<Index, 1>& row_list,
    const pr::ndarray<Index, 1>& node_list,
    pr::ndarray<Index, 1>& oob_row_count_list,
    pr::ndarray<Index, 1>& oob_row_list,
    Index global_row_count,
    Index block_row_count,
    Index node_count,
    const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(get_oob_row_list, queue_);

    const Index absent_mark = -1;
    const Index krn_local_size = preferable_sbg_size_;
    const Index sbg_sum_count =
        max_local_sums_ * krn_local_size < block_row_count
            ? max_local_sums_
            : (block_row_count / krn_local_size + !(block_row_count / krn_local_size));

    ONEDAL_ASSERT(row_list.get_count() == global_row_count * node_count);
    ONEDAL_ASSERT(oob_row_count_list.get_count() == node_count + 1);
    // oob_row_list will be created here

    sycl::event::wait_and_throw(deps);

    // it is filled with marks Present/Absent for each rows
    auto [row_buffer, last_event] = pr::ndarray<Index, 1>::full(queue_,
                                                                { block_row_count * node_count },
                                                                absent_mark,
                                                                alloc::device);
    last_event.wait_and_throw();

    auto part_sum_list = pr::ndarray<Index, 1>::empty(queue_, { sbg_sum_count }, alloc::device);
    auto part_prefix_sum_list =
        pr::ndarray<Index, 1>::empty(queue_, { sbg_sum_count * node_count }, alloc::device);
    Index total_oob_row_count = 0;

    last_event = oob_row_count_list.fill(queue_, 0);

    auto node_list_host = node_list.to_host(queue_, deps);
    auto node_list_host_ptr = node_list_host.get_data();

    for (Index node_idx = 0; node_idx < node_count; ++node_idx) {
        auto node_row_count =
            node_list_host_ptr[node_idx * impl_const_t::node_prop_count_ + impl_const_t::ind_lrc];
        last_event = mark_present_rows(row_list,
                                       row_buffer,
                                       global_row_count,
                                       block_row_count,
                                       node_row_count,
                                       node_count,
                                       node_idx,
                                       krn_local_size,
                                       sbg_sum_count,
                                       { last_event });
        last_event = count_absent_rows_for_blocks(row_buffer,
                                                  part_sum_list,
                                                  block_row_count,
                                                  node_count,
                                                  node_idx,
                                                  krn_local_size,
                                                  sbg_sum_count,
                                                  { last_event });
        last_event = count_absent_rows_total(part_sum_list,
                                             part_prefix_sum_list,
                                             oob_row_count_list,
                                             node_count,
                                             node_idx,
                                             krn_local_size,
                                             sbg_sum_count,
                                             { last_event });
    }

    auto oob_row_count_host = oob_row_count_list.to_host(queue_, { last_event });
    const Index* oob_row_count_host_ptr = oob_row_count_host.get_data();
    total_oob_row_count = oob_row_count_host_ptr[node_count];

    if (total_oob_row_count > 0) {
        // assign buffer of required size to the input oob_row_list buffer
        oob_row_list = pr::ndarray<Index, 1>::empty(queue_, { total_oob_row_count }, alloc::device);

        for (Index node_idx = 0; node_idx < node_count; ++node_idx) {
            Index oob_row_count =
                oob_row_count_host_ptr[node_idx + 1] - oob_row_count_host_ptr[node_idx];

            if (oob_row_count > 0) {
                last_event = fill_oob_rows_list_by_blocks(row_buffer,
                                                          part_prefix_sum_list,
                                                          oob_row_count_list,
                                                          oob_row_list,
                                                          block_row_count,
                                                          node_count,
                                                          node_idx,
                                                          total_oob_row_count,
                                                          krn_local_size,
                                                          sbg_sum_count,
                                                          { last_event });
                last_event.wait_and_throw();
            }
        }
    }

    return last_event;
}

#define INSTANTIATE(F, B, I, T) template class train_service_kernels<F, B, I, T>;

INSTANTIATE(float, std::uint32_t, std::int32_t, task::classification);
INSTANTIATE(float, std::uint32_t, std::int32_t, task::regression);

INSTANTIATE(double, std::uint32_t, std::int32_t, task::classification);
INSTANTIATE(double, std::uint32_t, std::int32_t, task::regression);

} // namespace oneapi::dal::decision_forest::backend

#endif
