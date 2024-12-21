/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_helpers.hpp"
#include <iostream>
#ifdef ONEDAL_DATA_PARALLEL

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_splitter_impl.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_splitter_helpers.hpp"

namespace oneapi::dal::decision_forest::backend {

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

using alloc = sycl::usm::alloc;
using address = sycl::access::address_space;

using sycl::ext::oneapi::plus;
using sycl::ext::oneapi::minimum;
using sycl::ext::oneapi::maximum;

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_splitter_impl<Float, Bin, Index, Task>::random_split(
    sycl::queue& queue,
    const context_t& ctx,
    const pr::ndview<Bin, 2>& data,
    const pr::ndview<Float, 1>& response,
    const pr::ndview<Index, 1>& tree_order,
    const pr::ndview<Index, 1>& selected_ftr_list,
    const pr::ndview<Float, 1>& random_bins_com,
    const pr::ndview<Index, 1>& bin_offset_list,
    const imp_data_t& imp_data_list,
    pr::ndview<Index, 1>& node_list,
    imp_data_t& left_child_imp_data_list,
    pr::ndview<Float, 1>& node_imp_dec_list,
    bool update_imp_dec_required,
    Index node_count,
    const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(random_split, queue);
    using split_smp_t = split_smp<Float, Index, Task>;
    using split_scalar_t = split_scalar<Float, Index, Task>;
    using split_info_t = split_info<Float, Index, Task>;

    Index hist_prop_count = 0;
    if constexpr (std::is_same_v<std::decay_t<Task>, task::classification>) {
        hist_prop_count = ctx.class_count_;
    }
    else {
        hist_prop_count = impl_const<Index, task::regression>::hist_prop_count_;
    }

    ONEDAL_ASSERT(data.get_count() == ctx.row_count_ * ctx.column_count_);
    ONEDAL_ASSERT(response.get_count() == ctx.row_count_);
    ONEDAL_ASSERT(tree_order.get_count() == ctx.tree_in_block_ * ctx.selected_row_total_count_);
    ONEDAL_ASSERT(selected_ftr_list.get_count() >= node_count * ctx.selected_ftr_count_);
    ONEDAL_ASSERT(bin_offset_list.get_count() == ctx.column_count_ + 1);
    ONEDAL_ASSERT(imp_data_list.imp_list_.get_count() >=
                  node_count * impl_const_t::node_imp_prop_count_);
    if constexpr (std::is_same_v<Task, task::classification>) {
        ONEDAL_ASSERT(imp_data_list.class_hist_list_.get_count() >= node_count * ctx.class_count_);
    }
    ONEDAL_ASSERT(node_list.get_count() >= node_count * impl_const_t::node_prop_count_);
    ONEDAL_ASSERT(left_child_imp_data_list.imp_list_.get_count() >=
                  node_count * impl_const_t::node_imp_prop_count_);
    if constexpr (std::is_same_v<Task, task::classification>) {
        ONEDAL_ASSERT(left_child_imp_data_list.class_hist_list_.get_count() >=
                      node_count * ctx.class_count_);
    }

    if (update_imp_dec_required) {
        ONEDAL_ASSERT(node_imp_dec_list.get_count() >= node_count);
    }

    const Bin* const data_ptr = data.get_data();
    const Float* const response_ptr = response.get_data();
    const Index* const tree_order_ptr = tree_order.get_data();

    const Index* const selected_ftr_list_ptr = selected_ftr_list.get_data();

    imp_data_list_ptr<Float, Index, Task> imp_list_ptr(imp_data_list);

    Index* const node_list_ptr = node_list.get_mutable_data();
    Float* const node_imp_decr_list_ptr =
        update_imp_dec_required ? node_imp_dec_list.get_mutable_data() : nullptr;

    imp_data_list_ptr_mutable<Float, Index, Task> left_imp_list_ptr(left_child_imp_data_list);

    const Float* const ftr_rnd_ptr = random_bins_com.get_data();

    const Index column_count = ctx.column_count_;
    const Index selected_ftr_count = ctx.selected_ftr_count_;
    const Index index_max = ctx.index_max_;

    Index local_size = bk::device_max_wg_size(queue);

    const std::size_t bins_per_wg = 2; // 2 for best and current split
    const std::size_t hist_size = hist_prop_count * bins_per_wg;

    const Index class_count = ctx.class_count_;
    const Float imp_threshold = ctx.impurity_threshold_;
    const Index min_obs_leaf = ctx.min_observations_in_leaf_node_;
    const Index max_bin_count_among_ftrs = ctx.max_bin_count_among_ftrs_;

    const Float* node_imp_list_ptr = imp_list_ptr.imp_list_ptr_;
    Float* left_child_imp_list_ptr = left_imp_list_ptr.imp_list_ptr_;

    // following vars are not used for regression, but should present to compile kernel
    const Index* class_hist_list_ptr = imp_list_ptr.get_class_hist_list_ptr_or_null();
    Index* left_child_class_hist_list_ptr = left_imp_list_ptr.get_class_hist_list_ptr_or_null();

    Index node_in_block_count = node_count;

    std::size_t local_buf_byte_size = local_size * sizeof(Float) + hist_size * sizeof(hist_type_t);
    ONEDAL_ASSERT(device_has_enough_local_mem(queue, local_buf_byte_size));

    const auto nd_range =
        bk::make_multiple_nd_range_2d({ local_size, node_in_block_count }, { local_size, 1 });
    sycl::event last_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        local_accessor_rw_t<hist_type_t> local_hist_buf(hist_size, cgh);
        local_accessor_rw_t<Float> local_float_buf(local_size, cgh);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<2> item) {
            auto sbg = item.get_sub_group();
            const Index node_idx = item.get_global_id(1);
            if (node_idx > (node_count - 1)) {
                return;
            }

            const Index node_id = node_idx;
            Index* node_ptr = node_list_ptr + node_id * impl_const_t::node_prop_count_;

            const Index local_id = item.get_local_id(0);

            const Index row_ofs = node_ptr[impl_const_t::ind_ofs];
            const Index row_count = node_ptr[impl_const_t::ind_lrc];

            split_smp_t sp_hlp;
            // Check node impurity
            if (!sp_hlp.is_valid_impurity(node_imp_list_ptr, node_id, imp_threshold, row_count)) {
                return;
            }
            split_info_t bs;

            // slm pointers declaration
#if __SYCL_COMPILER_VERSION >= 20230828
            hist_type_t* hist_ptr =
                local_hist_buf.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
            Float* local_buf_float_ptr =
                local_float_buf.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
#else
            hist_type_t* hist_ptr = local_hist_buf.get_pointer().get();
            Float* local_buf_float_ptr = local_float_buf.get_pointer().get();
#endif

            bs.init_clear(hist_ptr + 0 * hist_prop_count, hist_prop_count);
            split_scalar_t& bs_scal = bs.scalars;

            for (Index ftr_idx = 0; ftr_idx < selected_ftr_count; ftr_idx++) {
                split_info_t ts;
                ts.init(hist_ptr + 1 * hist_prop_count, hist_prop_count);
                split_scalar_t& ts_scal = ts.scalars;
                ts_scal.ftr_id = selected_ftr_list_ptr[node_id * selected_ftr_count + ftr_idx];
                const Index id =
                    (local_id < row_count) ? tree_order_ptr[row_ofs + local_id] : index_max;
                const Index bin = (local_id < row_count)
                                      ? data_ptr[id * column_count + ts_scal.ftr_id]
                                      : index_max;
                const Float response = (local_id < row_count) ? response_ptr[id] : Float(0);
                const Index response_int =
                    (local_id < row_count) ? static_cast<Index>(response) : -1;

                const Index min_bin =
                    sycl::reduce_over_group(item.get_group(),
                                            bin < index_max ? bin : max_bin_count_among_ftrs,
                                            minimum<Index>());
                const Index max_bin =
                    sycl::reduce_over_group(item.get_group(),
                                            bin < max_bin_count_among_ftrs ? bin : 0,
                                            maximum<Index>());

                const Float rand_val = ftr_rnd_ptr[node_id * selected_ftr_count + ftr_idx];
                const Index random_bin_ofs = static_cast<Index>(rand_val * (max_bin - min_bin + 1));
                ts_scal.ftr_bin = min_bin + random_bin_ofs;

                const Index count = Index(bin <= ts_scal.ftr_bin);

                if constexpr (std::is_same_v<Task, task::classification>) {
                    const Index left_count =
                        sycl::reduce_over_group(item.get_group(), count, plus<Index>());
                    const Index val = (bin <= ts_scal.ftr_bin) ? response_int : -1;
                    Index all_class_count = 0;

                    for (Index class_id = 0; class_id < class_count - 1; ++class_id) {
                        Index total_class_count = sycl::reduce_over_group(item.get_group(),
                                                                          Index(class_id == val),
                                                                          plus<Index>());
                        all_class_count += total_class_count;
                        ts.left_hist[class_id] = total_class_count;
                    }

                    ts_scal.left_count = left_count;

                    ts.left_hist[class_count - 1] = ts_scal.left_count - all_class_count;
                }
                else {
                    const Float val = (bin <= ts_scal.ftr_bin) ? response : Float(0);

                    Float left_count = Float(sycl::reduce_over_group(sbg, count, plus<Index>()));
                    Float sum = sycl::reduce_over_group(sbg, val, plus<Float>());

                    Float mean = sum / left_count;

                    const Float val_s2c =
                        (bin <= ts_scal.ftr_bin) ? (val - mean) * (val - mean) : Float(0);

                    Float sum2cent = sycl::reduce_over_group(sbg, val_s2c, plus<Float>());

                    reduce_hist_over_group(item, local_buf_float_ptr, left_count, mean, sum2cent);

                    ts_scal.left_count = Index(left_count);

                    ts.left_hist[0] = left_count;
                    ts.left_hist[1] = mean;
                    ts.left_hist[2] = sum2cent;
                }

                if (local_id == 0) {
                    if constexpr (std::is_same_v<Task, task::classification>) {
                        sp_hlp.calc_imp_dec(ts,
                                            node_ptr,
                                            node_imp_list_ptr,
                                            class_hist_list_ptr,
                                            class_count,
                                            node_id,
                                            false);
                        sp_hlp.choose_best_split(bs, ts, class_count, min_obs_leaf);
                    }
                    else {
                        sp_hlp.calc_imp_dec(ts, node_ptr, node_imp_list_ptr, node_id, false);
                        sp_hlp.choose_best_split(bs,
                                                 ts,
                                                 impl_const_t::hist_prop_count_,
                                                 min_obs_leaf);
                    }
                }
            }

            if (local_id == 0) {
                sp_hlp.update_node_bs_info(bs,
                                           node_ptr,
                                           node_imp_decr_list_ptr,
                                           node_id,
                                           index_max,
                                           update_imp_dec_required);
                if constexpr (std::is_same_v<Task, task::classification>) {
                    sp_hlp.update_left_child_imp(left_child_imp_list_ptr,
                                                 left_child_class_hist_list_ptr,
                                                 bs_scal.left_imp,
                                                 bs.left_hist,
                                                 node_id,
                                                 class_count);
                }
                else {
                    sp_hlp.update_left_child_imp(left_child_imp_list_ptr, bs.left_hist, node_id);
                }
            }
        });
    });

    last_event.wait_and_throw();
    return last_event;
}

// Computes possible block size for kernel according histogram size and device
template <typename hist_type_t, typename Index, typename Float, typename Task>
inline Index compute_bin_block_size(sycl::queue& queue, Index hist_prop_count, Index bin_count) {
    using split_scalar_t = split_scalar<Float, Index, Task>;
    using msg = de::error_messages;

    auto device = queue.get_device();
    std::int64_t device_local_mem_size =
        device.get_info<sycl::info::device::local_mem_size>() * 0.8;
    std::int64_t hist_size = hist_prop_count * sizeof(hist_type_t);
    std::int64_t bin_mem_size = hist_size + sizeof(Float) + sizeof(split_scalar_t);
    std::int64_t possible_block_size = device_local_mem_size / bin_mem_size;

    if constexpr (std::is_same_v<Task, task::regression>) {
        // Need to align block size to wg size for old GPU generations
        possible_block_size = std::min<Index>(possible_block_size, bk::device_max_wg_size(queue));
    }

    if (possible_block_size <= 0) {
        // Not enough memory to build at least one histogram
        throw domain_error(msg::not_enough_local_memory_for_hist());
    }

    return std::min<Index>(possible_block_size, bin_count);
}

// Cleans local memory before histogram calculation.
template <typename hist_type_t, typename Index, typename Float>
inline void clean_local_hists(hist_type_t* hist,
                              Float* weights,
                              Index id,
                              Index wg_size,
                              Index hist_prop_count,
                              Index bin_block) {
    for (Index i = id; i < bin_block * hist_prop_count; i += wg_size) {
        hist[i] = hist_type_t(0);
    }
    for (Index i = id; i < bin_block; i += wg_size) {
        weights[i] = Float(0);
    }
}

// Additional structure to use data arrays all-in-one
template <typename Index, typename Float, typename Bin>
struct input_data_arrays {
    const Bin* data_;
    const Index* order_;
    const Float* response_;
    const Float* weight_;
    input_data_arrays(const pr::ndview<Bin, 2>& data,
                      const pr::ndview<Index, 1>& tree_order,
                      const pr::ndview<Float, 1>& response,
                      const pr::ndview<Float, 1>& weights)
            : data_(data.get_data()),
              order_(tree_order.get_data()),
              response_(response.get_data()),
              weight_(weights.get_data()) {}
};

// Calculates histogram in GPU kernel.
// Contains both classification and regression cases.
template <typename Index, typename Float, typename Task, typename Bin, typename hist_type_t>
inline void compute_histogram(const local_accessor_rw_t<hist_type_t>& hist,
                              const local_accessor_rw_t<Float>& local_weight,
                              const sycl::nd_item<3>& item,
                              const train_context<Float, Index, Task>& ctx,
                              const Index act_bin_block,
                              const Index bin_ofs,
                              const Index row_ofs,
                              const Index row_count,
                              const Index ts_ftr_id,
                              const Index hist_prop_count,
                              const input_data_arrays<Index, Float, Bin>& data) {
    const Index id = item.get_local_id(2);
    const Index local_size = item.get_local_range(2);
    hist_type_t* const local_hist =
        hist.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
    const auto column_count = ctx.column_count_;
    const auto is_weighted = ctx.is_weighted_;
    if constexpr (std::is_same_v<Task, task::classification>) {
        // Classification case
        for (Index row_idx = id; row_idx < row_count; row_idx += local_size) {
            const Index id = data.order_[row_ofs + row_idx];
            const Index bin = data.data_[id * column_count + ts_ftr_id];
            const Index response_int = static_cast<Index>(data.response_[id]);
            const Index start = sycl::max(0, bin - bin_ofs);
            for (Index bin_id = start; bin_id < act_bin_block; ++bin_id) {
                const Index loc_bin_pos = bin_id * hist_prop_count;
                sycl::atomic_ref<Index,
                                 sycl::memory_order_relaxed,
                                 sycl::memory_scope_work_group,
                                 sycl::access::address_space::local_space>
                    hist_resp(hist[loc_bin_pos + response_int]);
                hist_resp += 1;
                if (is_weighted) {
                    sycl::atomic_ref<Float,
                                     sycl::memory_order_relaxed,
                                     sycl::memory_scope_work_group,
                                     sycl::access::address_space::local_space>
                        hist_weight(local_weight[bin_id]);
                    hist_weight += data.weight_[id];
                }
            }
        }
    }
    else {
        // Regression case
        const Index work_size = local_size / act_bin_block;
        Index count = 0;
        Float sum = 0;
        Float weight = 0;
        const Index bin_id = id % act_bin_block;
        const Index loc_bin_pos = bin_id * hist_prop_count;
        for (Index row_idx = id / act_bin_block; row_idx < row_count; row_idx += work_size) {
            const Index id = data.order_[row_ofs + row_idx];
            const Index bin = data.data_[id * column_count + ts_ftr_id];
            const Float response = data.response_[id];
            if ((bin_id + bin_ofs) >= bin) {
                count++;
                sum += response;
                if (is_weighted) {
                    weight += data.weight_[id];
                }
            }
        }
        if (id < work_size * act_bin_block) {
            sycl::atomic_ref<Float,
                             sycl::memory_order_relaxed,
                             sycl::memory_scope_work_group,
                             sycl::access::address_space::local_space>
                hist_count(hist[loc_bin_pos + 0]);
            sycl::atomic_ref<Float,
                             sycl::memory_order_relaxed,
                             sycl::memory_scope_work_group,
                             sycl::access::address_space::local_space>
                hist_sum(hist[loc_bin_pos + 1]);
            hist_count += count;
            hist_sum += static_cast<Float>(sum);
            if (is_weighted) {
                sycl::atomic_ref<Float,
                                 sycl::memory_order_relaxed,
                                 sycl::memory_scope_work_group,
                                 sycl::access::address_space::local_space>
                    hist_weight(local_weight[bin_id]);
                hist_weight += weight;
            }
        }
        // Finalize regression case by calculating MSE
        item.barrier(sycl::access::fence_space::local_space);
        Float mse = 0;
        const Float mean = local_hist[loc_bin_pos + 1] / local_hist[loc_bin_pos + 0];
        for (Index row_idx = id / act_bin_block; row_idx < row_count; row_idx += work_size) {
            const Index id = data.order_[row_ofs + row_idx];
            const Index bin = data.data_[id * column_count + ts_ftr_id];
            const Float response = data.response_[id];
            if ((bin_id + bin_ofs) >= bin) {
                mse += (response - mean) * (response - mean);
            }
        }
        if (id < work_size * act_bin_block) {
            sycl::atomic_ref<Float,
                             sycl::memory_order_relaxed,
                             sycl::memory_scope_work_group,
                             sycl::access::address_space::local_space>
                hist_mse(hist[loc_bin_pos + 2]);
            hist_mse += static_cast<Float>(mse);
        }
    }
}

/// Best splitter kernel
/// Kernel is consist of 2 kernel.
/// First kernel calculates histograms for each (node, feature, bin) and selects
/// best split among all bins for one feature in terms of impurity decrease.
/// Second kernel merges results of first kernel and selects best split among
/// all features in one node.
///
template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_splitter_impl<Float, Bin, Index, Task>::best_split(
    sycl::queue& queue,
    const context_t& ctx,
    const pr::ndview<Bin, 2>& data,
    const pr::ndview<Float, 1>& response,
    const pr::ndview<Float, 1>& weights,
    const pr::ndview<Index, 1>& tree_order,
    const pr::ndview<Index, 1>& selected_ftr_list,
    const pr::ndview<Index, 1>& bin_offset_list,
    const imp_data_t& imp_data_list,
    pr::ndview<Index, 1>& node_list,
    imp_data_t& left_child_imp_data_list,
    pr::ndview<Float, 1>& node_imp_dec_list,
    bool update_imp_dec_required,
    Index node_count,
    const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(best_split, queue);
    ONEDAL_ASSERT(data.get_count() == ctx.row_count_ * ctx.column_count_);
    ONEDAL_ASSERT(response.get_count() == ctx.row_count_);
    ONEDAL_ASSERT(tree_order.get_count() == ctx.tree_in_block_ * ctx.selected_row_total_count_);
    ONEDAL_ASSERT(selected_ftr_list.get_count() >= node_count * ctx.selected_ftr_count_);
    ONEDAL_ASSERT(bin_offset_list.get_count() == ctx.column_count_ + 1);
    ONEDAL_ASSERT(imp_data_list.imp_list_.get_count() >=
                  node_count * impl_const_t::node_imp_prop_count_);

    const input_data_arrays arr(data, tree_order, response, weights);

    const Index* const selected_ftr_list_ptr = selected_ftr_list.get_data();

    imp_data_list_ptr<Float, Index, Task> imp_list_ptr(imp_data_list);

    Index* node_list_ptr = node_list.get_mutable_data();
    Float* node_imp_decr_list_ptr =
        update_imp_dec_required ? node_imp_dec_list.get_mutable_data() : nullptr;

    imp_data_list_ptr_mutable<Float, Index, Task> left_imp_list_ptr(left_child_imp_data_list);

    const Float* const node_imp_list_ptr = imp_list_ptr.imp_list_ptr_;
    Float* const left_child_imp_list_ptr = left_imp_list_ptr.imp_list_ptr_;

    using split_smp_t = split_smp<Float, Index, Task>;
    using split_scalar_t = split_scalar<Float, Index, Task>;
    using split_info_t = split_info<Float, Index, Task>;

    Index hist_prop_count = 0;
    if constexpr (std::is_same_v<std::decay_t<Task>, task::classification>) {
        hist_prop_count = ctx.class_count_;
    }
    else {
        hist_prop_count = impl_const<Index, task::regression>::hist_prop_count_;
    }

    const Float imp_threshold = ctx.impurity_threshold_;
    const Index min_obs_leaf = ctx.min_observations_in_leaf_node_;
    const Index index_max = ctx.index_max_;
    const bool is_weighted = ctx.is_weighted_;

    // following vars are not used for regression, but should present to compile kernel
    const Index* class_hist_list_ptr = imp_list_ptr.get_class_hist_list_ptr_or_null();
    Index* left_child_class_hist_list_ptr = left_imp_list_ptr.get_class_hist_list_ptr_or_null();

    const Index ftr_count = ctx.selected_ftr_count_;
    const Index bin_count = ctx.max_bin_count_among_ftrs_;
    ONEDAL_ASSERT(bin_count > 1);

    const Index bin_block =
        compute_bin_block_size<hist_type_t, Index, Float, Task>(queue, hist_prop_count, bin_count);

    const Index local_size_initial = bk::device_max_wg_size(queue);
    Index local_size = local_size_initial;
    const auto max_int_limit = std::numeric_limits<int>::max();

    if (node_count * ftr_count > 0 && node_count * ftr_count <= max_int_limit) {
        while (node_count * ftr_count * local_size > max_int_limit) {
            local_size /= 2;
        }
    }
    else {
        std::cerr << "Error: node_count * ftr_count exceeds int limit" << std::endl;
    }

    std::cout << "node count = " << node_count << std::endl;
    std::cout << "ftr_count = " << ftr_count << std::endl;
    std::cout << "local_size = " << local_size << std::endl;
    std::cout << "total range size = " << node_count * ftr_count * local_size << std::endl;

    const auto nd_range =
        bk::make_multiple_nd_range_3d({ node_count, ftr_count, local_size }, { 1, 1, local_size });

    const auto best_ftr_splits =
        pr::ndarray<split_scalar_t, 1>::empty(queue, { node_count * ftr_count }, alloc::device);
    const auto splits_ptr = best_ftr_splits.get_mutable_data();

    const auto best_ftr_hists =
        pr::ndarray<hist_type_t, 1>::empty(queue,
                                           { node_count * ftr_count * hist_prop_count },
                                           alloc::device);
    const auto hists_ptr = best_ftr_hists.get_mutable_data();

    // Main kernel:
    // calculates histograms and impurity decrease based on histograms
    // and selects best split for each feature.
    sycl::event last_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        local_accessor_rw_t<hist_type_t> hist(bin_block * hist_prop_count, cgh);
        local_accessor_rw_t<split_scalar_t> scalars(bin_block, cgh);
        local_accessor_rw_t<Float> l_weight(bin_block, cgh);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<3> item) {
            const Index node_id = item.get_global_id(0);
            const Index ftr_id = item.get_global_id(1);
            const Index local_id = item.get_local_id(2);
            Index* node_ptr = node_list_ptr + node_id * impl_const_t::node_prop_count_;
            const Index row_ofs = node_ptr[impl_const_t::ind_ofs];
            const Index row_count = node_ptr[impl_const_t::ind_lrc];

            const Index ftr_position = node_id * ftr_count + ftr_id;
            hist_type_t* const ftr_hist = hists_ptr + ftr_position * hist_prop_count;

            const Index ts_ftr_id = selected_ftr_list_ptr[ftr_position];
            // Clean global data
            // In case of early stop it will be clean and can be processed correctly
            // in merging kernel.
            if (local_id == 0) {
                splits_ptr[ftr_position].clear();
            }

            for (Index idx = local_id; idx < hist_prop_count; idx += local_size) {
                ftr_hist[idx] = 0;
            }

            split_smp_t sp_hlp;
            // Check node impurity
            if (!sp_hlp.is_valid_impurity(node_imp_list_ptr, node_id, imp_threshold, row_count)) {
                return;
            }
            if (row_count < 2 * min_obs_leaf) {
                return;
            }
            hist_type_t* const local_hist =
                hist.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
            split_scalar_t* const local_scalars =
                scalars.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
            Float* const local_weight =
                l_weight.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();

            for (Index bin_ofs = 0; bin_ofs < bin_count; bin_ofs += bin_block) {
                // Clean histogram before calculating
                clean_local_hists(local_hist,
                                  local_weight,
                                  local_id,
                                  local_size,
                                  hist_prop_count,
                                  bin_block);
                item.barrier(sycl::access::fence_space::local_space);
                // Calculate histogram for bin block
                const Index act_bin_block = sycl::min(bin_block, bin_count - bin_ofs);
                compute_histogram<Index, Float, Task, Bin, hist_type_t>(hist,
                                                                        l_weight,
                                                                        item,
                                                                        ctx,
                                                                        act_bin_block,
                                                                        bin_ofs,
                                                                        row_ofs,
                                                                        row_count,
                                                                        ts_ftr_id,
                                                                        hist_prop_count,
                                                                        arr);
                // Wait until histogram computing will be finished
                item.barrier(sycl::access::fence_space::local_space);
                // Calculate impurity decrease for block of bins
                if (local_id < act_bin_block) {
                    auto cur_hist = local_hist + local_id * hist_prop_count;
                    split_info_t ts;
                    ts.scalars.clear();
                    ts.scalars.ftr_id = ts_ftr_id;
                    ts.scalars.ftr_bin = local_id + bin_ofs;
                    if (is_weighted) {
                        ts.scalars.left_weight_sum = local_weight[local_id];
                    }
                    ts.init(cur_hist, hist_prop_count);
                    if constexpr (std::is_same_v<Task, task::classification>) {
                        Index left_count = 0;
                        for (Index idx = 0; idx < hist_prop_count; ++idx) {
                            left_count += cur_hist[idx];
                        }
                        ts.scalars.left_count = left_count;
                        sp_hlp.calc_imp_dec(ts,
                                            node_ptr,
                                            node_imp_list_ptr,
                                            class_hist_list_ptr,
                                            hist_prop_count,
                                            node_id,
                                            is_weighted);
                    }
                    else {
                        ts.scalars.left_count = Index(cur_hist[0]);
                        cur_hist[1] /= cur_hist[0];
                        sp_hlp.calc_imp_dec(ts, node_ptr, node_imp_list_ptr, node_id, is_weighted);
                    }
                    split_scalar_t& scal = local_scalars[local_id];
                    scal.clear();
                    if (sp_hlp.test_split_is_best(scal, ts.scalars, min_obs_leaf)) {
                        scal.copy(ts.scalars);
                    }
                }
                // Select best split among bin block
                for (Index i = act_bin_block / 2; i > 0; i >>= 1) {
                    item.barrier(sycl::access::fence_space::local_space);
                    if (local_id < i && (local_id + i) < act_bin_block) {
                        split_scalar_t& s1 = local_scalars[local_id];
                        split_scalar_t& s2 = local_scalars[local_id + i];
                        if (sp_hlp.test_split_is_best(s1, s2, min_obs_leaf)) {
                            s1.copy(s2);
                        }
                    }
                }

                // Assign best split from block to global memory
                if (local_id == 0) {
                    split_scalar_t& best_split = splits_ptr[ftr_position];
                    split_scalar_t& block_split = local_scalars[0];
                    if (sp_hlp.test_split_is_best(best_split, block_split, min_obs_leaf)) {
                        best_split.copy(block_split);
                        const Index best_hist_pos = block_split.ftr_bin - bin_ofs;
                        auto cur_hist = local_hist + best_hist_pos * hist_prop_count;
                        for (Index idx = 0; idx < hist_prop_count; ++idx) {
                            ftr_hist[idx] = cur_hist[idx];
                        }
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);
            }
        });
    });
    // Merging kernel: selects best split among all features.
    const auto merge_range =
        bk::make_multiple_nd_range_2d({ node_count, local_size }, { 1, local_size });
    last_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ last_event });
        local_accessor_rw_t<Index> ftr_ids(local_size, cgh);
        cgh.parallel_for(merge_range, [=](sycl::nd_item<2> item) {
            const Index node_id = item.get_global_id(0);
            const Index local_id = item.get_local_id(1);
            Index* node_ptr = node_list_ptr + node_id * impl_const_t::node_prop_count_;
            const Index row_count = node_ptr[impl_const_t::ind_lrc];
            split_smp_t sp_hlp;
            // Check node impurity
            if (!sp_hlp.is_valid_impurity(node_imp_list_ptr, node_id, imp_threshold, row_count)) {
                return;
            }
            auto node_splits = splits_ptr + node_id * ftr_count;
            Index* const ftr_indices =
                ftr_ids.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();

            ftr_indices[local_id] = local_id;
            item.barrier(sycl::access::fence_space::local_space);

            // Select best split inside one working item
            for (Index ftr_id = local_id; ftr_id < ftr_count; ftr_id += local_size) {
                split_scalar_t& s1 = node_splits[local_id];
                split_scalar_t& s2 = node_splits[ftr_id];
                if (sp_hlp.test_split_is_best(s1, s2, min_obs_leaf)) {
                    s1.copy(s2);
                    ftr_indices[local_id] = ftr_id;
                }
            }

            // Select best split among working group
            for (Index i = local_size / 2; i > 0; i >>= 1) {
                item.barrier(sycl::access::fence_space::local_space);
                if (local_id < i && (local_id + i) < ftr_count) {
                    split_scalar_t& s1 = node_splits[local_id];
                    split_scalar_t& s2 = node_splits[local_id + i];
                    if (sp_hlp.test_split_is_best(s1, s2, min_obs_leaf)) {
                        s1.copy(s2);
                        ftr_indices[local_id] = ftr_indices[local_id + i];
                    }
                }
            }

            // Assign best split mong all features to global result
            if (local_id == 0) {
                split_info_t bs;
                bs.scalars.copy(node_splits[0]);
                const auto hist =
                    hists_ptr + (node_id * ftr_count + ftr_indices[0]) * hist_prop_count;
                bs.init(hist, hist_prop_count);
                sp_hlp.update_node_bs_info(bs,
                                           node_ptr,
                                           node_imp_decr_list_ptr,
                                           node_id,
                                           index_max,
                                           update_imp_dec_required);
                if constexpr (std::is_same_v<Task, task::classification>) {
                    sp_hlp.update_left_child_imp(left_child_imp_list_ptr,
                                                 left_child_class_hist_list_ptr,
                                                 bs.scalars.left_imp,
                                                 bs.left_hist,
                                                 node_id,
                                                 hist_prop_count);
                }
                else {
                    sp_hlp.update_left_child_imp(left_child_imp_list_ptr, bs.left_hist, node_id);
                }
            }
        });
    });

    last_event.wait_and_throw();
    return last_event;
}

#define INSTANTIATE(F, B, I, T) template class train_splitter_impl<F, B, I, T>;

INSTANTIATE(float, std::uint32_t, std::int32_t, task::classification);
INSTANTIATE(float, std::uint32_t, std::int32_t, task::regression);

INSTANTIATE(double, std::uint32_t, std::int32_t, task::classification);
INSTANTIATE(double, std::uint32_t, std::int32_t, task::regression);

} // namespace oneapi::dal::decision_forest::backend

#endif
