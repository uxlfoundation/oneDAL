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

#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_helpers.hpp"

#ifdef ONEDAL_DATA_PARALLEL

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_kernel_hist_impl.hpp"

namespace oneapi::dal::decision_forest::backend {

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

using alloc = sycl::usm::alloc;
using address = sycl::access::address_space;

using sycl::ext::oneapi::plus;
using sycl::ext::oneapi::minimum;
using sycl::ext::oneapi::maximum;

template <typename T>
using enable_if_float_t = std::enable_if_t<detail::is_valid_float_v<T>>;

template <typename Data>
using local_accessor_rw_t = sycl::local_accessor<Data, 1>;

template <typename F>
struct float_accuracy;

template <>
struct float_accuracy<float> {
    static constexpr float val = float(1e-5);
};

template <>
struct float_accuracy<double> {
    static constexpr double val = double(1e-10);
};

template <typename Float, typename Bin, typename Index, typename Task>
void train_kernel_hist_impl<Float, Bin, Index, Task>::validate_input(const descriptor_t& desc,
                                                                     const table& data,
                                                                     const table& labels) const {
    if (data.get_row_count() > de::limits<Index>::max()) {
        throw domain_error(msg::invalid_range_of_rows());
    }
    if (data.get_column_count() > de::limits<Index>::max()) {
        throw domain_error(msg::invalid_range_of_columns());
    }
    if (labels.get_row_count() != data.get_row_count()) {
        throw domain_error(msg::invalid_range_of_rows());
    }
    if (desc.get_tree_count() > de::limits<Index>::max()) {
        throw domain_error(msg::invalid_number_of_trees());
    }
    if (desc.get_min_observations_in_leaf_node() > de::limits<Index>::max()) {
        throw domain_error(msg::invalid_number_of_min_observations_in_leaf_node());
    }
    if (desc.get_features_per_node() > de::limits<Index>::max()) {
        throw domain_error(msg::invalid_number_of_feature_per_node());
    }
    if (desc.get_max_bins() > de::limits<Index>::max()) {
        throw domain_error(msg::invalid_number_of_max_bins());
    }
    if (desc.get_min_bin_size() > de::limits<Index>::max()) {
        throw domain_error(msg::invalid_value_for_min_bin_size());
    }
    if constexpr (std::is_same_v<Task, task::classification>) {
        if (desc.get_class_count() > de::limits<Index>::max()) {
            throw domain_error(msg::invalid_number_of_classes());
        }
    }
}

template <typename Float, typename Bin, typename Index, typename Task>
Index train_kernel_hist_impl<Float, Bin, Index, Task>::get_row_total_count(bool distr_mode,
                                                                           Index row_count) {
    Index row_total_count = row_count;

    if (distr_mode) {
        ONEDAL_PROFILER_TASK(allreduce_row_total_count);
        comm_.allreduce(row_total_count).wait();
    }

    return row_total_count;
}

template <typename Float, typename Bin, typename Index, typename Task>
Index train_kernel_hist_impl<Float, Bin, Index, Task>::get_global_row_offset(bool distr_mode,
                                                                             Index row_count) {
    Index global_row_offset = 0;

    if (distr_mode) {
        auto row_count_list_host = pr::ndarray<Index, 1>::empty({ comm_.get_rank_count() });
        Index* row_count_list_host_ptr = row_count_list_host.get_mutable_data();
        {
            ONEDAL_PROFILER_TASK(allgather_row_count);
            comm_.allgather(row_count, row_count_list_host.flatten()).wait();
        }

        for (std::int64_t i = 0; i < comm_.get_rank(); ++i) {
            global_row_offset += row_count_list_host_ptr[i];
        }
    }

    return global_row_offset;
}

template <typename Float, typename Bin, typename Index, typename Task>
void train_kernel_hist_impl<Float, Bin, Index, Task>::init_params(train_context_t& ctx,
                                                                  const descriptor_t& desc,
                                                                  const table& data,
                                                                  const table& responses,
                                                                  const table& weights) {
    ctx.distr_mode_ = (comm_.get_rank_count() > 1);

    ctx.use_private_mem_buf_ = true;

    ctx.is_weighted_ = (weights.get_row_count() == data.get_row_count());

    if constexpr (std::is_same_v<Task, task::classification>) {
        ctx.class_count_ = de::integral_cast<Index>(desc.get_class_count());
        ctx.use_private_mem_buf_ = false;
    }

    ctx.row_count_ = de::integral_cast<Index>(data.get_row_count());
    ctx.row_total_count_ = get_row_total_count(ctx.distr_mode_, ctx.row_count_);

    ctx.column_count_ = de::integral_cast<Index>(data.get_column_count());

    // in case of distributed mode selected_row_count is defined during initial gen of tree order
    ctx.selected_row_count_ = ctx.distr_mode_
                                  ? impl_const_t::bad_val_
                                  : desc.get_observations_per_tree_fraction() * ctx.row_count_;
    ctx.selected_row_total_count_ =
        desc.get_observations_per_tree_fraction() * ctx.row_total_count_;

    ctx.global_row_offset_ = get_global_row_offset(ctx.distr_mode_, ctx.row_count_);

    ctx.tree_count_ = de::integral_cast<Index>(desc.get_tree_count());

    ctx.bootstrap_ = desc.get_bootstrap();
    ctx.max_tree_depth_ = desc.get_max_tree_depth();

    ctx.splitter_mode_value_ = desc.get_splitter_mode();
    ctx.seed_ = desc.get_seed();

    if constexpr (std::is_same_v<Task, task::classification>) {
        ctx.selected_ftr_count_ = desc.get_features_per_node() ? desc.get_features_per_node()
                                                               : std::sqrt(ctx.column_count_);
    }
    else {
        ctx.selected_ftr_count_ = desc.get_features_per_node() ? desc.get_features_per_node()
                                  : ctx.column_count_ / 3      ? ctx.column_count_ / 3
                                                               : 1;
    }
    ctx.min_observations_in_leaf_node_ = desc.get_min_observations_in_leaf_node();
    ctx.impurity_threshold_ = desc.get_impurity_threshold();

    if (0 >= ctx.selected_row_total_count_) {
        throw domain_error(msg::invalid_value_for_observations_per_tree_fraction());
    }

    auto vimp = desc.get_variable_importance_mode();
    ctx.mda_required_ =
        (variable_importance_mode::mda_raw == vimp || variable_importance_mode::mda_scaled == vimp);
    ctx.mdi_required_ = variable_importance_mode::mdi == vimp;
    ctx.mda_scaled_required_ = (variable_importance_mode::mda_scaled == vimp);

    auto emm = desc.get_error_metric_mode();
    ctx.oob_required_ =
        (check_mask_flag(emm, error_metric_mode::out_of_bag_error) ||
         check_mask_flag(emm, error_metric_mode::out_of_bag_error_per_observation) ||
         ctx.mda_required_);
    ctx.oob_err_required_ = check_mask_flag(emm, error_metric_mode::out_of_bag_error);
    ctx.oob_err_obs_required_ =
        check_mask_flag(emm, error_metric_mode::out_of_bag_error_per_observation);

    // init ftr -> bins map and related params
    indexed_features<Float, Bin, Index> ind_ftrs(queue_,
                                                 comm_,
                                                 desc.get_min_bin_size(),
                                                 desc.get_max_bins());
    ind_ftrs(data).wait_and_throw();

    ctx.total_bin_count_ = ind_ftrs.get_total_bin_count();
    full_data_nd_ = ind_ftrs.get_full_data();
    ftr_bin_offsets_nd_ = ind_ftrs.get_bin_offsets();

    bin_borders_host_.resize(ctx.column_count_);
    for (Index clmn_idx = 0; clmn_idx < ctx.column_count_; ++clmn_idx) {
        bin_borders_host_[clmn_idx] = ind_ftrs.get_bin_borders(clmn_idx).to_host(queue_);
    }

    data_host_ = pr::table2ndarray_1d<Float>(queue_, data, alloc::device).to_host(queue_);

    response_nd_ = pr::table2ndarray_1d<Float>(queue_, responses, alloc::device);

    if (ctx.is_weighted_) {
        weights_nd_ = pr::table2ndarray_1d<Float>(queue_, weights, alloc::device);
    }

    response_host_ = response_nd_.to_host(queue_);

    // calculating the maximal number of bins for feature among all features
    ctx.max_bin_count_among_ftrs_ = 0;
    for (Index clmn_idx = 0; clmn_idx < ctx.column_count_; ++clmn_idx) {
        auto ftr_bins = ind_ftrs.get_bin_count(clmn_idx);
        ctx.max_bin_count_among_ftrs_ = std::max(ctx.max_bin_count_among_ftrs_, ftr_bins);
    }

    ctx.float_min_ = de::limits<Float>::min();
    ctx.index_max_ = de::limits<Index>::max();

    // define number of trees which can be built in parallel
    const std::uint64_t device_global_mem_size =
        queue_.get_device().get_info<sycl::info::device::global_mem_size>();
    const std::uint64_t device_max_mem_alloc_size =
        queue_.get_device().get_info<sycl::info::device::max_mem_alloc_size>();

    Index hist_prop_count = 0;
    if constexpr (std::is_same_v<task::classification, Task>) {
        hist_prop_count = ctx.class_count_;
    }
    else {
        hist_prop_count = impl_const_t::hist_prop_count_;
    }

    std::uint64_t used_mem_size =
        sizeof(Float) * ctx.row_count_ * (ctx.column_count_ + 1); // input table size + response
    used_mem_size +=
        ind_ftrs.get_required_mem_size(ctx.row_count_, ctx.column_count_, desc.get_max_bins());
    used_mem_size += ctx.oob_required_ ? sizeof(Float) * ctx.row_count_ * ctx.class_count_ : 0;

    std::uint64_t available_global_mem_size =
        device_global_mem_size > used_mem_size ? device_global_mem_size - used_mem_size : 0;

    std::uint64_t available_mem_size_for_tree_block =
        std::min(device_max_mem_alloc_size, static_cast<std::uint64_t>(available_global_mem_size));

    std::uint64_t required_mem_size_for_one_tree =
        ctx.oob_required_ ? train_service_kernels_.get_oob_rows_required_mem_size(
                                ctx.row_count_,
                                1 /* for 1 tree */,
                                desc.get_observations_per_tree_fraction())
                          : 0;

    // two buffers for row indices for each tree
    required_mem_size_for_one_tree += sizeof(Index) * ctx.selected_row_total_count_ * 2;

    // Max node_count in tree = last level
    // If depth is set to zero or less than 2, then it is limited by row_count or set to 2
    // TODO: replace this block with std::log2() and std::exp2(),
    // when the compiler is ready for use it.
    std::int64_t data_tree_depth = ctx.max_tree_depth_;
    if (data_tree_depth == 0) {
        std::int64_t row_count = ctx.row_count_;
        while (row_count > 1) { // std::log2(row_count)
            row_count = row_count >> 1;
            data_tree_depth++;
        }
        data_tree_depth = std::max<std::int64_t>(2, data_tree_depth);
    }
    std::uint64_t max_node_count_per_tree = 1; // std::exp2(data_tree_depth - 2);
    for (std::int32_t i = 0; i < data_tree_depth - 2; ++i) {
        max_node_count_per_tree *= 2;
    }
    // node_lists for one tree
    required_mem_size_for_one_tree +=
        de::check_mul_overflow(sizeof(Index) * impl_const_t::node_prop_count_,
                               max_node_count_per_tree);
    // node_vs_tree_map_list structure
    required_mem_size_for_one_tree +=
        de::check_mul_overflow(sizeof(Index), max_node_count_per_tree);
    // Selected features and random bin tresholds
    required_mem_size_for_one_tree +=
        de::check_mul_overflow((sizeof(Index) + sizeof(Float)) * ctx.selected_ftr_count_,
                               max_node_count_per_tree);
    // Impurity data for each node
    required_mem_size_for_one_tree +=
        de::check_mul_overflow(sizeof(Float) * impl_const_t::node_imp_prop_count_,
                               max_node_count_per_tree);

    // // Internal scalars to find best split
    required_mem_size_for_one_tree +=
        max_node_count_per_tree * ctx.selected_ftr_count_ * 5 * (sizeof(Float) + sizeof(Index));
    required_mem_size_for_one_tree +=
        max_node_count_per_tree * ctx.selected_ftr_count_ * hist_prop_count * sizeof(hist_type_t);
    // Impurity decrease list
    if (ctx.mdi_required_) {
        required_mem_size_for_one_tree += sizeof(Float) * max_node_count_per_tree;
    }

    ctx.tree_in_block_ = de::integral_cast<Index>(available_mem_size_for_tree_block /
                                                  required_mem_size_for_one_tree);
    if (ctx.tree_in_block_ <= 0) {
        // not enough memory even for one tree
        throw domain_error(msg::not_enough_memory_to_build_one_tree());
    }

    ctx.tree_in_block_ = std::min(ctx.tree_count_, ctx.tree_in_block_);

    available_global_mem_size =
        available_global_mem_size > (ctx.tree_in_block_ * required_mem_size_for_one_tree)
            ? available_global_mem_size - (ctx.tree_in_block_ * required_mem_size_for_one_tree)
            : 0;

    ctx.oob_prop_count_ = impl_const_t::oob_aux_prop_count_;
    if constexpr (std::is_same_v<Task, task::classification>) {
        ctx.oob_prop_count_ = ctx.class_count_;
    }
}

template <typename Float, typename Bin, typename Index, typename Task>
void train_kernel_hist_impl<Float, Bin, Index, Task>::allocate_buffers(const train_context_t& ctx) {
    de::check_mul_overflow(ctx.selected_row_total_count_, ctx.tree_in_block_);

    // main tree order and auxilliary one are used for partitioning
    tree_order_lev_ =
        pr::ndarray<Index, 1>::empty(queue_,
                                     { ctx.selected_row_total_count_ * ctx.tree_in_block_ },
                                     alloc::device);
    tree_order_lev_buf_ =
        pr::ndarray<Index, 1>::empty(queue_,
                                     { ctx.selected_row_total_count_ * ctx.tree_in_block_ },
                                     alloc::device);
    if (ctx.oob_required_) {
        // oob_per_obs_list contains class_count number of counters for all out of bag observations for all trees
        de::check_mul_overflow(ctx.row_count_, ctx.class_count_);
        auto [oob_per_obs_list, event] =
            pr::ndarray<hist_type_t, 1>::zeros(queue_,
                                               { ctx.row_count_ * ctx.oob_prop_count_ },
                                               alloc::device);
        oob_per_obs_list_ = oob_per_obs_list;
        event.wait_and_throw();
    }

    // blocks for MDA scaled error calculation
    if (ctx.mda_scaled_required_) {
        var_imp_variance_host_ = pr::ndarray<Float, 1>::zeros({ ctx.column_count_ });
    }

    if (ctx.mdi_required_ || ctx.mda_required_) {
        res_var_imp_ = pr::ndarray<Float, 1>::empty(queue_, { ctx.column_count_ }, alloc::device);
        res_var_imp_.fill(queue_, 0);
    }
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::gen_initial_tree_order(
    train_context_t& ctx,
    rng_engine_list_t& rng_engine_list,
    pr::ndarray<Index, 1>& node_list_host,
    pr::ndarray<Index, 1>& tree_order_level,
    Index engine_offset,
    Index node_count) {
    ONEDAL_PROFILER_TASK(gen_initial_tree_order, queue_);

    ONEDAL_ASSERT(node_list_host.get_count() == node_count * impl_const_t::node_prop_count_);
    ONEDAL_ASSERT(tree_order_level.get_count() ==
                  ctx.tree_in_block_ * ctx.selected_row_total_count_);

    sycl::event last_event;

    if (ctx.bootstrap_) {
        auto selected_row_global_host =
            pr::ndarray<Index, 1>::empty({ ctx.selected_row_total_count_ * ctx.tree_in_block_ });
        pr::ndarray<Index, 1> selected_row_host;
        if (ctx.distr_mode_) {
            selected_row_host = pr::ndarray<Index, 1>::empty(
                { ctx.selected_row_total_count_ * ctx.tree_in_block_ });
        }

        Index* const selected_row_global_ptr = selected_row_global_host.get_mutable_data();
        Index* const selected_row_ptr =
            ctx.distr_mode_ ? selected_row_host.get_mutable_data() : nullptr;
        Index* const node_list_ptr = node_list_host.get_mutable_data();

        for (Index node_idx = 0; node_idx < node_count; ++node_idx) {
            Index* gen_row_idx_global_ptr =
                selected_row_global_ptr + ctx.selected_row_total_count_ * node_idx;
            pr::uniform<Index>(ctx.selected_row_total_count_,
                               gen_row_idx_global_ptr,
                               rng_engine_list[engine_offset + node_idx],
                               0,
                               ctx.row_total_count_);

            if (ctx.distr_mode_) {
                Index* node_ptr = node_list_ptr + node_idx * impl_const_t::node_prop_count_;
                Index* src = gen_row_idx_global_ptr;

                Index* const dst = selected_row_ptr + ctx.selected_row_total_count_ * node_idx;

                Index row_idx = 0;
                for (Index i = 0; i < ctx.selected_row_total_count_; ++i) {
                    dst[i] = 0;
                    if (src[i] >= ctx.global_row_offset_ &&
                        src[i] < (ctx.global_row_offset_ + ctx.row_count_)) {
                        dst[row_idx++] = src[i] - ctx.global_row_offset_;
                    }
                }
                node_ptr[impl_const_t::ind_lrc] = row_idx;
            }
        }

        last_event = ctx.distr_mode_
                         ? tree_order_level.assign_from_host(queue_, selected_row_host)
                         : tree_order_level.assign_from_host(queue_, selected_row_global_host);
    }
    else {
        Index row_count = ctx.selected_row_count_;
        Index stride = ctx.selected_row_total_count_;
        if (ctx.distr_mode_) {
            row_count = 0;
            if (ctx.global_row_offset_ < ctx.selected_row_total_count_) {
                row_count = std::min(ctx.selected_row_total_count_ - ctx.global_row_offset_,
                                     ctx.row_count_);
            }
            // in case of no bootstrap
            // it is valid case if this worker's rows set wasn't taken for tree build
            // i.e. row_count can be eq 0

            Index* node_list_ptr = node_list_host.get_mutable_data();

            for (Index node_idx = 0; node_idx < node_count; ++node_idx) {
                Index* node_ptr = node_list_ptr + node_idx * impl_const_t::node_prop_count_;
                node_ptr[impl_const_t::ind_lrc] = row_count;
            }
        }

        if (row_count > 0) {
            last_event = train_service_kernels_.initialize_tree_order(tree_order_level,
                                                                      node_count,
                                                                      row_count,
                                                                      stride);
        }
    }

    return last_event;
}

template <typename Float, typename Bin, typename Index, typename Task>
std::tuple<pr::ndarray<Index, 1>, sycl::event>
train_kernel_hist_impl<Float, Bin, Index, Task>::gen_feature_list(
    const train_context_t& ctx,
    Index node_count,
    const pr::ndarray<Index, 1>& node_vs_tree_map_list,
    rng_engine_list_t& rng_engine_list) {
    ONEDAL_PROFILER_TASK(gen_feature_list, queue_);

    ONEDAL_ASSERT(node_vs_tree_map_list.get_count() == node_count);

    de::check_mul_overflow((node_count + 1), ctx.selected_ftr_count_);
    // first part is used for features indices, +1 block - part for generator
    auto selected_features_host =
        pr::ndarray<Index, 1>::empty({ (node_count + 1) * ctx.selected_ftr_count_ });
    auto selected_features_com =
        pr::ndarray<Index, 1>::empty(queue_,
                                     { node_count * ctx.selected_ftr_count_ },
                                     alloc::device);

    auto selected_features_host_ptr = selected_features_host.get_mutable_data();

    auto node_vs_tree_map_list_host = node_vs_tree_map_list.to_host(queue_);

    auto tree_map_ptr = node_vs_tree_map_list_host.get_mutable_data();
    if (ctx.selected_ftr_count_ != ctx.column_count_) {
        for (Index node = 0; node < node_count; ++node) {
            pr::uniform_without_replacement<Index>(
                ctx.selected_ftr_count_,
                selected_features_host_ptr + node * ctx.selected_ftr_count_,
                selected_features_host_ptr + (node + 1) * ctx.selected_ftr_count_,
                rng_engine_list[tree_map_ptr[node]],
                0,
                ctx.column_count_);
        }
    }
    else {
        for (Index node = 0; node < node_count; ++node) {
            for (Index i = 0; i < ctx.selected_ftr_count_; ++i) {
                selected_features_host_ptr[node * ctx.selected_ftr_count_ + i] = i;
            }
        }
    }

    auto event = selected_features_com.assign_from_host(queue_,
                                                        selected_features_host_ptr,
                                                        selected_features_com.get_count());

    return std::tuple{ selected_features_com, event };
}

template <typename Float, typename Bin, typename Index, typename Task>
std::tuple<pr::ndarray<Float, 1>, sycl::event>
train_kernel_hist_impl<Float, Bin, Index, Task>::gen_random_thresholds(
    const train_context_t& ctx,
    Index node_count,
    const pr::ndarray<Index, 1>& node_vs_tree_map,
    rng_engine_list_t& rng_engine_list) {
    ONEDAL_PROFILER_TASK(gen_random_thresholds, queue_);

    ONEDAL_ASSERT(node_vs_tree_map.get_count() == node_count);

    auto node_vs_tree_map_list_host = node_vs_tree_map.to_host(queue_);

    auto tree_map_ptr = node_vs_tree_map_list_host.get_mutable_data();

    // Create arrays for random generated bins
    auto random_bins_host =
        pr::ndarray<Float, 1>::empty(queue_, { node_count * ctx.selected_ftr_count_ });
    auto random_bins_com = pr::ndarray<Float, 1>::empty(queue_,
                                                        { node_count * ctx.selected_ftr_count_ },
                                                        alloc::device);
    auto random_bins_host_ptr = random_bins_host.get_mutable_data();

    // Generate random bins for selected features
    for (Index node = 0; node < node_count; ++node) {
        pr::uniform<Float>(ctx.selected_ftr_count_,
                           random_bins_host_ptr + node * ctx.selected_ftr_count_,
                           rng_engine_list[tree_map_ptr[node]],
                           0.0f,
                           1.0f);
    }
    auto event_rnd_generate =
        random_bins_com.assign_from_host(queue_, random_bins_host_ptr, random_bins_com.get_count());

    return std::tuple{ random_bins_com, event_rnd_generate };
};

template <typename Float, typename Index, typename Task>
struct kernel_context {
    kernel_context(const train_context<Float, Index, Task>& ctx)
            : float_min_(ctx.float_min_),
              index_max_(ctx.index_max_),
              min_observations_in_leaf_node_(ctx.min_observations_in_leaf_node_),
              class_count_(ctx.class_count_),
              impurity_threshold_(ctx.impurity_threshold_) {}

    Float float_min_;
    Index index_max_;
    Index min_observations_in_leaf_node_;
    Index class_count_;
    Float impurity_threshold_;
};

template <typename T, typename Index = std::size_t>
inline T* fill_zero(T* dst, Index elem_count) {
    for (Index i = 0; i < elem_count; ++i) {
        dst[i] = T(0);
    }
    return dst;
}

template <typename Float, typename Index>
inline void add_val_to_hist(
    typename task_types<Float, Index, task::classification>::hist_type_t* hist_ptr,
    Float val) {
    Index class_id = static_cast<Index>(val);
    hist_ptr[class_id] += 1;
}

template <typename Float, typename Index>
inline void add_val_to_hist(
    typename task_types<Float, Index, task::regression>::hist_type_t* hist_ptr,
    Float val) {
    hist_ptr[0] += Float(1);
    Float inv_n = Float(1) / hist_ptr[0];
    Float delta = val - hist_ptr[1]; // y[i] - mean
    hist_ptr[1] += delta * inv_n; // updated mean
    hist_ptr[2] += delta * (val - hist_ptr[1]); // updated sum2Cent
}

// merge src and dst stat
template <typename Float, typename Index, typename T = enable_if_float_t<Float>>
inline void merge_stat(Float* dst, const Float* src, Index) {
    if (Float(0) == src[0])
        return;

    Float sum_n1n2 = dst[0] + src[0];
    Float mul_n1n2 = dst[0] * src[0];
    Float delta_scl = mul_n1n2 / sum_n1n2;
    Float mean_scl = Float(1) / sum_n1n2;
    Float delta = src[1] - dst[1];

    dst[2] = dst[2] + src[2] + delta * delta * delta_scl;
    dst[1] = (dst[1] * dst[0] + src[1] * src[0]) * mean_scl;
    dst[0] = sum_n1n2;
}

template <typename Index>
inline void merge_stat(Index* dst, const Index* src, Index count) {
    for (Index i = 0; i < count; ++i) {
        dst[i] += src[i];
    }
}

template <typename Index>
inline void merge_stat(Index* dst, Index* accum, const Index* src, Index count) {
    for (Index i = 0; i < count; ++i) {
        *accum += src[i];
        dst[i] += src[i];
    }
}

template <typename Float, typename Index, typename T = enable_if_float_t<Float>>
inline void sub_stat(Float* dst, const Float* src, const Float* mrg, Index elem_count) {
    dst[0] = mrg[0] - src[0];

    dst[1] = dst[0] >= Float(1) ? mrg[1] + (src[0] * (mrg[1] - src[1])) / dst[0] : Float(0);

    Float sum_n1n2 = mrg[0];
    Float mul_n1n2 = src[0] * dst[0];
    Float delta_scl = mul_n1n2 / sum_n1n2;
    Float delta = src[1] - dst[1];

    dst[2] = dst[0] >= Float(1) ? (mrg[2] - src[2] - delta * delta * delta_scl) : Float(0);
}

// classification compute_hist_for_node
template <typename Float, typename Index, bool use_private_mem>
inline void compute_hist_for_node(
    sycl::nd_item<2> item,
    Index ind_start,
    Index ind_end,
    const Float* response_ptr,
    Index* node_ptr,
    const Index* node_tree_order_ptr,
    typename task_types<Float, Index, task::classification>::hist_type_t* local_buf_ptr,
    const imp_data_list_ptr_mutable<Float, Index, task::classification>& imp_list_ptr,
    const kernel_context<Float, Index, task::classification>& ctx,
    Index node_id) {
    using task_t = task::classification;
    using impl_const_t = impl_const<Index, task_t>;
    using hist_type_t = typename task_types<Float, Index, task_t>::hist_type_t;

    const Index local_id = item.get_local_id()[0];
    constexpr Index buff_size = impl_const_t::max_private_class_hist_buff_size;

    hist_type_t* prv_hist_ptr = nullptr;
    hist_type_t prv_hist_buf[buff_size] = { hist_type_t(0) };
    if constexpr (use_private_mem) {
        prv_hist_ptr = &prv_hist_buf[0];
    }
    else {
        prv_hist_ptr = fill_zero(local_buf_ptr + local_id * ctx.class_count_, ctx.class_count_);
    }

    Index* node_histogram_ptr = imp_list_ptr.class_hist_list_ptr_ + node_id * ctx.class_count_;
    Float* node_imp_ptr = imp_list_ptr.imp_list_ptr_ + node_id * impl_const_t::node_imp_prop_count_;
    const Index row_count = node_ptr[impl_const_t::ind_lrc];

    for (Index i = ind_start; i < ind_end; ++i) {
        Index id = node_tree_order_ptr[i];
        add_val_to_hist<Float, Index>(prv_hist_ptr, response_ptr[id]);
    }

    for (Index cls_idx = 0; cls_idx < ctx.class_count_; ++cls_idx) {
        bk::atomic_global_add(node_histogram_ptr + cls_idx, prv_hist_ptr[cls_idx]);
    }

    item.barrier(sycl::access::fence_space::local_space);

    Float imp = Float(1);
    Float div = Float(1) / (Float(row_count) * row_count);
    Index max_cls_count = 0;
    Index win_cls = 0;
    Index cls_count = 0;

    for (Index cls_idx = 0; cls_idx < ctx.class_count_; ++cls_idx) {
        cls_count = node_histogram_ptr[cls_idx];
        imp -= Float(cls_count) * (cls_count)*div;

        if (cls_count > max_cls_count) {
            max_cls_count = cls_count;
            win_cls = cls_idx;
        }
    }

    node_ptr[5] = win_cls;
    node_imp_ptr[0] = sycl::max(imp, Float(0));
}

// regression compute_hist_for_node
template <typename Float, typename Index, bool use_private_mem>
inline void compute_hist_for_node(
    sycl::nd_item<2> item,
    Index ind_start,
    Index ind_end,
    const Float* response_ptr,
    Index* node_ptr,
    const Index* node_tree_order_ptr,
    typename task_types<Float, Index, task::regression>::hist_type_t* local_buf_ptr,
    const imp_data_list_ptr_mutable<Float, Index, task::regression>& imp_list_ptr,
    const kernel_context<Float, Index, task::regression>& ctx,
    Index node_id) {
    using task_t = task::regression;
    using impl_const_t = impl_const<Index, task_t>;
    using hist_type_t = typename task_types<Float, Index, task_t>::hist_type_t;

    const Index local_id = item.get_local_id()[0];
    const Index local_size = item.get_local_range()[0];

    Float* node_imp_ptr = imp_list_ptr.imp_list_ptr_ + node_id * impl_const_t::node_imp_prop_count_;
    constexpr Index hist_prop_count = impl_const_t::hist_prop_count_;
    hist_type_t prv_hist[hist_prop_count] = { 0 };

    for (Index i = ind_start; i < ind_end; ++i) {
        Index id = node_tree_order_ptr[i];
        add_val_to_hist<Float, Index>(prv_hist, response_ptr[id]);
    }

    hist_type_t* local_h_ptr = local_buf_ptr + local_id * hist_prop_count;

    local_h_ptr[0] = prv_hist[0];
    local_h_ptr[1] = prv_hist[1];
    local_h_ptr[2] = prv_hist[2];

    for (Index offset = local_size / 2; offset > 0; offset >>= 1) {
        item.barrier(sycl::access::fence_space::local_space);
        if (local_id < offset) {
            hist_type_t* h_ptr = local_buf_ptr + (local_id + offset) * hist_prop_count;
            merge_stat(local_h_ptr, h_ptr, hist_prop_count);
        }
    }

    if (local_id == 0) {
        node_imp_ptr[0] = local_h_ptr[1]; // store mean
        node_imp_ptr[1] = local_h_ptr[2]; // store sum2cent
    }
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::compute_initial_imp_for_node_list(
    const train_context_t& ctx,
    imp_data_t& imp_data_list,
    pr::ndarray<Index, 1>& node_list,
    Index node_count,
    const bk::event_vector& deps) {
    ONEDAL_ASSERT(imp_data_list.imp_list_.get_count() ==
                  node_count * impl_const_t::node_imp_prop_count_);
    if constexpr (std::is_same_v<task::classification, Task>) {
        ONEDAL_ASSERT(imp_data_list.class_hist_list_.get_count() == node_count * ctx.class_count_);
    }
    ONEDAL_ASSERT(node_list.get_count() == node_count * impl_const_t::node_prop_count_);

    if constexpr (std::is_same_v<task::classification, Task>) {
        auto class_hist_list_host = imp_data_list.class_hist_list_.to_host(queue_, deps);
        auto imp_list_host = imp_data_list.imp_list_.to_host(queue_);
        auto node_list_host = node_list.to_host(queue_);

        const Index* class_hist_list_host_ptr = class_hist_list_host.get_data();
        Float* imp_list_host_ptr = imp_list_host.get_mutable_data();
        Index* node_list_host_ptr = node_list_host.get_mutable_data();

        for (Index node_idx = 0; node_idx < node_count; ++node_idx) {
            const Index* node_histogram_ptr =
                class_hist_list_host_ptr + node_idx * ctx.class_count_;
            Float* node_imp_ptr = imp_list_host_ptr + node_idx * impl_const_t::node_imp_prop_count_;
            Index* node_ptr = node_list_host_ptr + node_idx * impl_const_t::node_prop_count_;

            Index row_count = node_ptr[impl_const_t::ind_grc];

            Float imp = Float(1);
            Float div = Float(1) / (Float(row_count) * row_count);
            Index max_cls_count = 0;
            Index win_cls = 0;
            Index cls_count = 0;

            for (Index cls_idx = 0; cls_idx < ctx.class_count_; ++cls_idx) {
                cls_count = node_histogram_ptr[cls_idx];
                imp -= Float(cls_count) * (cls_count)*div;

                if (cls_count > max_cls_count) {
                    max_cls_count = cls_count;
                    win_cls = cls_idx;
                }
            }

            node_ptr[impl_const_t::ind_win] = win_cls;
            node_imp_ptr[0] = sycl::max(imp, Float(0));
        }
        imp_data_list.imp_list_.assign_from_host(queue_, imp_list_host).wait_and_throw();
        node_list.assign_from_host(queue_, node_list_host).wait_and_throw();
    }

    return sycl::event{};
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::compute_initial_histogram_local(
    const train_context_t& ctx,
    const pr::ndarray<Float, 1>& response,
    const pr::ndarray<Index, 1>& tree_order,
    pr::ndarray<Index, 1>& node_list,
    imp_data_t& imp_data_list,
    Index node_count,
    const bk::event_vector& deps) {
    ONEDAL_ASSERT(response.get_count() == ctx.row_count_);
    ONEDAL_ASSERT(tree_order.get_count() == ctx.tree_in_block_ * ctx.selected_row_total_count_);
    ONEDAL_ASSERT(node_list.get_count() == node_count * impl_const_t::node_prop_count_);
    ONEDAL_ASSERT(imp_data_list.imp_list_.get_count() ==
                  node_count * impl_const_t::node_imp_prop_count_);
    if constexpr (std::is_same_v<Task, task::classification>) {
        ONEDAL_ASSERT(imp_data_list.class_hist_list_.get_count() == node_count * ctx.class_count_);
    }

    const Float* response_ptr = response.get_data();
    const Index* tree_order_ptr = tree_order.get_data();
    Index* node_list_ptr = node_list.get_mutable_data();

    // num of split attributes for node
    const Index node_prop_count = impl_const_t::node_prop_count_;
    // num of impurity attributes for node

    sycl::event fill_event = {};
    if constexpr (std::is_same_v<Task, task::classification>) {
        fill_event = imp_data_list.class_hist_list_.fill(queue_, 0, deps);
    }

    imp_data_list_ptr_mutable<Float, Index, Task> imp_list_ptr(imp_data_list);
    const kernel_context<Float, Index, Task> krn_ctx(ctx);

    auto local_size = ctx.preferable_group_size_;
    const sycl::nd_range<2> nd_range =
        bk::make_multiple_nd_range_2d({ local_size, node_count }, { local_size, 1 });

    std::size_t local_buf_size = 0;
    bool use_private_mem_buf = ctx.use_private_mem_buf_;
    if constexpr (std::is_same_v<task::classification, Task>) {
        if (use_private_mem_buf) {
            local_buf_size = 1; // just some non zero value
        }
        else {
            local_buf_size = local_size * ctx.class_count_;
        }
    }
    else { //regression
        local_buf_size = local_size * impl_const_t::hist_prop_count_;
    }

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.depends_on(fill_event);
        // local_buf is used for regression only, but need to be present for classification also
        local_accessor_rw_t<hist_type_t> local_buf(local_buf_size, cgh);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<2> item) {
            const Index node_id = item.get_global_id()[1];
            const Index local_id = item.get_local_id()[0];
            const Index local_size = item.get_local_range()[0];

            Index* node_ptr = node_list_ptr + node_id * node_prop_count;

            const Index row_offset = node_ptr[impl_const_t::ind_ofs];
            const Index row_count = node_ptr[impl_const_t::ind_lrc];

            const Index elem_count = row_count / local_size + bool(row_count % local_size);

            const Index ind_start = local_id * elem_count;
            const Index ind_end =
                sycl::min(static_cast<Index>((local_id + 1) * elem_count), row_count);

            const Index* node_tree_order_ptr = &tree_order_ptr[row_offset];

            hist_type_t* local_buf_ptr = nullptr;
#if __SYCL_COMPILER_VERSION >= 20230828
            local_buf_ptr =
                local_buf.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
#else
            local_buf_ptr = local_buf.get_pointer().get();
#endif
            if (use_private_mem_buf) {
                compute_hist_for_node<Float, Index, true>(item,
                                                          ind_start,
                                                          ind_end,
                                                          response_ptr,
                                                          node_ptr,
                                                          node_tree_order_ptr,
                                                          local_buf_ptr,
                                                          imp_list_ptr,
                                                          krn_ctx,
                                                          node_id);
            }
            else {
                compute_hist_for_node<Float, Index, false>(item,
                                                           ind_start,
                                                           ind_end,
                                                           response_ptr,
                                                           node_ptr,
                                                           node_tree_order_ptr,
                                                           local_buf_ptr,
                                                           imp_list_ptr,
                                                           krn_ctx,
                                                           node_id);
            }
        });
    });

    event.wait_and_throw();
    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::compute_initial_sum_local(
    const train_context_t& ctx,
    const pr::ndarray<Float, 1>& response,
    const pr::ndarray<Index, 1>& tree_order,
    const pr::ndarray<Index, 1>& node_list,
    pr::ndarray<Float, 1>& sum_list,
    Index node_count,
    const bk::event_vector& deps) {
    ONEDAL_ASSERT(response.get_count() == ctx.row_count_);
    ONEDAL_ASSERT(tree_order.get_count() == ctx.tree_in_block_ * ctx.selected_row_total_count_);
    ONEDAL_ASSERT(node_list.get_count() == node_count * impl_const_t::node_prop_count_);
    ONEDAL_ASSERT(sum_list.get_count() == node_count);

    auto fill_event = sum_list.fill(queue_, 0, deps);
    fill_event.wait_and_throw();

    const Float* response_ptr = response.get_data();
    const Index* tree_order_ptr = tree_order.get_data();
    const Index* node_list_ptr = node_list.get_data();
    Float* sum_list_ptr = sum_list.get_mutable_data();

    // num of attributes for node
    const Index node_prop_count = impl_const_t::node_prop_count_;

    auto local_size = ctx.preferable_group_size_;
    const sycl::nd_range<2> nd_range =
        bk::make_multiple_nd_range_2d({ local_size, node_count }, { local_size, 1 });

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        local_accessor_rw_t<Float> local_buf(local_size, cgh);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<2> item) {
            const Index node_id = item.get_global_id()[1];
            const Index local_id = item.get_local_id()[0];
            const Index local_size = item.get_local_range()[0];

            const Index* node_ptr = node_list_ptr + node_id * node_prop_count;

            const Index row_offset = node_ptr[impl_const_t::ind_ofs];
            const Index row_count = node_ptr[impl_const_t::ind_lrc];

            const Index* node_tree_order_ptr = &tree_order_ptr[row_offset];
#if __SYCL_COMPILER_VERSION >= 20230828
            Float* local_buf_ptr =
                local_buf.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
#else
Float* local_buf_ptr = local_buf.get_pointer().get();
#endif
            Float sum = Float(0);
            for (Index i = local_id; i < row_count; i += local_size) {
                sum += response_ptr[node_tree_order_ptr[i]];
            }

            local_buf_ptr[local_id] = sum;

            for (Index offset = local_size / 2; offset > 0; offset >>= 1) {
                item.barrier(sycl::access::fence_space::local_space);
                if (local_id < offset) {
                    local_buf_ptr[local_id] += local_buf_ptr[local_id + offset];
                }
            }

            if (local_id == 0) {
                sum_list_ptr[node_id] = local_buf_ptr[local_id];
            }
        });
    });

    event.wait_and_throw();
    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::compute_initial_sum2cent_local(
    const train_context_t& ctx,
    const pr::ndarray<Float, 1>& response,
    const pr::ndarray<Index, 1>& tree_order,
    const pr::ndarray<Index, 1>& node_list,
    const pr::ndarray<Float, 1>& sum_list,
    pr::ndarray<Float, 1>& sum2cent_list,
    Index node_count,
    const bk::event_vector& deps) {
    ONEDAL_ASSERT(response.get_count() == ctx.row_count_);
    ONEDAL_ASSERT(tree_order.get_count() == ctx.tree_in_block_ * ctx.selected_row_total_count_);
    ONEDAL_ASSERT(node_list.get_count() == node_count * impl_const_t::node_prop_count_);
    ONEDAL_ASSERT(sum_list.get_count() == node_count);
    ONEDAL_ASSERT(sum2cent_list.get_count() == node_count);

    auto fill_event = sum2cent_list.fill(queue_, 0, deps);
    fill_event.wait_and_throw();

    const Float* response_ptr = response.get_data();
    const Index* tree_order_ptr = tree_order.get_data();
    const Index* node_list_ptr = node_list.get_data();
    const Float* sum_list_ptr = sum_list.get_data();
    Float* sum2cent_list_ptr = sum2cent_list.get_mutable_data();

    // num of attributes for node
    const Index node_prop_count = impl_const_t::node_prop_count_;

    auto local_size = ctx.preferable_group_size_;
    const sycl::nd_range<2> nd_range =
        bk::make_multiple_nd_range_2d({ local_size, node_count }, { local_size, 1 });

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        local_accessor_rw_t<Float> local_buf(local_size, cgh);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<2> item) {
            const Index node_id = item.get_global_id()[1];
            const Index local_id = item.get_local_id()[0];
            const Index local_size = item.get_local_range()[0];

            const Index* node_ptr = node_list_ptr + node_id * node_prop_count;

            const Index row_offset = node_ptr[impl_const_t::ind_ofs];
            const Index row_count = node_ptr[impl_const_t::ind_lrc];
            const Index global_row_count = node_ptr[impl_const_t::ind_grc];

            const Index* node_tree_order_ptr = &tree_order_ptr[row_offset];

            const Float mean = sum_list_ptr[node_id] / global_row_count;
#if __SYCL_COMPILER_VERSION >= 20230828
            Float* local_buf_ptr =
                local_buf.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
#else
Float* local_buf_ptr = local_buf.get_pointer().get();
#endif
            Float sum2cent = Float(0);
            for (Index i = local_id; i < row_count; i += local_size) {
                sum2cent += (response_ptr[node_tree_order_ptr[i]] - mean) *
                            (response_ptr[node_tree_order_ptr[i]] - mean);
            }

            local_buf_ptr[local_id] = sum2cent;

            for (Index offset = local_size / 2; offset > 0; offset >>= 1) {
                item.barrier(sycl::access::fence_space::local_space);
                if (local_id < offset) {
                    local_buf_ptr[local_id] += local_buf_ptr[local_id + offset];
                }
            }

            if (local_id == 0) {
                sum2cent_list_ptr[node_id] = local_buf_ptr[local_id];
            }
        });
    });

    event.wait_and_throw();
    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::fin_initial_imp(
    const train_context_t& ctx,
    const pr::ndarray<Index, 1>& node_list,
    const pr::ndarray<Float, 1>& sum_list,
    const pr::ndarray<Float, 1>& sum2cent_list,
    imp_data_t& imp_data_list,
    Index node_count,
    const bk::event_vector& deps) {
    ONEDAL_ASSERT(node_list.get_count() == node_count * impl_const_t::node_prop_count_);
    ONEDAL_ASSERT(sum_list.get_count() == node_count);
    ONEDAL_ASSERT(sum2cent_list.get_count() == node_count);
    ONEDAL_ASSERT(imp_data_list.imp_list_.get_count() ==
                  node_count * impl_const_t::node_imp_prop_count_);

    const Index* node_list_ptr = node_list.get_data();
    const Float* sum_list_ptr = sum_list.get_data();
    const Float* sum2cent_list_ptr = sum2cent_list.get_data();
    Float* imp_list_ptr = imp_data_list.imp_list_.get_mutable_data();

    const sycl::range<1> range{ de::integral_cast<std::size_t>(node_count) };

    auto last_event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::id<1> node_idx) {
            // set mean
            // node grc can't be 0 due to this is initial computation on whole ds block
            imp_list_ptr[node_idx * impl_const_t::node_imp_prop_count_ + 0] =
                sum_list_ptr[node_idx] /
                node_list_ptr[node_idx * impl_const_t::node_prop_count_ + impl_const_t::ind_grc];
            // set sum2cent
            imp_list_ptr[node_idx * impl_const_t::node_imp_prop_count_ + 1] =
                sum2cent_list_ptr[node_idx];
        });
    });

    return last_event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::compute_initial_histogram(
    const train_context_t& ctx,
    const pr::ndarray<Float, 1>& response,
    const pr::ndarray<Index, 1>& tree_order,
    pr::ndarray<Index, 1>& node_list,
    imp_data_t& imp_data_list,
    Index node_count,
    const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(compute_initial_histogram, queue_);

    ONEDAL_ASSERT(response.get_count() == ctx.row_count_);
    ONEDAL_ASSERT(tree_order.get_count() == ctx.tree_in_block_ * ctx.selected_row_total_count_);
    ONEDAL_ASSERT(node_list.get_count() == node_count * impl_const_t::node_prop_count_);
    ONEDAL_ASSERT(imp_data_list.imp_list_.get_count() ==
                  node_count * impl_const_t::node_imp_prop_count_);
    if constexpr (std::is_same_v<Task, task::classification>) {
        ONEDAL_ASSERT(imp_data_list.class_hist_list_.get_count() == node_count * ctx.class_count_);
    }

    sycl::event last_event;

    if (ctx.distr_mode_) {
        if constexpr (std::is_same_v<Task, task::classification>) {
            last_event = compute_initial_histogram_local(ctx,
                                                         response,
                                                         tree_order,
                                                         node_list,
                                                         imp_data_list,
                                                         node_count,
                                                         deps);
            {
                ONEDAL_PROFILER_TASK(allreduce_class_hist_list, queue_);
                comm_.allreduce(imp_data_list.class_hist_list_.flatten(queue_, { last_event }))
                    .wait();
            }
            last_event = compute_initial_imp_for_node_list(ctx,
                                                           imp_data_list,
                                                           node_list,
                                                           node_count,
                                                           { last_event });
        }
        else {
            auto sum_list = pr::ndarray<Float, 1>::empty(queue_, { node_count });
            auto sum2cent_list = pr::ndarray<Float, 1>::empty(queue_, { node_count });
            last_event = compute_initial_sum_local(ctx,
                                                   response,
                                                   tree_order,
                                                   node_list,
                                                   sum_list,
                                                   node_count,
                                                   deps);
            {
                ONEDAL_PROFILER_TASK(sum_list, queue_);
                comm_.allreduce(sum_list.flatten(queue_, { last_event })).wait();
            }
            last_event = compute_initial_sum2cent_local(ctx,
                                                        response,
                                                        tree_order,
                                                        node_list,
                                                        sum_list,
                                                        sum2cent_list,
                                                        node_count,
                                                        { last_event });
            {
                ONEDAL_PROFILER_TASK(allreduce_sum2cent_list, queue_);
                comm_.allreduce(sum2cent_list.flatten(queue_, { last_event })).wait();
            }
            last_event = fin_initial_imp(ctx,
                                         node_list,
                                         sum_list,
                                         sum2cent_list,
                                         imp_data_list,
                                         node_count,
                                         { last_event });
            last_event.wait_and_throw();
        }
    }
    else {
        last_event = compute_initial_histogram_local(ctx,
                                                     response,
                                                     tree_order,
                                                     node_list,
                                                     imp_data_list,
                                                     node_count,
                                                     deps);
        last_event.wait_and_throw();
    }

    return last_event;
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::compute_best_split(
    const train_context_t& ctx,
    const pr::ndarray<Bin, 2>& data,
    const pr::ndview<Float, 1>& response,
    const pr::ndview<Float, 1>& weights,
    const pr::ndarray<Index, 1>& tree_order,
    const pr::ndarray<Index, 1>& selected_ftr_list,
    const pr::ndarray<Float, 1>& random_bins_com,
    const pr::ndarray<Index, 1>& bin_offset_list,
    const imp_data_t& imp_data_list,
    pr::ndarray<Index, 1>& node_list,
    imp_data_t& left_child_imp_data_list,
    pr::ndarray<Float, 1>& node_imp_decrease_list,
    bool update_imp_dec_required,
    Index node_count,
    const bk::event_vector& deps) {
    using bs_kernels_opt_t = train_splitter_impl<Float, Bin, Index, Task>;

    sycl::event last_event;
    if (ctx.splitter_mode_value_ == splitter_mode::best) {
        last_event = bs_kernels_opt_t::best_split(queue_,
                                                  ctx,
                                                  data,
                                                  response,
                                                  weights,
                                                  tree_order,
                                                  selected_ftr_list,
                                                  bin_offset_list,
                                                  imp_data_list,
                                                  node_list,
                                                  left_child_imp_data_list,
                                                  node_imp_decrease_list,
                                                  update_imp_dec_required,
                                                  node_count,
                                                  deps);
    }
    else {
        last_event = bs_kernels_opt_t::random_split(queue_,
                                                    ctx,
                                                    data,
                                                    response,
                                                    tree_order,
                                                    selected_ftr_list,
                                                    random_bins_com,
                                                    bin_offset_list,
                                                    imp_data_list,
                                                    node_list,
                                                    left_child_imp_data_list,
                                                    node_imp_decrease_list,
                                                    update_imp_dec_required,
                                                    node_count,
                                                    deps);
    }
    return last_event;
}

template <typename Index>
inline void get_block_borders(Index total_elem_count,
                              Index block_count,
                              Index block_id,
                              Index& ind_start,
                              Index& ind_end) {
    const Index elem_count = total_elem_count / block_count + bool(total_elem_count % block_count);

    ind_start = block_id * elem_count;
    ind_end = sycl::min(static_cast<Index>(block_id + 1) * elem_count, total_elem_count);
}

template <typename Float, typename Index, typename Task>
static void do_node_imp_split(const imp_data_list_ptr<Float, Index, Task>& imp_list_ptr,
                              const imp_data_list_ptr<Float, Index, Task>& left_imp_list_ptr,
                              const imp_data_list_ptr_mutable<Float, Index, Task>& imp_list_ptr_new,
                              const Index* node_par,
                              Index* node_lch,
                              Index* node_rch,
                              const kernel_context<Float, Index, Task>& ctx,
                              Index node_id,
                              Index new_left_node_pos) {
    using impl_const_t = impl_const<Index, Task>;

    if constexpr (std::is_same_v<Task, task::classification>) {
        // assign class hist and compute winner for new nodes
        const Index* class_hist_p = imp_list_ptr.class_hist_list_ptr_ + node_id * ctx.class_count_;
        const Index* left_child_class_hist =
            left_imp_list_ptr.class_hist_list_ptr_ + node_id * ctx.class_count_;
        Index* class_hist_left =
            imp_list_ptr_new.class_hist_list_ptr_ + new_left_node_pos * ctx.class_count_;
        Index* class_hist_right =
            imp_list_ptr_new.class_hist_list_ptr_ + (new_left_node_pos + 1) * ctx.class_count_;

        Index max_cls_count_left = 0;
        Index max_cls_count_right = 0;
        Index win_cls_left = 0;
        Index win_cls_right = 0;

        Index rows_right = node_rch[impl_const_t::ind_grc];
        Float imp_right = Float(1);
        Float div_right = (0 < rows_right) ? Float(1) / (Float(rows_right) * rows_right) : Float(0);

        for (Index class_id = 0; class_id < ctx.class_count_; ++class_id) {
            class_hist_left[class_id] = left_child_class_hist[class_id];
            Index class_count_right = class_hist_p[class_id] - left_child_class_hist[class_id];
            class_hist_right[class_id] = class_count_right;

            imp_right -= Float(class_count_right) * (class_count_right)*div_right;

            if (class_hist_left[class_id] > max_cls_count_left) {
                max_cls_count_left = class_hist_left[class_id];
                win_cls_left = class_id;
            }

            if (class_hist_right[class_id] > max_cls_count_right) {
                max_cls_count_right = class_hist_right[class_id];
                win_cls_right = class_id;
            }
        }

        node_lch[impl_const_t::ind_win] = win_cls_left;
        node_rch[impl_const_t::ind_win] = win_cls_right;

        // assign impurity for new nodes
        const Float* left_child_imp =
            left_imp_list_ptr.imp_list_ptr_ + node_id * impl_const_t::node_imp_prop_count_;
        Float* node_lch_imp =
            imp_list_ptr_new.imp_list_ptr_ + new_left_node_pos * impl_const_t::node_imp_prop_count_;
        Float* node_rch_imp = imp_list_ptr_new.imp_list_ptr_ +
                              (new_left_node_pos + 1) * impl_const_t::node_imp_prop_count_;
        node_lch_imp[0] = left_child_imp[0];
        node_rch_imp[0] = sycl::max(imp_right, Float(0));
    }
    else {
        constexpr Index buff_size = impl_const_t::node_imp_prop_count_ + 1;
        const Float* left_child_imp =
            left_imp_list_ptr.imp_list_ptr_ + node_id * impl_const_t::node_imp_prop_count_;
        const Float* impP =
            imp_list_ptr.imp_list_ptr_ + node_id * impl_const_t::node_imp_prop_count_;
        Float* node_lch_imp =
            imp_list_ptr_new.imp_list_ptr_ + new_left_node_pos * impl_const_t::node_imp_prop_count_;
        Float* node_rch_imp = imp_list_ptr_new.imp_list_ptr_ +
                              (new_left_node_pos + 1) * impl_const_t::node_imp_prop_count_;

        Float node_hist[buff_size] = { static_cast<Float>(node_par[impl_const_t::ind_grc]),
                                       impP[0],
                                       impP[1] };
        Float left_hist[buff_size] = { static_cast<Float>(node_lch[impl_const_t::ind_grc]),
                                       left_child_imp[0],
                                       left_child_imp[1] };
        Float right_hist[buff_size] = { Float(0) };

        sub_stat(&right_hist[0], &left_hist[0], &node_hist[0], buff_size);

        node_lch_imp[0] = left_child_imp[0];
        node_lch_imp[1] = left_child_imp[1];

        node_rch_imp[0] = right_hist[1];
        node_rch_imp[1] = right_hist[2];
    }
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::do_node_split(
    const train_context_t& ctx,
    const pr::ndarray<Index, 1>& node_list,
    const pr::ndarray<Index, 1>& node_vs_tree_map_list,
    const imp_data_t& imp_data_list,
    const imp_data_t& left_child_imp_data_list,
    pr::ndarray<Index, 1>& node_list_new,
    pr::ndarray<Index, 1>& node_vs_tree_map_list_new,
    imp_data_t& imp_data_list_new,
    Index node_count,
    Index node_count_new,
    const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(do_node_split, queue_);

    ONEDAL_ASSERT(node_list.get_count() == node_count * impl_const_t::node_prop_count_);
    ONEDAL_ASSERT(node_vs_tree_map_list.get_count() == node_count);
    ONEDAL_ASSERT(imp_data_list.imp_list_.get_count() ==
                  node_count * impl_const_t::node_imp_prop_count_);
    if constexpr (std::is_same_v<Task, task::classification>) {
        ONEDAL_ASSERT(imp_data_list.class_hist_list_.get_count() == node_count * ctx.class_count_);
    }
    ONEDAL_ASSERT(left_child_imp_data_list.imp_list_.get_count() ==
                  node_count * impl_const_t::node_imp_prop_count_);
    if constexpr (std::is_same_v<Task, task::classification>) {
        ONEDAL_ASSERT(left_child_imp_data_list.class_hist_list_.get_count() ==
                      node_count * ctx.class_count_);
    }
    ONEDAL_ASSERT(node_list_new.get_count() == node_count_new * impl_const_t::node_prop_count_);
    ONEDAL_ASSERT(node_vs_tree_map_list_new.get_count() == node_count_new);
    ONEDAL_ASSERT(imp_data_list_new.imp_list_.get_count() ==
                  node_count_new * impl_const_t::node_imp_prop_count_);
    if constexpr (std::is_same_v<Task, task::classification>) {
        ONEDAL_ASSERT(imp_data_list_new.class_hist_list_.get_count() ==
                      node_count_new * ctx.class_count_);
    }

    const Index node_prop_count =
        impl_const_t::node_prop_count_; // num of split attributes for node
    const Index bad_val = impl_const_t::bad_val_;

    const Index* node_list_ptr = node_list.get_data();
    const Index* node_vs_tree_map_list_ptr = node_vs_tree_map_list.get_data();

    const bool distr_mode = ctx.distr_mode_;

    Index* node_list_new_ptr = node_list_new.get_mutable_data();
    Index* node_vs_tree_map_list_new_ptr = node_vs_tree_map_list_new.get_mutable_data();

    imp_data_list_ptr<Float, Index, Task> imp_list_ptr(imp_data_list);
    imp_data_list_ptr<Float, Index, Task> left_imp_list_ptr(left_child_imp_data_list);

    imp_data_list_ptr_mutable<Float, Index, Task> imp_list_ptr_new(imp_data_list_new);

    const kernel_context<Float, Index, Task> krn_ctx(ctx);

    auto local_size = bk::device_max_sg_size(queue_);
    const sycl::nd_range<1> nd_range = bk::make_multiple_nd_range_1d(local_size, local_size);

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();
            if (sbg.get_group_id() > 0) {
                return;
            }

            const Index local_id = sbg.get_local_id();
            const Index local_size = sbg.get_local_range()[0];

            Index created_node_count = 0;
            for (Index node_id = local_id; node_id < node_count; node_id += local_size) {
                Index split_node = Index(
                    node_list_ptr[node_id * node_prop_count + impl_const_t::ind_fid] != bad_val);
                Index new_left_node_pos =
                    created_node_count +
                    sycl::exclusive_scan_over_group(sbg, split_node, plus<Index>()) * 2;
                if (split_node) {
                    // split parent node on left and right nodes
                    const Index* node_prn = node_list_ptr + node_id * node_prop_count;
                    Index* node_lch = node_list_new_ptr + new_left_node_pos * node_prop_count;
                    Index* node_rch = node_list_new_ptr + (new_left_node_pos + 1) * node_prop_count;

                    node_lch[impl_const_t::ind_ofs] = node_prn[impl_const_t::ind_ofs];
                    node_lch[impl_const_t::ind_lrc] = distr_mode
                                                          ? node_prn[impl_const_t::ind_lch_lrc]
                                                          : node_prn[impl_const_t::ind_lch_grc];
                    node_lch[impl_const_t::ind_grc] = node_prn[impl_const_t::ind_lch_grc];
                    node_lch[impl_const_t::ind_fid] = impl_const_t::leaf_mark_;
                    node_lch[impl_const_t::ind_bin] = impl_const_t::leaf_mark_;
                    node_lch[impl_const_t::ind_lch_lrc] = 0;
                    node_lch[impl_const_t::ind_lch_grc] = 0;

                    node_rch[impl_const_t::ind_ofs] =
                        node_prn[impl_const_t::ind_ofs] + node_lch[impl_const_t::ind_lrc];
                    node_rch[impl_const_t::ind_lrc] =
                        node_prn[impl_const_t::ind_lrc] - node_lch[impl_const_t::ind_lrc];
                    node_rch[impl_const_t::ind_grc] =
                        node_prn[impl_const_t::ind_grc] - node_lch[impl_const_t::ind_grc];
                    node_rch[impl_const_t::ind_fid] = impl_const_t::leaf_mark_;
                    node_rch[impl_const_t::ind_bin] = impl_const_t::leaf_mark_;
                    node_rch[impl_const_t::ind_lch_lrc] = 0;
                    node_rch[impl_const_t::ind_lch_grc] = 0;

                    node_vs_tree_map_list_new_ptr[new_left_node_pos] =
                        node_vs_tree_map_list_ptr[node_id];
                    node_vs_tree_map_list_new_ptr[new_left_node_pos + 1] =
                        node_vs_tree_map_list_ptr[node_id];

                    do_node_imp_split<Float, Index, Task>(imp_list_ptr,
                                                          left_imp_list_ptr,
                                                          imp_list_ptr_new,
                                                          node_prn,
                                                          node_lch,
                                                          node_rch,
                                                          krn_ctx,
                                                          node_id,
                                                          new_left_node_pos);
                }
                created_node_count += sycl::reduce_over_group(sbg, split_node, plus<Index>()) * 2;
            }
        });
    });
    event.wait_and_throw();

    return event;
}

template <typename Float, typename Bin, typename Index, typename Task>
Float train_kernel_hist_impl<Float, Bin, Index, Task>::compute_oob_error(
    const train_context_t& ctx,
    const model_manager_t& model_manager,
    const pr::ndarray<Float, 1>& data_host,
    const pr::ndarray<Float, 1>& response_host,
    const pr::ndarray<Index, 1>& oob_row_list,
    pr::ndarray<hist_type_t, 1>& oob_per_obs_list,
    Index tree_idx,
    Index ind_ofs,
    Index n,
    const bk::event_vector& deps) {
    ONEDAL_ASSERT(data_host.get_count() == ctx.row_count_ * ctx.column_count_);
    ONEDAL_ASSERT(response_host.get_count() == ctx.row_count_);
    ONEDAL_ASSERT(oob_row_list.get_count() >= ind_ofs + n);
    ONEDAL_ASSERT(oob_per_obs_list.get_count() == ctx.row_count_ * ctx.oob_prop_count_);

    auto oob_row_list_host = oob_row_list.to_host(queue_, deps);
    auto oob_per_obs_list_host = oob_per_obs_list.to_host(queue_, deps);

    const Float* data_host_ptr = data_host.get_data();
    const Float* response_host_ptr = response_host.get_data();
    const Index* oob_row_list_host_ptr = oob_row_list_host.get_data();
    hist_type_t* oob_per_obs_list_host_ptr = oob_per_obs_list_host.get_mutable_data();

    Float mean = 0;
    for (Index i = 0; i < n; ++i) {
        Index row_ind = oob_row_list_host_ptr[ind_ofs + i];
        ONEDAL_ASSERT(row_ind < ctx.row_count_);

        auto prediction =
            model_manager.get_tree_response(tree_idx, &data_host_ptr[row_ind * ctx.column_count_]);

        if constexpr (std::is_same_v<Task, task::classification>) {
            ONEDAL_ASSERT(ctx.class_count_ == ctx.oob_prop_count_);
            Index class_idx = static_cast<Index>(prediction);
            oob_per_obs_list_host_ptr[row_ind * ctx.oob_prop_count_ + class_idx]++;
            mean += Float(class_idx != Index(response_host_ptr[row_ind]));
        }
        else {
            ONEDAL_ASSERT(2 == ctx.oob_prop_count_);
            oob_per_obs_list_host_ptr[row_ind * ctx.oob_prop_count_ + 0] += prediction;
            oob_per_obs_list_host_ptr[row_ind * ctx.oob_prop_count_ + 1] += Float(1);
            mean += (prediction - response_host_ptr[row_ind]) *
                    (prediction - response_host_ptr[row_ind]);
        }
    }

    oob_per_obs_list = oob_per_obs_list_host.to_device(queue_);

    return mean / Float(n);
}

template <typename Float, typename Bin, typename Index, typename Task>
Float train_kernel_hist_impl<Float, Bin, Index, Task>::compute_oob_error_perm(
    const train_context_t& ctx,
    const model_manager_t& model_manager,
    const pr::ndarray<Float, 1>& data_host,
    const pr::ndarray<Float, 1>& response_host,
    const pr::ndarray<Index, 1>& oob_row_list,
    const pr::ndarray<Index, 1>& permutation_host,
    Index tree_idx,
    Index ind_ofs,
    Index n,
    Index column_idx,
    const bk::event_vector& deps) {
    ONEDAL_ASSERT(data_host.get_count() == ctx.row_count_ * ctx.column_count_);
    ONEDAL_ASSERT(response_host.get_count() == ctx.column_count_);
    ONEDAL_ASSERT(oob_row_list.get_count() >= ind_ofs + n);
    ONEDAL_ASSERT(permutation_host.get_count() == n);
    ONEDAL_ASSERT(tree_idx < ctx.tree_count_);
    ONEDAL_ASSERT(column_idx < ctx.column_count_);

    auto oob_row_list_host = oob_row_list.to_host(queue_, deps);

    const Float* data_host_ptr = data_host.get_data();
    const Float* response_host_ptr = response_host.get_data();
    const Index* oob_row_list_host_ptr = oob_row_list_host.get_data();
    const Index* permutation_ptr = permutation_host.get_data();

    auto buf = pr::ndarray<Float, 1>::empty({ ctx.column_count_ });
    Float* buf_ptr = buf.get_mutable_data();

    Float mean = Float(0);
    for (Index i = 0; i < n; ++i) {
        int row_ind = oob_row_list_host_ptr[ind_ofs + i];
        int row_ind_perm = permutation_ptr[i];
        ONEDAL_ASSERT(row_ind < ctx.row_count_);
        ONEDAL_ASSERT(row_ind_perm < ctx.row_count_);

        memcpy(de::default_host_policy{},
               buf_ptr,
               &data_host_ptr[row_ind * ctx.column_count_],
               ctx.column_count_ * sizeof(Float));
        buf_ptr[column_idx] = data_host_ptr[row_ind_perm * ctx.column_count_ + column_idx];
        auto prediction = model_manager.get_tree_response(tree_idx, buf_ptr);
        if constexpr (std::is_same_v<Task, task::classification>) {
            Index class_idx = static_cast<Index>(prediction);
            mean += Float(class_idx != Index(response_host_ptr[row_ind]));
        }
        else {
            mean += (prediction - response_host_ptr[row_ind]) *
                    (prediction - response_host_ptr[row_ind]);
        }
    }

    return mean / Float(n);
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::compute_results(
    const train_context_t& ctx,
    const model_manager_t& model_manager,
    const pr::ndarray<Float, 1>& data_host,
    const pr::ndarray<Float, 1>& response_host,
    const pr::ndarray<Index, 1>& oob_row_list,
    const pr::ndarray<Index, 1>& oob_row_count_list,
    pr::ndarray<hist_type_t, 1>& oob_per_obs_list,
    pr::ndarray<Float, 1>& var_imp,
    pr::ndarray<Float, 1>& var_imp_variance,
    const rng_engine_list_t& engine_arr,
    Index tree_idx_in_block,
    Index tree_in_block_count,
    Index built_tree_count,
    const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(compute_results, queue_);

    ONEDAL_ASSERT(oob_row_count_list.get_count() == tree_in_block_count + 1);
    ONEDAL_ASSERT(
        (ctx.mdi_required_ || ctx.mda_required_) ? var_imp.get_count() == ctx.column_count_ : true);
    ONEDAL_ASSERT(ctx.mda_scaled_required_ ? var_imp_variance.get_count() == ctx.column_count_
                                           : true);

    auto oob_rows_count_list_host = oob_row_count_list.to_host(queue_, deps);
    const Index* oob_rows_count_list_host_ptr = oob_rows_count_list_host.get_data();
    Index oob_indices_offset = oob_rows_count_list_host_ptr[tree_idx_in_block];
    Index oob_row_count = oob_rows_count_list_host_ptr[tree_idx_in_block + 1] -
                          oob_rows_count_list_host_ptr[tree_idx_in_block];

    if ((ctx.oob_required_ || ctx.mda_required_) && oob_row_count) {
        const Float oob_err = compute_oob_error(ctx,
                                                model_manager,
                                                data_host,
                                                response_host,
                                                oob_row_list,
                                                oob_per_obs_list,
                                                tree_idx_in_block,
                                                oob_indices_offset,
                                                oob_row_count);

        if (ctx.mda_required_) {
            auto permutation = pr::ndarray<Index, 1>::empty({ oob_row_count });
            Index* permutation_ptr = permutation.get_mutable_data();
            for (Index i = 0; i < oob_row_count; ++i) {
                permutation_ptr[i] = i;
            }

            auto var_imp_host = var_imp.to_host(queue_);
            Float* var_imp_host_ptr = var_imp_host.get_mutable_data();
            Float* var_imp_var_host_ptr = nullptr;
            if (ctx.mda_scaled_required_) {
                auto var_imp_var_host = var_imp_variance.to_host(queue_);
                var_imp_var_host_ptr = var_imp_var_host.get_mutable_data();
            }

            const Float div1 = Float(1) / Float(built_tree_count + tree_idx_in_block + 1);

            for (Index column_idx = 0; column_idx < ctx.column_count_; ++column_idx) {
                pr::shuffle<Index>(oob_row_count,
                                   permutation_ptr,
                                   engine_arr[built_tree_count + tree_idx_in_block]);
                const Float oob_err_perm = compute_oob_error_perm(ctx,
                                                                  model_manager,
                                                                  data_host,
                                                                  response_host,
                                                                  oob_row_list,
                                                                  permutation,
                                                                  tree_idx_in_block,
                                                                  oob_indices_offset,
                                                                  oob_row_count,
                                                                  column_idx);

                const Float diff = (oob_err_perm - oob_err);
                const Float delta = diff - var_imp_host_ptr[column_idx];
                var_imp_host_ptr[column_idx] += div1 * delta;
                if (var_imp_var_host_ptr) {
                    var_imp_var_host_ptr[column_idx] +=
                        delta * (diff - var_imp_host_ptr[column_idx]);
                }
            }

            var_imp = var_imp_host.to_device(queue_);
            if (ctx.mda_scaled_required_) {
                var_imp_variance = var_imp_variance_host_.to_device(queue_);
            }
        }
    }

    return sycl::event{};
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::finalize_oob_error(
    const train_context_t& ctx,
    const pr::ndarray<Float, 1>& response_host,
    pr::ndarray<hist_type_t, 1>& oob_per_obs_list,
    pr::ndarray<Float, 1>& res_oob_err,
    pr::ndarray<Float, 1>& res_oob_err_obs,
    const bk::event_vector& deps) {
    ONEDAL_ASSERT(response_host.get_count() == ctx.row_count_);
    ONEDAL_ASSERT(oob_per_obs_list.get_count() == ctx.row_count_ * ctx.oob_prop_count_);
    // no need for assert for res_oob_err or res_oob_err_obs because they are created here

    auto oob_per_obs_list_host = oob_per_obs_list.to_host(queue_, deps);

    const Float* response_host_ptr = response_host.get_data();
    const hist_type_t* oob_per_obs_list_host_ptr = oob_per_obs_list_host.get_mutable_data();

    auto res_oob_err_host = pr::ndarray<Float, 1>::empty({ 1 });
    auto res_oob_err_obs_host = pr::ndarray<Float, 1>::empty({ ctx.row_count_ });
    Float* res_oob_err_host_ptr = res_oob_err_host.get_mutable_data();
    Float* res_oob_err_obs_host_ptr = res_oob_err_obs_host.get_mutable_data();

    Index predicted_count = 0;
    Float oob_err = 0;

    for (Index i = 0; i < ctx.row_count_; ++i) {
        bool predicted = false;
        hist_type_t prediction = hist_type_t(0);

        if constexpr (std::is_same_v<Task, task::classification>) {
            ONEDAL_ASSERT(ctx.class_count_ == ctx.oob_prop_count_);
            Index max_val = 0;
            for (Index class_idx = 0; class_idx < ctx.class_count_; ++class_idx) {
                Index val = oob_per_obs_list_host_ptr[i * ctx.oob_prop_count_ + class_idx];
                if (val > max_val) {
                    max_val = val;
                    prediction = class_idx;
                }
            }

            predicted = bool(0 < max_val);
        }
        else {
            ONEDAL_ASSERT(2 == ctx.oob_prop_count_);
            Float value = oob_per_obs_list_host_ptr[i * ctx.oob_prop_count_ + 0];
            Float count = oob_per_obs_list_host_ptr[i * ctx.oob_prop_count_ + 1];

            predicted = bool(count > Float(0));
            prediction = value / count;
        }

        if (predicted) {
            Float prediction_res = Float(0);
            if constexpr (std::is_same_v<Task, task::classification>) {
                prediction_res = Float(prediction != static_cast<Index>(response_host_ptr[i]));
            }
            else {
                prediction_res =
                    (prediction - response_host_ptr[i]) * (prediction - response_host_ptr[i]);
            }

            if (ctx.oob_err_obs_required_)
                res_oob_err_obs_host_ptr[i] = prediction_res;
            oob_err += prediction_res;
            ++predicted_count;
        }
        else if (ctx.oob_err_obs_required_) {
            //was not in OOB set of any tree and hence not predicted
            res_oob_err_obs_host_ptr[i] = Float(-1);
        }
    }

    if (ctx.oob_err_required_) {
        if (ctx.distr_mode_) {
            {
                ONEDAL_PROFILER_TASK(allreduce_predicted_count);
                comm_.allreduce(predicted_count).wait();
            }
            {
                ONEDAL_PROFILER_TASK(allreduce_oob_err);
                comm_.allreduce(oob_err).wait();
            }
        }

        *res_oob_err_host_ptr = (0 < predicted_count) ? oob_err / Float(predicted_count) : 0;
        res_oob_err = res_oob_err_host.to_device(queue_);
    }

    if (ctx.oob_err_obs_required_) {
        res_oob_err_obs = res_oob_err_obs_host.to_device(queue_);
    }

    return sycl::event{};
}

template <typename Float, typename Bin, typename Index, typename Task>
sycl::event train_kernel_hist_impl<Float, Bin, Index, Task>::finalize_var_imp(
    const train_context_t& ctx,
    pr::ndarray<Float, 1>& var_imp,
    pr::ndarray<Float, 1>& var_imp_variance,
    const bk::event_vector& deps) {
    ONEDAL_ASSERT(var_imp.get_count() == ctx.column_count_);

    auto var_imp_host = var_imp.to_host(queue_);
    Float* var_imp_host_ptr = var_imp_host.get_mutable_data();

    if (ctx.mda_scaled_required_) {
        if (ctx.tree_count_ > 1) {
            ONEDAL_ASSERT(var_imp_variance.get_count() == ctx.column_count_);
            auto var_imp_var_host = var_imp_variance.to_host(queue_);
            Float* var_imp_var_host_ptr = var_imp_var_host.get_mutable_data();

            const Float div = Float(1) / Float(ctx.tree_count_);
            for (Index i = 0; i < ctx.column_count_; ++i) {
                var_imp_var_host_ptr[i] *= div;
                if (var_imp_var_host_ptr[i] > Float(0)) {
                    var_imp_host_ptr[i] /= std::sqrt(var_imp_var_host_ptr[i] * div);
                }
            }
            var_imp = var_imp_host.to_device(queue_);
        }
        else {
            var_imp.fill(queue_, 0);
        }
    }
    else if (ctx.mdi_required_) {
        const Float div = Float(1) / ctx.tree_count_;
        for (Index i = 0; i < ctx.column_count_; ++i)
            var_imp_host_ptr[i] *= div;
        var_imp = var_imp_host.to_device(queue_);
    }

    return sycl::event{};
}

template <typename Float, typename Bin, typename Index, typename Task>
train_result<Task> train_kernel_hist_impl<Float, Bin, Index, Task>::operator()(
    const descriptor_t& desc,
    const table& data,
    const table& responses,
    const table& weights) {
    using imp_data_mng_t = impurity_data_manager<Float, Index, Task>;
    using tree_level_record_t = tree_level_record<Float, Index, Task>;

    validate_input(desc, data, responses);

    train_context_t ctx;
    init_params(ctx, desc, data, responses, weights);

    allocate_buffers(ctx);

    result_t res;

    model_manager_t model_manager(ctx, ctx.tree_count_, ctx.column_count_);

    /*init engines*/
    auto skip_num =
        de::check_mul_overflow<std::size_t>(ctx.row_total_count_, (ctx.selected_ftr_count_ + 1));
    skip_num = de::check_mul_overflow<std::size_t>(ctx.tree_count_, skip_num);

    de::check_mul_overflow<std::size_t>((ctx.tree_count_ - 1), skip_num);

    pr::engine_collection collection(ctx.tree_count_, desc.get_seed());
    rng_engine_list_t engine_arr = collection([&](std::size_t i, std::size_t& skip) {
        skip = i * skip_num;
    });

    pr::ndarray<Float, 1> node_imp_decrease_list;

    sycl::event last_event;

    for (Index iter = 0; iter < ctx.tree_count_; iter += ctx.tree_in_block_) {
        Index iter_tree_count = std::min(ctx.tree_count_ - iter, ctx.tree_in_block_);

        Index node_count = iter_tree_count; // num of potential nodes to split on current tree level
        auto oob_row_count_list =
            pr::ndarray<Index, 1>::empty(queue_, { iter_tree_count + 1 }, alloc::device);
        pr::ndarray<Index, 1> oob_rows_list;

        std::vector<tree_level_record_t> level_records;
        // lists of nodes int props(row_ofs, rows, ftrId, ftrVal ... )
        std::vector<pr::ndarray<Index, 1>> level_node_lists;

        imp_data_mng_t imp_data_holder(queue_, ctx);
        // initilizing imp_list and class_hist_list (for classification)
        imp_data_holder.init_new_level(node_count);

        de::check_mul_overflow(node_count, impl_const_t::node_prop_count_);
        de::check_mul_overflow(node_count, impl_const_t::node_imp_prop_count_);
        auto node_vs_tree_map_list_host = pr::ndarray<Index, 1>::empty({ node_count });
        auto level_node_list_init_host =
            pr::ndarray<Index, 1>::empty({ node_count * impl_const_t::node_prop_count_ });

        auto tree_map = node_vs_tree_map_list_host.get_mutable_data();
        auto node_list_ptr = level_node_list_init_host.get_mutable_data();

        for (Index node = 0; node < node_count; ++node) {
            Index* node_ptr = node_list_ptr + node * impl_const_t::node_prop_count_;
            tree_map[node] = iter + node;
            node_ptr[impl_const_t::ind_ofs] =
                ctx.selected_row_total_count_ * node; // local row offset
            node_ptr[impl_const_t::ind_lrc] =
                ctx.distr_mode_
                    ? 0
                    : ctx.selected_row_count_; // for distr_mode it will be updated during gen_initial_tree_order
            node_ptr[impl_const_t::ind_grc] =
                ctx.selected_row_total_count_; // global selected rows - it is already filtered for current block
            node_ptr[impl_const_t::ind_lch_lrc] =
                0; // for distr_mode it will be updated during tree_order_gen
            node_ptr[impl_const_t::ind_fid] = impl_const_t::bad_val_;
        }

        last_event = gen_initial_tree_order(ctx,
                                            engine_arr,
                                            level_node_list_init_host,
                                            tree_order_lev_,
                                            iter,
                                            node_count);

        auto node_vs_tree_map_list = node_vs_tree_map_list_host.to_device(queue_);
        level_node_lists.push_back(level_node_list_init_host.to_device(queue_));

        last_event = compute_initial_histogram(ctx,
                                               response_nd_,
                                               tree_order_lev_,
                                               level_node_lists[0],
                                               imp_data_holder.get_mutable_data(0),
                                               node_count,
                                               { last_event });
        last_event.wait_and_throw();

        if (ctx.oob_required_) {
            sycl::event event = train_service_kernels_.get_oob_row_list(
                tree_order_lev_,
                level_node_lists[0],
                oob_row_count_list,
                oob_rows_list,
                ctx.selected_row_total_count_,
                ctx.row_count_,
                node_count); // oob_row_count_list and oob_rows_list are the output
            event.wait_and_throw();
        }

        for (Index level = 0; node_count > 0; ++level) {
            auto node_list = level_node_lists[level];
            imp_data_t left_child_imp_data(queue_, ctx, node_count);

            auto [selected_features_com, event] =
                gen_feature_list(ctx, node_count, node_vs_tree_map_list, engine_arr);
            event.wait_and_throw();

            auto [random_bins_com, gen_bins_event] =
                gen_random_thresholds(ctx, node_count, node_vs_tree_map_list, engine_arr);
            gen_bins_event.wait_and_throw();

            if (ctx.mdi_required_) {
                node_imp_decrease_list =
                    pr::ndarray<Float, 1>::empty(queue_, { node_count }, alloc::device);
            }
            last_event = compute_best_split(ctx,
                                            full_data_nd_,
                                            response_nd_,
                                            weights_nd_,
                                            tree_order_lev_,
                                            selected_features_com,
                                            random_bins_com,
                                            ftr_bin_offsets_nd_,
                                            imp_data_holder.get_data(level),
                                            node_list,
                                            left_child_imp_data,
                                            node_imp_decrease_list,
                                            ctx.mdi_required_,
                                            node_count,
                                            { last_event });
            last_event.wait_and_throw();

            tree_level_record_t level_record(queue_,
                                             node_list,
                                             imp_data_holder.get_data(level),
                                             node_count,
                                             ctx);
            level_records.push_back(level_record);

            if (ctx.mdi_required_) {
                //mdi is calculated only on split nodes and is not calculated on last level
                last_event =
                    train_service_kernels_.update_mdi_var_importance(node_list,
                                                                     node_imp_decrease_list,
                                                                     res_var_imp_,
                                                                     ctx.column_count_,
                                                                     node_count,
                                                                     { last_event });
            }

            Index node_count_new;
            last_event = train_service_kernels_.get_split_node_count(node_list,
                                                                     node_count,
                                                                     node_count_new,
                                                                     { last_event });
            last_event.wait_and_throw();

            if (node_count_new) {
                //there are split nodes -> next level is required
                node_count_new *= 2;

                de::check_mul_overflow(node_count_new, impl_const_t::node_prop_count_);
                auto node_list_new = pr::ndarray<Index, 1>::empty(
                    queue_,
                    { node_count_new * impl_const_t::node_prop_count_ },
                    alloc::device);

                imp_data_holder.init_new_level(node_count_new);

                auto node_vs_tree_map_list_new =
                    pr::ndarray<Index, 1>::empty(queue_, { node_count_new }, alloc::device);

                if (ctx.distr_mode_) {
                    last_event =
                        train_service_kernels_.calculate_left_child_row_count_on_local_data(
                            ctx,
                            full_data_nd_,
                            node_list,
                            tree_order_lev_,
                            ctx.column_count_,
                            node_count,
                            { last_event });
                    last_event.wait_and_throw();
                }

                last_event = do_node_split(ctx,
                                           node_list,
                                           node_vs_tree_map_list,
                                           imp_data_holder.get_data(level),
                                           left_child_imp_data,
                                           node_list_new,
                                           node_vs_tree_map_list_new,
                                           imp_data_holder.get_mutable_data(level + 1),
                                           node_count,
                                           node_count_new,
                                           { last_event });
                last_event.wait_and_throw();
                if (ctx.max_tree_depth_ > 0 && ctx.max_tree_depth_ == level) {
                    tree_level_record_t level_record(queue_,
                                                     node_list_new,
                                                     imp_data_holder.get_data(level + 1),
                                                     node_count_new,
                                                     ctx,
                                                     { last_event });
                    level_records.push_back(level_record);
                    node_count_new = 0;
                }
                else {
                    level_node_lists.push_back(node_list_new);

                    node_vs_tree_map_list = node_vs_tree_map_list_new;

                    last_event = train_service_kernels_.do_level_partition_by_groups(
                        ctx,
                        full_data_nd_,
                        node_list,
                        tree_order_lev_,
                        tree_order_lev_buf_,
                        ctx.row_count_,
                        ctx.selected_row_total_count_,
                        ctx.column_count_,
                        node_count,
                        ctx.tree_in_block_,
                        { last_event });
                }
            }
            last_event.wait_and_throw();
            node_count = node_count_new;
        }

        last_event.wait_and_throw();

        model_manager.add_tree_block(level_records, bin_borders_host_, iter_tree_count);

        for (Index tree_idx = 0; tree_idx < iter_tree_count; ++tree_idx) {
            compute_results(ctx,
                            model_manager,
                            data_host_,
                            response_host_,
                            oob_rows_list,
                            oob_row_count_list,
                            oob_per_obs_list_,
                            res_var_imp_,
                            var_imp_variance_host_,
                            engine_arr,
                            tree_idx,
                            iter_tree_count,
                            iter,
                            { last_event })
                .wait_and_throw();
        }
    }

    // Finalize results
    if (ctx.oob_err_required_ || ctx.oob_err_obs_required_) {
        pr::ndarray<Float, 1> res_oob_err;
        pr::ndarray<Float, 1> res_oob_err_obs;

        finalize_oob_error(ctx, response_host_, oob_per_obs_list_, res_oob_err, res_oob_err_obs)
            .wait_and_throw();

        if (ctx.oob_err_required_) {
            auto res_oob_err_host = res_oob_err.to_host(queue_);
            res.set_oob_err(homogen_table::wrap(res_oob_err_host.flatten(), 1, 1));
        }

        if (ctx.oob_err_obs_required_) {
            auto res_oob_err_obs_host = res_oob_err_obs.to_host(queue_);
            res.set_oob_err_per_observation(
                homogen_table::wrap(res_oob_err_obs_host.flatten(), ctx.row_count_, 1));
        }
    }

    if (ctx.mdi_required_ || ctx.mda_required_) {
        finalize_var_imp(ctx, res_var_imp_, var_imp_variance_host_).wait_and_throw();
        auto res_var_imp_host = res_var_imp_.to_host(queue_);
        res.set_var_importance(
            homogen_table::wrap(res_var_imp_host.flatten(), 1, ctx.column_count_));
    }

    return res.set_model(model_manager.get_model());
}

#define INSTANTIATE(F, B, I, T) template class train_kernel_hist_impl<F, B, I, T>;

INSTANTIATE(float, std::uint32_t, std::int32_t, task::classification);
INSTANTIATE(float, std::uint32_t, std::int32_t, task::regression);

INSTANTIATE(double, std::uint32_t, std::int32_t, task::classification);
INSTANTIATE(double, std::uint32_t, std::int32_t, task::regression);

} // namespace oneapi::dal::decision_forest::backend

#endif
