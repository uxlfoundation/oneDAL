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

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/algo/decision_forest/train_types.hpp"

#include "oneapi/dal/backend/primitives/rng/rng_engine_collection.hpp"

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_misc_structs.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_impurity_data.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_service_kernels.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_feature_type.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_model_manager.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_splitter_impl.hpp"

namespace oneapi::dal::decision_forest::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

template <typename Float,
          typename Bin = std::uint32_t,
          typename Index = std::int32_t,
          typename Task = task::by_default>
class train_kernel_hist_impl {
    using result_t = train_result<Task>;
    using train_service_kernels_t = train_service_kernels<Float, Bin, Index, Task>;
    using impl_const_t = impl_const<Index, Task>;
    using descriptor_t = detail::descriptor_base<Task>;
    using model_manager_t = train_model_manager<Float, Index, Task>;
    using train_context_t = train_context<Float, Index, Task>;
    using imp_data_t = impurity_data<Float, Index, Task>;
    using rng_engine_t = pr::host_engine<pr::engine_method::mt2203>;
    using rng_engine_list_t = std::vector<rng_engine_t>;
    using msg = dal::detail::error_messages;
    using comm_t = bk::communicator<spmd::device_memory_access::usm>;
    using node_t = node<Index>;

public:
    using hist_type_t = typename task_types<Float, Index, Task>::hist_type_t;

    train_kernel_hist_impl(const bk::context_gpu& ctx)
            : queue_(ctx.get_queue()),
              comm_(ctx.get_communicator()),
              train_service_kernels_(queue_) {}
    ~train_kernel_hist_impl() = default;

    result_t operator()(const descriptor_t& desc,
                        const table& data,
                        const table& labels,
                        const table& weights);

private:
    std::int64_t get_part_hist_required_mem_size(Index selected_ftr_count,
                                                 Index max_bin_count_among_ftrs,
                                                 Index class_count) const;
    std::int64_t get_part_hist_elem_count(Index selected_ftr_count,
                                          Index max_bin_count_among_ftrs,
                                          Index class_count) const;

    sycl::event gen_initial_tree_order(train_context_t& ctx,
                                       rng_engine_list_t& rng_engine_list,
                                       pr::ndarray<Index, 1>& node_list,
                                       pr::ndarray<Index, 1>& tree_order_level,
                                       Index engine_offset,
                                       Index node_count);

    void validate_input(const descriptor_t& desc, const table& data, const table& labels) const;

    Index get_row_total_count(bool distr_mode, Index row_count);
    Index get_global_row_offset(bool distr_mode, Index row_count);

    /// Initializes `ctx` training context structure based on data and
    /// descriptor class. Filling and calculating all parameters in context,
    /// for example, tree count, required memory size, calculating indexed features, etc.
    ///
    /// @param[in] ctx          a training context structure for a GPU backend
    /// @param[in] desc         a structure containing training parameters
    /// @param[in] data         a table with training data
    /// @param[in] labels       a table with training labels
    void init_params(train_context_t& ctx,
                     const descriptor_t& desc,
                     const table& data,
                     const table& labels,
                     const table& weights);
    /// Allocates all buffers that are used for training.
    /// @param[in] ctx  a training context structure for a GPU backend
    void allocate_buffers(const train_context_t& ctx);

    /// Generates feature list for each node. If `boostrap=true`,
    /// selects a random subset of features for each node. Otherwise,
    /// assigns all features for each node. Returns the array of selected features
    /// on device and sycl::event.
    ///
    /// @param[in] ctx              a training context structure for a GPU backend
    /// @param[in] node_count       number of nodes on the current level
    /// @param[in] node_vs_tree_map an initial tree order
    /// @param[in] rng_engine_list  a list of random generator engines
    std::tuple<pr::ndarray<Index, 1>, sycl::event> gen_feature_list(
        const train_context_t& ctx,
        Index node_count,
        const pr::ndarray<Index, 1>& node_vs_tree_map,
        rng_engine_list_t& rng_engine_list);

    /// Generates random thresholds for each node and for each selected feature for node.
    /// Thresholds are used for a random splitter kernel to split each node.
    /// Returns an array of thresholds scaled in `[0,1]` on device and sycl::event.
    ///
    /// @param[in] ctx              a training context structure for a GPU backend
    /// @param[in] node_count       number of nodes on the current level
    /// @param[in] node_vs_tree_map an initial tree order
    /// @param[in] rng_engine_list  a list of random generator engines
    std::tuple<pr::ndarray<Float, 1>, sycl::event> gen_random_thresholds(
        const train_context_t& ctx,
        Index node_count,
        const pr::ndarray<Index, 1>& node_vs_tree_map,
        rng_engine_list_t& rng_engine_list);

    /// Computes initial impurity for each node.
    ///
    /// @param[in] ctx              a training context structure for a GPU backend
    /// @param[in] imp_data_list    an impurity data list
    /// @param[in] node_list        a node list containing splitting information
    /// @param[in] node_count       number of nodes on the current level
    /// @param[in] deps             a set of SYCL events this kernel depends on
    sycl::event compute_initial_imp_for_node_list(const train_context_t& ctx,
                                                  imp_data_t& imp_data_list,
                                                  pr::ndarray<Index, 1>& node_list,
                                                  Index node_count,
                                                  const bk::event_vector& deps = {});

    /// Computes initial histograms for each node to compute impurity.
    ///
    /// @param[in] ctx              a training context structure for a GPU backend
    /// @param[in] response         an array with data responses (labels)
    /// @param[in] tree_order       current tree order
    /// @param[in] node_list        a node list containing splitting information
    /// @param[in] imp_data_list    a list of nodes' impurity
    /// @param[in] node_count       number of nodes to process
    /// @param[in] deps             a set of SYCL events this kernel depends on
    sycl::event compute_initial_histogram_local(const train_context_t& ctx,
                                                const pr::ndarray<Float, 1>& response,
                                                const pr::ndarray<Index, 1>& tree_order,
                                                pr::ndarray<Index, 1>& node_list,
                                                imp_data_t& imp_data_list,
                                                Index node_count,
                                                const bk::event_vector& deps);

    /// Computes initial sum locally for each node. It is an internal kernel
    /// used for computing histograms.
    ///
    /// @param[in] ctx              a training context structure for a GPU backend
    /// @param[in] response         an array with data responses (labels)
    /// @param[in] tree_order       current tree order
    /// @param[in] node_list        a node list containing splitting information
    /// @param[in] sum_list         a list of nodes' sum
    /// @param[in] node_count       number of nodes to process
    /// @param[in] deps             a set of SYCL events this kernel depends on
    sycl::event compute_initial_sum_local(const train_context_t& ctx,
                                          const pr::ndarray<Float, 1>& response,
                                          const pr::ndarray<Index, 1>& tree_order,
                                          const pr::ndarray<Index, 1>& node_list,
                                          pr::ndarray<Float, 1>& sum_list,
                                          Index node_count,
                                          const bk::event_vector& deps);

    /// Computes initial distances to center. It is an internal kernel
    /// used for computing histograms.
    ///
    /// @param[in] ctx              a training context structure for a GPU backend
    /// @param[in] response         an array with data responses (labels)
    /// @param[in] tree_order       current tree order
    /// @param[in] node_list        a node list containing splitting information
    /// @param[in] sum_list         a list of nodes' sum
    /// @param[in] sum2cent_list    a list of nodes' distance to center
    /// @param[in] node_count       number of nodes to process
    /// @param[in] deps             a set of SYCL events this kernel depends on
    sycl::event compute_initial_sum2cent_local(const train_context_t& ctx,
                                               const pr::ndarray<Float, 1>& response,
                                               const pr::ndarray<Index, 1>& tree_order,
                                               const pr::ndarray<Index, 1>& node_list,
                                               const pr::ndarray<Float, 1>& sum_list,
                                               pr::ndarray<Float, 1>& sum2cent_list,
                                               Index node_count,
                                               const bk::event_vector& deps);

    /// Finalizes computation of initial impurity for each node.
    ///
    /// @param[in] ctx              a training context structure for a GPU backend
    /// @param[in] node_list        a node list containing splitting information
    /// @param[in] sum_list         a list of nodes' sum
    /// @param[in] sum2cent_list    a list of nodes' distance to center
    /// @param[in] imp_data_list    a list of nodes' impurity
    /// @param[in] node_count       number of nodes to process
    /// @param[in] deps             a set of SYCL events this kernel depends on
    sycl::event fin_initial_imp(const train_context_t& ctx,
                                const pr::ndarray<Index, 1>& node_list,
                                const pr::ndarray<Float, 1>& sum_list,
                                const pr::ndarray<Float, 1>& sum2cent_list,
                                imp_data_t& imp_data_list,
                                Index node_count,
                                const bk::event_vector& deps);

    /// Computes initial histogram. It is a high-level kernel,
    /// which uses auxiliary kernels defined above.
    ///
    /// @param[in] ctx              a training context structure for a GPU backend
    /// @param[in] response         an array with data responses (labels)
    /// @param[in] tree_order       current tree order
    /// @param[in] node_list        a node list containing splitting information
    /// @param[in] imp_data_list    a list of nodes' impurity
    /// @param[in] node_count       number of nodes to process
    /// @param[in] deps             a set of SYCL events this kernel depends on
    sycl::event compute_initial_histogram(const train_context_t& ctx,
                                          const pr::ndarray<Float, 1>& response,
                                          const pr::ndarray<Index, 1>& tree_order,
                                          pr::ndarray<Index, 1>& node_list,
                                          imp_data_t& imp_data_list,
                                          Index node_count,
                                          const bk::event_vector& deps);

    /// Computes node splitting based on split information in `node_list`.
    /// Adds split nodes to new lists.
    ///
    /// @param[in] ctx                          a training context structure for a GPU backend
    /// @param[in] node_list                    a node list containing splitting information
    /// @param[in] node_vs_tree_map             an initial tree order
    /// @param[in] imp_data_list                a list of nodes' impurity
    /// @param[in] left_child_imp_data_list     an array of left-child impurity values
    /// @param[in] node_list_new                a new node list containing splitting information
    /// @param[in] node_vs_tree_map_new         a new tree order
    /// @param[in] imp_data_list                a list of nodes' impurity
    /// @param[in] node_count                   number of nodes to process
    /// @param[in] node_count_new               new number of nodes
    /// @param[in] deps                         a set of SYCL events this kernel depends on
    sycl::event do_node_split(const train_context_t& ctx,
                              const pr::ndarray<Index, 1>& node_list,
                              const pr::ndarray<Index, 1>& node_vs_tree_map,
                              const imp_data_t& imp_data_list,
                              const imp_data_t& left_child_imp_data_list,
                              pr::ndarray<Index, 1>& node_list_new,
                              pr::ndarray<Index, 1>& node_vs_tree_map_new,
                              imp_data_t& imp_data_list_new,
                              Index node_count,
                              Index node_count_new,
                              const bk::event_vector& deps);

    /// Provides split computations for each node. This kernel filters nodes based on the
    /// theirs row count. For each row count group this kernel chooses appropriate splitting kernel.
    /// For example, for all nodes with row count less than
    /// node_t::get_elementary_node_max_row_count() best_split_single_pass_small invokes.
    ///
    /// @param[in] ctx                      a training context structure for a GPU backend
    /// @param[in] data                     a training data converted to bins
    /// @param[in] response                 an array with data responses (labels)
    /// @param[in] tree_order               current tree order
    /// @param[in] selected_ftr_list        an array of selected features for each node
    /// @param[in] random_bins_com          an array of random thresholds for each feature and for each node
    /// @param[in] bin_offset_list          an array of bin offsets
    /// @param[in] imp_data_list            a list of nodes' impurity
    /// @param[in] node_list                a node list containing splitting information
    /// @param[in] left_child_imp_data_list an array of left-child impurity values
    /// @param[in] node_imp_decrease_list   an array of a node decreases list
    /// @param[in] update_imp_dec_required  a boolean flag if required to update `node_imp_decrease_list`
    /// @param[in] node_count               number of nodes to process
    /// @param[in] deps                     a set of SYCL events this kernel depends on
    sycl::event compute_best_split(const train_context_t& ctx,
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
                                   const bk::event_vector& deps = {});

    /// Computes a histogram for each node at the current level. It can process
    /// histograms partially or a single run, depending on data size.
    ///
    /// @param[in] ctx                      a training context structure for a GPU backend
    /// @param[in] data                     a training data converted to bins
    /// @param[in] response                 an array with data responses (labels)
    /// @param[in] tree_order               current tree order
    /// @param[in] selected_ftr_list        an array of selected features for each node
    /// @param[in] bin_offset_list          an array of bin offsets
    /// @param[in] node_list                a node list containing splitting information
    /// @param[in] node_ind_list            a node indices list
    /// @param[in] node_ind_ofs             a node indices offset
    /// @param[in] npart_hist_list          number of partial histogram lists
    /// @param[in] node_count               number of nodes to process
    /// @param[in] deps                     a set of SYCL events this kernel depends on
    std::tuple<pr::ndarray<hist_type_t, 1>, sycl::event> compute_histogram(
        const train_context_t& ctx,
        const pr::ndarray<Bin, 2>& data,
        const pr::ndview<Float, 1>& response,
        const pr::ndarray<Index, 1>& tree_order,
        const pr::ndarray<Index, 1>& selected_ftr_list,
        const pr::ndarray<Index, 1>& bin_offset_list,
        const pr::ndarray<Index, 1>& node_list,
        const pr::ndarray<Index, 1>& node_ind_list,
        Index node_ind_ofs,
        Index npart_hist_list,
        Index node_count,
        const bk::event_vector& deps = {});

    /// Computes a histogram for each node at the current level in distributed manner, if
    /// platform/device supports it and the flag `ctx.distr` is true.
    ///
    /// @param[in] ctx                      a training context structure for a GPU backend
    /// @param[in] data                     a training data converted to bins
    /// @param[in] response                 an array with data responses (labels)
    /// @param[in] tree_order               current tree order
    /// @param[in] selected_ftr_list        an array of selected features for each node
    /// @param[in] bin_offset_list          an array of bin offsets
    /// @param[in] node_list                a node list containing splitting information
    /// @param[in] node_ind_list            a node indices list
    /// @param[in] node_ind_ofs             a node indices offset
    /// @param[in] npart_hist_list          number of partial histogram lists
    /// @param[in] node_count               number of nodes to process
    /// @param[in] deps                     a set of SYCL events this kernel depends on
    std::tuple<pr::ndarray<hist_type_t, 1>, sycl::event> compute_histogram_distr(
        const train_context_t& ctx,
        const pr::ndarray<Bin, 2>& data,
        const pr::ndview<Float, 1>& response,
        const pr::ndarray<Index, 1>& tree_order,
        const pr::ndarray<Index, 1>& selected_ftr_list,
        const pr::ndarray<Index, 1>& bin_offset_list,
        const pr::ndarray<Index, 1>& node_list,
        const pr::ndarray<Index, 1>& node_ind_list,
        Index node_ind_ofs,
        Index npart_hist_list,
        Index node_count,
        const bk::event_vector& deps = {});

    /// Computes partial histograms for each node. It is an internal kernel
    /// used in the `compute_histogram` kernel.
    ///
    /// @param[in] ctx                      a training context structure for a GPU backend
    /// @param[in] data                     a training data converted to bins
    /// @param[in] response                 an array with data responses (labels)
    /// @param[in] tree_order               current tree order
    /// @param[in] selected_ftr_list        an array of selected features for each node
    /// @param[in] bin_offset_list          an array of bin offsets
    /// @param[in] node_list                a node list containing splitting information
    /// @param[in] node_ind_list            a node indices list
    /// @param[in] node_ind_ofs             a node indices offset
    /// @param[in] part_hist_list           an array of partial histograms
    /// @param[in] part_hist_count          number of partial histograms
    /// @param[in] node_count               number of nodes to process
    /// @param[in] deps                     a set of SYCL events this kernel depends on
    sycl::event compute_partial_histograms(const train_context_t& ctx,
                                           const pr::ndarray<Bin, 2>& data,
                                           const pr::ndview<Float, 1>& response,
                                           const pr::ndarray<Index, 1>& tree_order,
                                           const pr::ndarray<Index, 1>& selected_ftr_list,
                                           const pr::ndarray<Index, 1>& bin_offset_list,
                                           const pr::ndarray<Index, 1>& node_list,
                                           const pr::ndarray<Index, 1>& node_ind_list,
                                           Index node_ind_ofs,
                                           pr::ndarray<hist_type_t, 1>& part_hist_list,
                                           Index part_hist_count,
                                           Index node_count,
                                           const bk::event_vector& deps = {});

    /// Reduces a partial histogram to one `hist_list`. It is an internal kernel
    /// used in the `compute_histogram` kernel.
    ///
    /// @param[in] ctx                      a training context structure for a GPU backend
    /// @param[in] part_hist_list           an array of partial histograms
    /// @param[in] hist_list                final histogram list
    /// @param[in] part_hist_count          number of partial histograms
    /// @param[in] node_count               number of nodes to process
    /// @param[in] deps                     a set of SYCL events this kernel depends on
    sycl::event reduce_partial_histograms(const train_context_t& ctx,
                                          const pr::ndarray<hist_type_t, 1>& part_hist_list,
                                          pr::ndarray<hist_type_t, 1>& hist_list,
                                          Index part_hist_count,
                                          Index node_count,
                                          const bk::event_vector& deps = {});

    /// Computes histogram statistics (count and sum) partially. It is an internal auxiliary kernel,
    /// which is used in the `compute_histogram` kernel.
    ///
    /// @param[in] ctx                      a training context structure for a GPU backend
    /// @param[in] data                     a training data converted to bins
    /// @param[in] response                 an array with data responses (labels)
    /// @param[in] tree_order               current tree order
    /// @param[in] selected_ftr_list        an array of selected features for each node
    /// @param[in] bin_offset_list          an array of bin offsets
    /// @param[in] node_list                a node list containing splitting information
    /// @param[in] node_ind_list            a node indices list
    /// @param[in] node_ind_ofs             a node indices offset
    /// @param[in] part_hist_list           an array of partial histograms
    /// @param[in] part_hist_count          number of partial histograms
    /// @param[in] node_count               number of nodes to process
    /// @param[in] deps                     a set of SYCL events this kernel depends on
    /// @param[in] task_val                 a task type: regression or classification
    sycl::event compute_partial_count_and_sum(const train_context_t& ctx,
                                              const pr::ndarray<Bin, 2>& data,
                                              const pr::ndview<Float, 1>& response,
                                              const pr::ndarray<Index, 1>& tree_order,
                                              const pr::ndarray<Index, 1>& selected_ftr_list,
                                              const pr::ndarray<Index, 1>& bin_offset_list,
                                              const pr::ndarray<Index, 1>& node_list,
                                              const pr::ndarray<Index, 1>& node_ind_list,
                                              Index node_ind_ofs,
                                              pr::ndarray<Float, 1>& part_hist_list,
                                              Index part_hist_count,
                                              Index node_count,
                                              const bk::event_vector& deps = {},
                                              const task::regression task_val = {});

    /// Computes distances to center partially. It is an internal auxiliary kernel
    /// used in the `compute_histogram` kernel.
    ///
    /// @param[in] ctx                      a training context structure for a GPU backend
    /// @param[in] data                     a training data converted to bins
    /// @param[in] response                 an array with data responses (labels)
    /// @param[in] sum_list                 an array of partial sums
    /// @param[in] tree_order               current tree order
    /// @param[in] selected_ftr_list        an array of selected features for each node
    /// @param[in] bin_offset_list          an array of bin offsets
    /// @param[in] node_list                a node list containing splitting information
    /// @param[in] node_ind_list            a node indices list
    /// @param[in] node_ind_ofs             a node indices offset
    /// @param[in] part_hist_list           an array of partial histograms
    /// @param[in] part_hist_count          number of partial histograms
    /// @param[in] node_count               number of nodes to process
    /// @param[in] deps                     a set of SYCL events this kernel depends on
    /// @param[in] task_val                 a task type: regression or classification
    sycl::event compute_partial_sum2cent(const train_context_t& ctx,
                                         const pr::ndarray<Bin, 2>& data,
                                         const pr::ndview<Float, 1>& response,
                                         const pr::ndview<Float, 1>& sum_list,
                                         const pr::ndarray<Index, 1>& tree_order,
                                         const pr::ndarray<Index, 1>& selected_ftr_list,
                                         const pr::ndarray<Index, 1>& bin_offset_list,
                                         const pr::ndarray<Index, 1>& node_list,
                                         const pr::ndarray<Index, 1>& node_ind_list,
                                         Index node_ind_ofs,
                                         pr::ndarray<Float, 1>& part_hist_list,
                                         Index part_hist_count,
                                         Index node_count,
                                         const bk::event_vector& deps = {},
                                         const task::regression task_val = {});

    /// Reduces partial sums into hist list. It is an internal auxiliary kernel
    /// used in the `compute_histogram` kernel.
    ///
    /// @param[in] ctx              a training context structure for a GPU backend
    /// @param[in] part_hist_list   an array of partial histograms
    /// @param[in] hist_list        final histogram list
    /// @param[in] part_hist_count  number of partial histograms
    /// @param[in] node_count       number of nodes to process
    /// @param[in] hist_prop_count  number of histogram properties
    /// @param[in] deps             a set of SYCL events this kernel depends on
    sycl::event sum_reduce_partial_histograms(const train_context_t& ctx,
                                              const pr::ndarray<Float, 1>& part_hist_list,
                                              pr::ndarray<Float, 1>& hist_list,
                                              Index part_hist_count,
                                              Index node_count,
                                              Index hist_prop_count,
                                              const bk::event_vector& deps = {});

    /// Finalizes distributed calculations for histogram using partially pre-computed statistics.
    ///
    /// @param[in] ctx              a training context structure for a GPU backend
    /// @param[in] sum_list         an array of partial sums
    /// @param[in] sum2cent_list    an array of partial distances to center
    /// @param[in] histogram_list   final histogram list
    /// @param[in] node_count       number of nodes to process
    /// @param[in] deps             a set of SYCL events this kernel depends on
    sycl::event fin_histogram_distr(const train_context_t& ctx,
                                    const pr::ndarray<Float, 1>& sum_list,
                                    const pr::ndarray<Float, 1>& sum2cent_list,
                                    pr::ndarray<Float, 1>& histogram_list,
                                    Index node_count,
                                    const bk::event_vector& deps = {});

    /// Computes Out-Of-Bag (OOB) error.
    ///
    /// @param[in] ctx              a training context structure for a GPU backend
    /// @param[in] model_manager    a model manager class
    /// @param[in] data_host        an array of training data on the host
    /// @param[in] response_host    an array of training labels on the host
    /// @param[in] oob_row_list     an array of OOB values
    /// @param[in] oob_per_obs_list an array of OOB values per observation
    /// @param[in] tree_idx         a tree index
    /// @param[in] ind_ofs          an offset of a tree
    /// @param[in] n                number of nodes
    /// @param[in] deps             a set of SYCL events this kernel depends on
    Float compute_oob_error(const train_context_t& ctx,
                            const model_manager_t& model_manager,
                            const pr::ndarray<Float, 1>& data_host,
                            const pr::ndarray<Float, 1>& response_host,
                            const pr::ndarray<Index, 1>& oob_row_list,
                            pr::ndarray<hist_type_t, 1>& oob_per_obs_list,
                            Index tree_idx,
                            Index ind_ofs,
                            Index n,
                            const bk::event_vector& deps = {});

    /// Computes Out-Of-Bag error with permutation.
    ///
    /// @param[in] ctx              a training context structure for a GPU backend
    /// @param[in] model_manager    a model manager class
    /// @param[in] data_host        an array of training data on the host
    /// @param[in] response_host    an array of training labels on the host
    /// @param[in] oob_row_list     an array of OOB values
    /// @param[in] permutation_host an array with indices' permutations
    /// @param[in] tree_idx         a tree index
    /// @param[in] ind_ofs          an offset of a tree
    /// @param[in] n                number of nodes
    /// @param[in] deps             a set of SYCL events this kernel depends on
    Float compute_oob_error_perm(const train_context_t& ctx,
                                 const model_manager_t& model_manager,
                                 const pr::ndarray<Float, 1>& data_host,
                                 const pr::ndarray<Float, 1>& response_host,
                                 const pr::ndarray<Index, 1>& oob_row_list,
                                 const pr::ndarray<Index, 1>& permutation_host,
                                 Index tree_idx,
                                 Index ind_ofs,
                                 Index n,
                                 Index column_idx,
                                 const bk::event_vector& deps = {});

    /// Computes results for Out-Of-Bag errors.
    ///
    /// @param[in] ctx                  a training context structure for a GPU backend
    /// @param[in] model_manager        a model manager class
    /// @param[in] data_host            an array of training data on the host
    /// @param[in] response_host        an array of training labels on the host
    /// @param[in] oob_row_list         an array of OOB values
    /// @param[in] oob_row_count_list   an array of OOB counts
    /// @param[in] oob_per_obs_list     an array of OOB values per observation
    /// @param[in] var_imp              variable importance values
    /// @param[in] var_imp_variance     variable importance variance values
    /// @param[in] rng_engine_arr       a list of random generator engines
    /// @param[in] tree_idx             a tree index
    /// @param[in] tree_in_block        number of trees in the computational block
    /// @param[in] built_tree_count     number of built trees
    /// @param[in] deps                 a set of SYCL events this kernel depends on
    sycl::event compute_results(const train_context_t& ctx,
                                const model_manager_t& model_manager,
                                const pr::ndarray<Float, 1>& data_host,
                                const pr::ndarray<Float, 1>& response_host,
                                const pr::ndarray<Index, 1>& oob_row_list,
                                const pr::ndarray<Index, 1>& oob_row_count_list,
                                pr::ndarray<hist_type_t, 1>& oob_per_obs_list,
                                pr::ndarray<Float, 1>& var_imp,
                                pr::ndarray<Float, 1>& var_imp_variance,
                                const rng_engine_list_t& rng_engine_arr,
                                Index tree_idx,
                                Index tree_in_block,
                                Index built_tree_count,
                                const bk::event_vector& deps = {});

    /// Finalizes Out-Of-Bug error computations.
    ///
    /// @param[in] ctx                  a training context structure for a GPU backend
    /// @param[in] response_host        an array of training labels on the host
    /// @param[in] oob_per_obs_list     an array of OOB values per observation
    /// @param[in] res_oob_err          final OOB error values
    /// @param[in] res_oob_err_obs      final OOB error values per observation
    /// @param[in] deps                 a set of SYCL events this kernel depends on
    sycl::event finalize_oob_error(const train_context_t& ctx,
                                   const pr::ndarray<Float, 1>& response_host,
                                   pr::ndarray<hist_type_t, 1>& oob_per_obs_list,
                                   pr::ndarray<Float, 1>& res_oob_err,
                                   pr::ndarray<Float, 1>& res_oob_err_obs,
                                   const bk::event_vector& deps = {});

    /// Finalizes variable importance computations.
    ///
    /// @param[in] ctx                  a training context structure for a GPU backend
    /// @param[in] var_imp              variable importance values
    /// @param[in] var_imp_variance     variable importance variance values
    /// @param[in] deps                 a set of SYCL events this kernel depends on
    sycl::event finalize_var_imp(const train_context_t& ctx,
                                 pr::ndarray<Float, 1>& var_imp,
                                 pr::ndarray<Float, 1>& var_imp_variance,
                                 const bk::event_vector& deps = {});

private:
    sycl::queue queue_;
    comm_t comm_;

    train_service_kernels_t train_service_kernels_;

    // algo buffers which are allocated one time at the beggining
    pr::ndarray<Bin, 2> full_data_nd_;
    pr::ndarray<Index, 1> ftr_bin_offsets_nd_;
    std::vector<pr::ndarray<Float, 1>> bin_borders_host_;
    pr::ndarray<Float, 1> response_nd_;
    pr::ndarray<Float, 1> weights_nd_;
    pr::ndarray<Float, 1> response_host_;
    pr::ndarray<Float, 1> data_host_;

    pr::ndarray<Index, 1> selected_row_host_;
    pr::ndarray<Index, 1> selected_row_global_host_;
    pr::ndarray<Index, 1> tree_order_lev_;
    pr::ndarray<Index, 1> tree_order_lev_buf_;

    pr::ndarray<Float, 1> node_imp_decr_list_;

    pr::ndarray<hist_type_t, 1> oob_per_obs_list_;
    pr::ndarray<Float, 1> var_imp_variance_host_;

    pr::ndarray<Float, 1> res_var_imp_;
};

#endif

} // namespace oneapi::dal::decision_forest::backend
