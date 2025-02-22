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

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_misc_structs.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_impurity_data.hpp"

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_node_helpers.hpp"

namespace oneapi::dal::decision_forest::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = bk::primitives;

template <typename Float,
          typename Bin = std::uint32_t,
          typename Index = std::int32_t,
          typename Task = task::by_default>
class train_splitter_impl {
    static_assert(std::is_signed_v<Index>);
    static_assert(std::is_integral_v<Index>);

    using result_t = train_result<Task>;
    using impl_const_t = impl_const<Index, Task>;
    using descriptor_t = detail::descriptor_base<Task>;
    using context_t = train_context<Float, Index, Task>;
    using imp_data_t = impurity_data<Float, Index, Task>;
    using msg = de::error_messages;
    using hist_type_t = typename task_types<Float, Index, Task>::hist_type_t;
    using node_t = node<Index>;

public:
    train_splitter_impl() = default;
    ~train_splitter_impl() = default;

    /// Chooses the best split for each feature from
    /// the `selected_ftr_list` in terms of impurity values.
    /// This kernel does not require pre-computed histograms because it computes
    /// histograms while processing nodes.
    ///
    /// @param[in] queue                            SYCL parallel queue
    /// @param[in] ctx                              a training context structure for the GPU backend
    /// @param[in] data                             an input data with the cast to Bin for each row_count * column_count
    /// @param[in] response                         an input array of training response values
    /// @param[in] weights                          an input array of weights for weighted training
    /// @param[in] tree_order                       column indices map for the corresponding tree
    /// @param[in] selected_ftr_list                a subset of feature indices selected for each node
    /// @param[in] bin_offset_list                  a bin offset list for each feature in dataset
    /// @param[in] imp_data_list                    a node impurity data array
    /// @param[in, out] node_list                   a node data structure containing split information
    /// @param[in, out] left_child_imp_data_list    an impurity array for left-child
    /// @param[in, out] node_imp_dec_list           an array for node impurity decrease data
    /// @param[in] update_imp_dec_required          boolean indicator to update impurity decrease structure
    /// @param[in] node_count                       number of the nodes to compute in the current step
    /// @param[in] deps                             a set of SYCL events this method depends on
    static sycl::event best_split(sycl::queue& queue,
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
                                  const bk::event_vector& deps = {});

    /// Choses a random split for each feature from
    /// the `selected_ftr_list` using `random_bins_com` values for splitting.
    /// And computes the best split among those random splits
    /// based on the impurity decrease calculated for each of those random splits.
    ///
    /// @param[in] queue                        sycl parallel queue
    /// @param[in] ctx                          a train context for GPU backend
    /// @param[in] data                         an input data with the cast to Bin for each row_count * column_count
    /// @param[in] response                     an input array of training response values
    /// @param[in] tree_order                   column indices map for the corresponding tree
    /// @param[in] selected_ftr_list            a subset of feature indices selected for each node
    /// @param[in] random_bins_com              random bin tresholds for each selected feature scaled at [0.0, 1.0] uniformly
    /// @param[in] bin_offset_list              a bin offset list for each feature in dataset
    /// @param[in] imp_data_list                a node impurity data array
    /// @param[in, out] node_list                    a node data structure containing split information
    /// @param[in, out] left_child_imp_data_list     an impurity array for left child
    /// @param[in, out] node_imp_dec_list            an array for node impurity decrease data
    /// @param[in] update_imp_dec_required      boolean indicator to update impurity decrease structure
    /// @param[in] node_count                   number of the nodes to compute in the current step
    /// @param[in] deps                         a set of SYCL events this method depends on
    static sycl::event random_split(sycl::queue& queue,
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
                                    const bk::event_vector& deps = {});
};

#endif

} // namespace oneapi::dal::decision_forest::backend
