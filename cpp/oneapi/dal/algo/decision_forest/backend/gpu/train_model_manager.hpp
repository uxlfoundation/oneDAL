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
#include "oneapi/dal/algo/decision_forest/common.hpp"
#include "oneapi/dal/algo/decision_forest/backend/model_impl.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_misc_structs.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_impurity_data.hpp"

#include <daal/src/algorithms/dtrees/forest/classification/df_classification_model_impl.h>
#include <daal/src/algorithms/dtrees/forest/regression/df_regression_model_impl.h>
#include <daal/src/algorithms/dtrees/dtrees_predict_dense_default_impl.i>

namespace oneapi::dal::decision_forest::backend {

#ifdef ONEDAL_DATA_PARALLEL

template <typename Task>
struct daal_types_map;

template <>
struct daal_types_map<task::classification> {
    using daal_model_interop_t = backend::model_interop_cls;
    using daal_tree_impl_t = daal::algorithms::dtrees::internal::TreeImpClassification<>;
    using daal_model_impl_t =
        daal::algorithms::decision_forest::classification::internal::ModelImpl;
    using daal_model_ptr_t = daal::algorithms::decision_forest::classification::ModelPtr;
};

template <>
struct daal_types_map<task::regression> {
    using daal_model_interop_t = backend::model_interop_reg;
    using daal_tree_impl_t = daal::algorithms::dtrees::internal::TreeImpRegression<>;
    using daal_model_impl_t = daal::algorithms::decision_forest::regression::internal::ModelImpl;
    using daal_model_ptr_t = daal::algorithms::decision_forest::regression::ModelPtr;
};

template <typename Float, typename Index, typename Task = task::by_default>
class tree_level_record {
    using impl_const_t = impl_const<Index, Task>;
    using context_t = train_context<Float, Index, Task>;
    using imp_data_t = impurity_data<Float, Index, Task>;

public:
    tree_level_record(sycl::queue& queue,
                      dal::backend::primitives::ndarray<Index, 1> node_list,
                      const imp_data_t& imp_data_list,
                      Index node_count,
                      const context_t& ctx,
                      const dal::backend::event_vector& deps = {})
            : node_count_(node_count),
              ctx_(ctx) {
        ONEDAL_ASSERT(node_list.get_count() == node_count * impl_const_t::node_prop_count_);
        ONEDAL_ASSERT(imp_data_list.imp_list_.get_count() ==
                      node_count * impl_const_t::node_imp_prop_count_);
        if constexpr (std::is_same_v<task::classification, Task>) {
            ONEDAL_ASSERT(imp_data_list.class_hist_list_.get_count() ==
                          node_count * ctx.class_count_);
        }

        node_list_ = node_list.to_host(queue, deps);
        imp_data_list_ = imp_data_list.to_host(queue);
    }

    Index get_node_count() const {
        return node_count_;
    }

    Index get_row_count(Index node_idx) const {
        return node_list_
            .get_data()[node_idx * impl_const_t::node_prop_count_ + impl_const_t::ind_grc];
    }
    Index get_feature_id(Index node_idx) const {
        return node_list_
            .get_data()[node_idx * impl_const_t::node_prop_count_ + impl_const_t::ind_fid];
    }
    Index get_feature_bin(Index node_idx) const {
        return node_list_
            .get_data()[node_idx * impl_const_t::node_prop_count_ + impl_const_t::ind_bin];
    }

    auto get_response(Index node_idx) const {
        if constexpr (std::is_same_v<Task, task::classification>) {
            return node_list_
                .get_data()[node_idx * impl_const_t::node_prop_count_ + impl_const_t::ind_win];
        }
        else {
            return imp_data_list_.imp_list_
                .get_data()[node_idx * impl_const_t::node_imp_prop_count_ + impl_const_t::ind_rsp];
        }
    }

    Float get_impurity(Index node_idx) const {
        if constexpr (std::is_same_v<Task, task::classification>) {
            return imp_data_list_.imp_list_
                .get_data()[node_idx * impl_const_t::node_imp_prop_count_ + impl_const_t::ind_imp];
        }
        else {
            return imp_data_list_.imp_list_
                       .get_data()[node_idx * impl_const_t::node_imp_prop_count_ +
                                   impl_const_t::ind_imp] /
                   get_row_count(node_idx);
        }
    }

    template <typename T = Task, typename = decision_forest::detail::enable_if_classification_t<T>>
    const Index* get_class_hist(Index node_idx) const {
        return &(imp_data_list_.class_hist_list_.get_data()[node_idx * ctx_.class_count_]);
    }

    bool is_leaf(Index node_idx) const {
        return get_feature_id(node_idx) == impl_const_t::leaf_mark_;
    }
    bool has_unordered_feature(Index node_idx) const {
        return false; /* unordered features are not supported yet */
    }

private:
    dal::backend::primitives::ndarray<Index, 1> node_list_;
    imp_data_t imp_data_list_;

    Index node_count_;
    const context_t& ctx_;
};

template <typename Float, typename Index, typename Task = task::by_default>
class train_model_manager {
    using train_context_t = train_context<Float, Index, Task>;
    using model_t = model<Task>;
    using model_impl_t = detail::model_impl<Task>;
    using model_interop_impl_t =
        model_interop_impl<typename daal_types_map<Task>::daal_model_ptr_t>;
    using daal_model_impl_t = typename daal_types_map<Task>::daal_model_impl_t;
    using daal_model_interop_t = typename daal_types_map<Task>::daal_model_interop_t;
    using daal_model_ptr_t = typename daal_types_map<Task>::daal_model_ptr_t;
    using TreeType = typename daal_types_map<Task>::daal_tree_impl_t;
    using NodeType = typename TreeType::NodeType;

public:
    explicit train_model_manager(const train_context_t& ctx, Index tree_count, Index column_count)
            : allocator_(allocator_node_count_hint_),
              daal_model_ptr_(new daal_model_impl_t(column_count)),
              daal_model_interface_ptr_(daal_model_ptr_),
              tree_list_(tree_count),
              ctx_(ctx) {
        daal_model_ptr_->resize(tree_count);
        total_tree_count = tree_count;
    }

    ~train_model_manager() = default;

    void copy_trees(model_t model_) {
        auto daal_model =
            static_cast<const daal_model_interop_t*>(dal::detail::get_impl(model_).get_interop())
                ->get_model();

        auto daal_model_ptr_local = static_cast<const daal_model_impl_t*>(daal_model.get());

        if (!daal_model_ptr_local) {
            throw std::runtime_error("Failed to cast to daal_model_impl_t");
        }
        if constexpr (std::is_same_v<task::classification, Task>) {
            daal_model_ptr_->copyModelCls(*daal_model_ptr_local, last_tree_copy_, total_tree_count);
        }
        else {
            daal_model_ptr_->copyModelReg(*daal_model_ptr_local, last_tree_copy_, total_tree_count);
        }
        last_tree_copy_ += daal_model_ptr_local->getNumberOfTrees();
    }

    void add_tree_block(std::vector<tree_level_record<Float, Index, Task>>& tree_level_list,
                        std::vector<dal::backend::primitives::ndarray<Float, 1>>& bin_value_list,
                        Index tree_count) {
        typedef std::vector<typename NodeType::Base*> df_tree_node_list_t;
        typedef dal::detail::shared<df_tree_node_list_t> df_tree_node_list_ptr_t;

        df_tree_node_list_ptr_t df_tree_level_node_list_prev;
        bool unord_ftr_used = false;

        Index level = tree_level_list.size();
        ONEDAL_ASSERT(level);

        do {
            level--;
            tree_level_record<Float, Index, Task>& record = tree_level_list[level];
            df_tree_node_list_ptr_t df_tree_level_node_list(
                new df_tree_node_list_t(record.get_node_count()));

            Index split_count = 0;
            // split_count is used to calculate index of child nodes on next level
            for (Index node_idx = 0; node_idx < record.get_node_count(); node_idx++) {
                if (record.is_leaf(node_idx)) {
                    (*df_tree_level_node_list)[node_idx] = make_leaf(record, node_idx);
                }
                else {
                    ONEDAL_ASSERT(df_tree_level_node_list_prev.get());
                    (*df_tree_level_node_list)[node_idx] =
                        make_split(record,
                                   bin_value_list,
                                   node_idx,
                                   (*df_tree_level_node_list_prev)[split_count * 2],
                                   (*df_tree_level_node_list_prev)[split_count * 2 + 1]);
                    split_count++;
                }
            }

            df_tree_level_node_list_prev = df_tree_level_node_list;
        } while (level > 0);

        ONEDAL_ASSERT(static_cast<std::size_t>(last_tree_pos_ + tree_count) <= tree_list_.size());

        for (Index tree_idx = 0; tree_idx < tree_count; tree_idx++) {
            tree_list_[last_tree_pos_ + tree_idx].reset((*df_tree_level_node_list_prev)[tree_idx],
                                                        unord_ftr_used);
            Index class_count = 0;
            if constexpr (std::is_same_v<task::classification, Task>) {
                class_count = ctx_.class_count_;
            }

            daal_model_ptr_->add(tree_list_[last_tree_pos_ + tree_idx],
                                 class_count,
                                 last_tree_pos_ + tree_idx);
        }
        last_tree_pos_ += tree_count;
    }

    Float get_tree_response(Index tree_idx, const Float* x) const {
        ONEDAL_ASSERT(static_cast<std::size_t>(tree_idx) < tree_list_.size());
        const typename NodeType::Base* node_ptr =
            daal::algorithms::dtrees::prediction::internal::findNode<Float, TreeType, daal::sse2>(
                tree_list_[tree_idx],
                x);
        ONEDAL_ASSERT(node_ptr);
        if constexpr (std::is_same_v<task::classification, Task>) {
            return NodeType::castLeaf(node_ptr)->response.value;
        }
        else {
            return NodeType::castLeaf(node_ptr)->response;
        }
    }

    model_t get_model() {
        const auto model_impl =
            std::make_shared<model_impl_t>(new model_interop_impl_t{ daal_model_interface_ptr_ });
        model_impl->tree_count = daal_model_ptr_->getNumberOfTrees();

        if constexpr (std::is_same_v<task::classification, Task>) {
            model_impl->class_count = daal_model_ptr_->getNumberOfClasses();
        }

        return dal::detail::make_private<model_t>(model_impl);
    }

private:
    typename NodeType::Leaf* make_leaf(tree_level_record<Float, Index, Task>& record,
                                       Index node_idx) {
        ONEDAL_ASSERT(record.get_row_count(node_idx) > 0);

        typename NodeType::Leaf* node_ptr;
        if constexpr (std::is_same_v<task::classification, Task>) {
            node_ptr = allocator_.allocLeaf(ctx_.class_count_);
            node_ptr->response.value = record.get_response(node_idx);
        }
        else {
            node_ptr = allocator_.allocLeaf();
            node_ptr->response = record.get_response(node_idx);
        }

        node_ptr->count = record.get_row_count(node_idx);
        node_ptr->impurity = record.get_impurity(node_idx);

        if constexpr (std::is_same_v<task::classification, Task>) {
            const Index* hist_ptr = record.get_class_hist(node_idx);
            for (Index i = 0; i < ctx_.class_count_; i++) {
                node_ptr->hist[i] = static_cast<Float>(hist_ptr[i]);
            }
        }

        return node_ptr;
    }

    typename NodeType::Split* make_split(
        const tree_level_record<Float, Index, Task>& record, //const ???
        std::vector<dal::backend::primitives::ndarray<Float, 1>>& feature_value_arr,
        Index node_idx,
        typename NodeType::Base* left,
        typename NodeType::Base* right) {
        typename NodeType::Split* node_ptr = allocator_.allocSplit();
        node_ptr->set(record.get_feature_id(node_idx),
                      feature_value_arr[record.get_feature_id(node_idx)]
                          .get_data()[record.get_feature_bin(node_idx)],
                      record.has_unordered_feature(node_idx));
        node_ptr->kid[0] = left;
        node_ptr->kid[1] = right;
        node_ptr->count = record.get_row_count(node_idx);
        node_ptr->impurity = record.get_impurity(node_idx);

        return node_ptr;
    }

private:
    //number of nodes as a hint for allocator to grow by
    constexpr static Index allocator_node_count_hint_ = 512;
    typename TreeType::Allocator allocator_;
    daal_model_impl_t* daal_model_ptr_;
    daal_model_ptr_t daal_model_interface_ptr_;

    Index last_tree_pos_ = 0;
    Index last_tree_copy_ = 0;
    Index total_tree_count = 0;

    std::vector<TreeType> tree_list_;
    const train_context_t& ctx_;
};

#endif

} // namespace oneapi::dal::decision_forest::backend
