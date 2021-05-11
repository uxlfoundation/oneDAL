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
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_cls_hist_aux_props.hpp"

#include <daal/src/algorithms/dtrees/forest/classification/df_classification_model_impl.h>
#include <daal/src/algorithms/dtrees/forest/regression/df_regression_model_impl.h>
#include <daal/src/algorithms/dtrees/dtrees_predict_dense_default_impl.i>

namespace oneapi::dal::decision_forest::backend {

#ifdef ONEDAL_DATA_PARALLEL

template <typename Task>
struct daal_types_map;

template <>
struct daal_types_map<task::classification> {
    using daal_tree_impl_t = daal::algorithms::dtrees::internal::TreeImpClassification<>;
    using daal_model_impl_t =
        daal::algorithms::decision_forest::classification::internal::ModelImpl;
    using daal_model_ptr_t = daal::algorithms::decision_forest::classification::ModelPtr;
};

template <>
struct daal_types_map<task::regression> {
    using daal_tree_impl_t = daal::algorithms::dtrees::internal::TreeImpRegression<>;
    using daal_model_impl_t = daal::algorithms::decision_forest::regression::internal::ModelImpl;
    using daal_model_ptr_t = daal::algorithms::decision_forest::regression::ModelPtr;
};

using namespace daal::algorithms::dtrees::internal;
using namespace daal::services::internal;

template <typename Float, typename Index, typename Task = task::by_default>
struct tree_level_record {
    using impl_const_t = impl_const<Task, Index>;

    tree_level_record(cl::sycl::queue queue,
                      dal::backend::primitives::ndarray<Index, 1> node_list,
                      dal::backend::primitives::ndarray<Float, 1> imp_list,
                      dal::backend::primitives::ndarray<Index, 1> class_hist_list,
                      Index node_count,
                      Index class_count,
                      const dal::backend::event_vector& deps = {})
            : node_count_(node_count),
              class_count_(class_count) {
        ONEDAL_ASSERT(node_list.get_count() == node_count * impl_const_t::node_prop_count_);
        ONEDAL_ASSERT(imp_list.get_count() == node_count * impl_const_t::node_imp_prop_count_);
        ONEDAL_ASSERT(class_hist_list.get_count() == node_count * class_count);

        // Add to host async??
        node_list_ = node_list.to_host(queue, deps);
        imp_list_ = imp_list.to_host(queue);
        class_hist_list_ = class_hist_list.to_host(queue);
    }

    Index get_node_count() {
        return node_count_;
    }

    Index get_row_count(Index node_idx) {
        return node_list_.get_data()[node_idx * impl_const_t::node_prop_count_ + 1];
    }
    Index get_feature_id(Index node_idx) {
        return node_list_.get_data()[node_idx * impl_const_t::node_prop_count_ + 2];
    }
    Index get_feature_bin(Index node_idx) {
        return node_list_.get_data()[node_idx * impl_const_t::node_prop_count_ + 3];
    }

    Index get_response(Index node_idx) {
        return node_list_.get_data()[node_idx * impl_const_t::node_prop_count_ + 5];
    }

    Float get_impurity(Index node_idx) {
        return imp_list_.get_data()[node_idx * impl_const_t::node_imp_prop_count_ + 0];
    }

    const Index* get_class_hist(Index node_idx) {
        return &(class_hist_list_.get_data()[node_idx * class_count_]);
    }

    bool is_leaf(Index node_idx) {
        return get_feature_id(node_idx) == impl_const_t::leaf_node_;
    }
    bool has_unordered_feature(Index node_idx) {
        return false; /* unordered features are not supported yet */
    }

    dal::backend::primitives::ndarray<Index, 1> node_list_;
    dal::backend::primitives::ndarray<Float, 1> imp_list_;
    dal::backend::primitives::ndarray<Index, 1> class_hist_list_;

    Index node_count_;
    Index class_count_;
};

template <typename Float, typename Index, typename Task = task::by_default>
class model_builder_interop {
    using model_t = model<Task>;
    using model_impl_t = detail::model_impl<Task>;
    using model_interop_impl_t =
        model_interop_impl<typename daal_types_map<Task>::daal_model_ptr_t>;
    using daal_model_impl_t = typename daal_types_map<Task>::daal_model_impl_t;
    using daal_model_ptr_t = typename daal_types_map<Task>::daal_model_ptr_t;
    using TreeType = typename daal_types_map<Task>::daal_tree_impl_t;
    using NodeType = typename TreeType::NodeType;

public:
    explicit model_builder_interop(Index tree_count, Index column_count, Index class_count)
            : allocator_(allocator_node_count_hint_),
              daal_model_ptr_(new daal_model_impl_t(column_count)),
              daal_model_interface_ptr_(daal_model_ptr_),
              class_count_(class_count),
              tree_list_(tree_count) {
        daal_model_ptr_->resize(tree_count);
    }

    void add_tree_block(std::vector<tree_level_record<Float, Index, Task>>& treeLevelsList,
                        std::vector<dal::backend::primitives::ndarray<Float, 1>>& binValues,
                        Index tree_count) {
        typedef std::vector<typename NodeType::Base*> DFTreeNodesArr;
        typedef dal::detail::shared<DFTreeNodesArr> DFTreeNodesArrPtr;

        DFTreeNodesArrPtr dfTreeLevelNodesPrev;
        bool unorderedFeaturesUsed = false;

        Index level = treeLevelsList.size();
        ONEDAL_ASSERT(level);
        //_P(" add tree first level %d", level);

        do {
            level--;
            tree_level_record<Float, Index, Task>& record = treeLevelsList[level];

            DFTreeNodesArrPtr dfTreeLevelNodes(new DFTreeNodesArr(record.get_node_count()));

            Index nSplits = 0;
            // nSplits is used to calculate index of child nodes on next level
            for (Index node_idx = 0; node_idx < record.get_node_count(); node_idx++) {
                if (record.is_leaf(node_idx)) {
                    (*dfTreeLevelNodes)[node_idx] = makeLeaf(record, node_idx);
                }
                else {
                    ONEDAL_ASSERT(dfTreeLevelNodesPrev.get());
                    (*dfTreeLevelNodes)[node_idx] =
                        makeSplit(record,
                                  binValues,
                                  node_idx,
                                  (*dfTreeLevelNodesPrev)[nSplits * 2],
                                  (*dfTreeLevelNodesPrev)[nSplits * 2 + 1]);
                    nSplits++;
                }
            }

            dfTreeLevelNodesPrev = dfTreeLevelNodes;
        } while (level > 0);

        ONEDAL_ASSERT(static_cast<size_t>(last_tree_pos_ + tree_count) <= tree_list_.size());

        for (Index tree_idx = 0; tree_idx < tree_count; tree_idx++) {
            tree_list_[last_tree_pos_ + tree_idx].reset((*dfTreeLevelNodesPrev)[tree_idx],
                                                        unorderedFeaturesUsed);
            daal_model_ptr_->add(tree_list_[last_tree_pos_ + tree_idx],
                                 class_count_,
                                 last_tree_pos_ + tree_idx);
        }
    }

    Float get_tree_response(Index tree_idx, const Float* x) const {
        ONEDAL_ASSERT(static_cast<size_t>(tree_idx) < tree_list_.size());
        const typename NodeType::Base* pNode =
            daal::algorithms::dtrees::prediction::internal::findNode<Float, TreeType, daal::sse2>(
                tree_list_[tree_idx],
                x);
        DAAL_ASSERT(pNode);
        //Float resp = NodeType::castLeaf(pNode)->response.value;
        //_P("pnode %p, resp %.6f", (void*)pNode, resp);
        return NodeType::castLeaf(pNode)->response.value;
    }

    model_t get_model() {
        const auto model_impl =
            std::make_shared<model_impl_t>(new model_interop_impl_t{ daal_model_interface_ptr_ });
        model_impl->tree_count = daal_model_ptr_->getNumberOfTrees();
        model_impl->class_count = daal_model_ptr_->getNumberOfClasses();

        return dal::detail::make_private<model_t>(model_impl);
    }

private:
    //typename NodeType::Leaf * makeLeaf(Index n, Index response, Float impurity, Float * hist, size_t nClasses)
    typename NodeType::Leaf* makeLeaf(tree_level_record<Float, Index, Task>& record,
                                      Index node_idx) {
        typename NodeType::Leaf* pNode = allocator_.allocLeaf(class_count_);
        DAAL_ASSERT(record.get_row_count(node_idx) > 0);
        pNode->response.value = record.get_response(node_idx);
        pNode->count = record.get_row_count(node_idx);
        pNode->impurity = record.get_impurity(node_idx);

        //_P("leaf node %d, resp %d, rc %d, imp %.6f ", node_idx, record.get_response(node_idx), record.get_row_count(node_idx), record.get_impurity(node_idx));
        if constexpr (std::is_same_v<task::classification, Task>) {
            const Index* hist_ptr = record.get_class_hist(node_idx);
            for (Index i = 0; i < class_count_; i++) {
                pNode->hist[i] = static_cast<Float>(hist_ptr[i]);
            }
        }

        return pNode;
    }

    //typename NodeType::Split * makeSplit(size_t n, size_t iFeature, Float featureValue, bool bUnordered, Float impurity,
    //                                     typename NodeType::Base * left, typename NodeType::Base * right)
    typename NodeType::Split* makeSplit(
        tree_level_record<Float, Index, Task>& record,
        std::vector<dal::backend::primitives::ndarray<Float, 1>>& feature_value_arr,
        Index node_idx,
        typename NodeType::Base* left,
        typename NodeType::Base* right) {
        typename NodeType::Split* pNode = allocator_.allocSplit();
        pNode->set(record.get_feature_id(node_idx),
                   feature_value_arr[record.get_feature_id(node_idx)]
                       .get_data()[record.get_feature_bin(node_idx)],
                   record.has_unordered_feature(node_idx));
        pNode->kid[0] = left;
        pNode->kid[1] = right;
        pNode->count = record.get_row_count(node_idx);
        pNode->impurity = record.get_impurity(node_idx);

        return pNode;
    }

private:
    constexpr static Index allocator_node_count_hint_ =
        512; //number of nodes as a hint for allocator to grow by
    typename TreeType::Allocator allocator_;
    daal_model_impl_t* daal_model_ptr_;
    daal_model_ptr_t daal_model_interface_ptr_;

    Index last_tree_pos_ = 0;
    Index class_count_ = 0;

    std::vector<TreeType> tree_list_;
};

} // namespace oneapi::dal::decision_forest::backend

#endif
