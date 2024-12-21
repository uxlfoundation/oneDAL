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

#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/algo/decision_forest/common.hpp"
#include "oneapi/dal/algo/decision_forest/backend/model_impl.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/infer_misc_structs.hpp"

#include <daal/src/algorithms/dtrees/forest/classification/df_classification_model_impl.h>
#include <daal/src/algorithms/dtrees/forest/regression/df_regression_model_impl.h>
#include <daal/src/algorithms/dtrees/dtrees_predict_dense_default_impl.i>
#include <iostream>

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

template <typename Float, typename Index, typename Task = task::by_default>
class infer_model_manager {
    using infer_context_t = infer_context<Float, Index, Task>;
    using model_t = model<Task>;
    using model_impl_t = detail::model_impl<Task>;
    using model_interop_impl_t =
        model_interop_impl<typename daal_types_map<Task>::daal_model_ptr_t>;
    using daal_model_impl_t = typename daal_types_map<Task>::daal_model_impl_t;
    using daal_model_ptr_t = typename daal_types_map<Task>::daal_model_ptr_t;
    using daal_decision_tree_table_t = daal::algorithms::dtrees::internal::DecisionTreeTable;
    using daal_decision_tree_node_t = daal::algorithms::dtrees::internal::DecisionTreeNode;

public:
    explicit infer_model_manager(const sycl::queue& q,
                                 const infer_context_t& ctx,
                                 const model_t& model)
            : queue_(q),
              ctx_(ctx) {
        const daal_model_impl_t* const daal_model_ptr = get_daal_model(model);

        ONEDAL_ASSERT(dal::detail::integral_cast<std::size_t>(ctx_.tree_count) ==
                      daal_model_ptr->size());

        const auto tree_count = ctx_.tree_count;

        std::vector<const daal_decision_tree_table_t*> tree_list;
        tree_list.resize(tree_count);

        std::size_t tree_size_max = 0;
        for (Index i = 0; i < tree_count; ++i) {
            tree_list[i] = daal_model_ptr->at(i);
            tree_size_max = std::max(tree_size_max, tree_list[i]->getNumberOfRows());
        }

        if (tree_size_max > dal::detail::limits<Index>::max()) {
            throw domain_error(dal::detail::error_messages::input_model_tree_has_invalid_size());
        }

        max_tree_size_ = dal::detail::integral_cast<Index>(tree_size_max);
        std::cout << "overflow here 4" << std::endl;
        const Index tree_block_size = dal::detail::check_mul_overflow(max_tree_size_, tree_count);

        auto fi_list_host = dal::backend::primitives::ndarray<Index, 1>::empty({ tree_block_size });
        auto lc_list_host = dal::backend::primitives::ndarray<Index, 1>::empty({ tree_block_size });
        auto fv_list_host = dal::backend::primitives::ndarray<Float, 1>::empty({ tree_block_size });

        std::cout << "overflow here 5" << std::endl;
        Index mul_class_count_and_tree_in_group_count =
            dal::detail::check_mul_overflow(ctx_.class_count, ctx_.tree_in_group_count);
        std::cout << "overflow here 6" << std::endl;
        dal::detail::check_mul_overflow(ctx_.row_count, mul_class_count_and_tree_in_group_count);

        dal::backend::primitives::ndarray<Float, 1> probas_list_host;

        if (ctx_.voting_mode == voting_mode::weighted && daal_model_ptr->getProbas(0)) {
            std::cout << "overflow here 7" << std::endl;
            dal::detail::check_mul_overflow<std::int64_t>(tree_block_size, ctx_.class_count);
            probas_list_host = dal::backend::primitives::ndarray<Float, 1>::empty(
                { tree_block_size * ctx_.class_count });
            weighted_available_ = true;
        }

        for (Index tree_idx = 0; tree_idx < ctx_.tree_count; tree_idx++) {
            const Index tree_size = tree_list[tree_idx]->getNumberOfRows();
            const daal_decision_tree_node_t* const dt_node_list =
                static_cast<const daal_decision_tree_node_t*>((*tree_list[tree_idx]).getArray());

            Index* const fi = fi_list_host.get_mutable_data() + tree_idx * max_tree_size_;
            Index* const lc = lc_list_host.get_mutable_data() + tree_idx * max_tree_size_;
            Float* const fv = fv_list_host.get_mutable_data() + tree_idx * max_tree_size_;

            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for (Index i = 0; i < tree_size; i++) {
                fi[i] = static_cast<Index>(dt_node_list[i].featureIndex);
                lc[i] = static_cast<Index>(dt_node_list[i].leftIndexOrClass);
                fv[i] = static_cast<Float>(dt_node_list[i].featureValueOrResponse);
            }

            if (weighted_available_) {
                const double* probas = daal_model_ptr->getProbas(tree_idx);
                Float* pv = probas_list_host.get_mutable_data() +
                            tree_idx * max_tree_size_ * ctx_.class_count;
                PRAGMA_IVDEP
                PRAGMA_VECTOR_ALWAYS
                for (Index i = 0; i < tree_size * ctx_.class_count; i++) {
                    pv[i] = static_cast<Float>(probas[i]);
                }
            }
        }

        if (weighted_available_) {
            probas_list_ = probas_list_host.to_device(queue_);
        }

        ftr_idx_list_ = fi_list_host.to_device(queue_);
        lch_or_class_id_list_ = lc_list_host.to_device(queue_);
        ftr_val_or_resp_list_ = fv_list_host.to_device(queue_);
    }

    ~infer_model_manager() = default;

    Index get_max_tree_size() const {
        return max_tree_size_;
    }

    auto get_serialized_data() const {
        return std::make_tuple(ftr_idx_list_, lch_or_class_id_list_, ftr_val_or_resp_list_);
    }

    bool is_weighted_available() const {
        return weighted_available_;
    }

    dal::backend::primitives::ndarray<Float, 1> get_class_probabilities_list() const {
        return probas_list_;
    }

private:
    const daal_model_impl_t* const get_daal_model(const model_t& trained_model) {
        const model_interop* interop_model = dal::detail::get_impl(trained_model).get_interop();
        if (!interop_model) {
            throw dal::internal_error(
                dal::detail::error_messages::input_model_does_not_match_kernel_function());
        }

        daal_model_ptr_t daal_model_interface_ptr =
            static_cast<const model_interop_impl_t*>(interop_model)->get_model();

        return static_cast<const daal_model_impl_t* const>(daal_model_interface_ptr.get());
    }

    sycl::queue queue_;
    const infer_context_t& ctx_;

    Index max_tree_size_;
    bool weighted_available_ = false;

    dal::backend::primitives::ndarray<Index, 1> ftr_idx_list_;
    dal::backend::primitives::ndarray<Index, 1> lch_or_class_id_list_;
    dal::backend::primitives::ndarray<Float, 1> ftr_val_or_resp_list_;
    dal::backend::primitives::ndarray<Float, 1> probas_list_;
};

} // namespace oneapi::dal::decision_forest::backend

#endif
