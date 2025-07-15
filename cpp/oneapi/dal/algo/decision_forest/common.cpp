/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/algo/decision_forest/common.hpp"
#include "oneapi/dal/algo/decision_forest/backend/model_impl.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::decision_forest {
namespace detail {
namespace v1 {

inline void check_domain_cond(bool value, const char* description) {
    if (!(value))
        throw dal::domain_error(description);
}

template <typename Task>
class descriptor_impl : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    explicit descriptor_impl() {
        if constexpr (std::is_same_v<Task, task::classification>) {
            class_count = 2;
            min_observations_in_leaf_node = 1;
        }
        else if constexpr (std::is_same_v<Task, task::regression>) {
            class_count = -1;
            min_observations_in_leaf_node = 5;
        }
    }

    double observations_per_tree_fraction = 1.0;
    double impurity_threshold = 0.0;
    double min_weight_fraction_in_leaf_node = 0.0;
    double min_impurity_decrease_in_split_node = 0.0;

    std::int64_t class_count = 2;
    std::int64_t tree_count = 100;
    std::int64_t features_per_node = 0;
    std::int64_t max_tree_depth = 0;
    std::int64_t min_observations_in_leaf_node = 2;
    std::int64_t min_observations_in_split_node = 2;
    std::int64_t max_leaf_nodes = 0;
    std::int64_t max_bins = 256;
    std::int64_t min_bin_size = 5;

    error_metric_mode error_metric_mode_value = error_metric_mode::none;
    infer_mode infer_mode_value = infer_mode::class_responses;

    // The default engine has been switched from mt2203 to philox for GPU,
    // as philox is more efficient in terms of performance on GPU architectures.
    // Note: Due to this change, some conformance(not critical) tests might fail as a result.
    engine_type df_engine_type = engine_type::philox4x32x10;
    bool memory_saving_mode = false;
    bool bootstrap = true;
    bool local_trees_mode = false;
    splitter_mode splitter_mode_value = splitter_mode::best;

    variable_importance_mode variable_importance_mode_value = variable_importance_mode::none;
    voting_mode voting_mode_value = voting_mode::weighted;

    std::int64_t seed = 777;
};

template <typename Task>
descriptor_base<Task>::descriptor_base() : impl_(new descriptor_impl<Task>{}) {}

template <typename Task>
double descriptor_base<Task>::get_observations_per_tree_fraction() const {
    return impl_->observations_per_tree_fraction;
}

template <typename Task>
double descriptor_base<Task>::get_impurity_threshold() const {
    return impl_->impurity_threshold;
}

template <typename Task>
double descriptor_base<Task>::get_min_weight_fraction_in_leaf_node() const {
    return impl_->min_weight_fraction_in_leaf_node;
}

template <typename Task>
double descriptor_base<Task>::get_min_impurity_decrease_in_split_node() const {
    return impl_->min_impurity_decrease_in_split_node;
}

template <typename Task>
std::int64_t descriptor_base<Task>::get_tree_count() const {
    return impl_->tree_count;
}

template <typename Task>
std::int64_t descriptor_base<Task>::get_features_per_node() const {
    return impl_->features_per_node;
}

template <typename Task>
std::int64_t descriptor_base<Task>::get_max_tree_depth() const {
    return impl_->max_tree_depth;
}

template <typename Task>
std::int64_t descriptor_base<Task>::get_min_observations_in_leaf_node() const {
    return impl_->min_observations_in_leaf_node;
}

template <typename Task>
std::int64_t descriptor_base<Task>::get_min_observations_in_split_node() const {
    return impl_->min_observations_in_split_node;
}

template <typename Task>
std::int64_t descriptor_base<Task>::get_max_leaf_nodes() const {
    return impl_->max_leaf_nodes;
}

template <typename Task>
std::int64_t descriptor_base<Task>::get_max_bins() const {
    return impl_->max_bins;
}

template <typename Task>
std::int64_t descriptor_base<Task>::get_min_bin_size() const {
    return impl_->min_bin_size;
}

template <typename Task>
error_metric_mode descriptor_base<Task>::get_error_metric_mode() const {
    return impl_->error_metric_mode_value;
}

template <typename Task>
bool descriptor_base<Task>::get_memory_saving_mode() const {
    return impl_->memory_saving_mode;
}

template <typename Task>
bool descriptor_base<Task>::get_local_trees_mode() const {
    return impl_->local_trees_mode;
}

template <typename Task>
bool descriptor_base<Task>::get_bootstrap() const {
    return impl_->bootstrap;
}

template <typename Task>
splitter_mode descriptor_base<Task>::get_splitter_mode() const {
    return impl_->splitter_mode_value;
}

template <typename Task>
variable_importance_mode descriptor_base<Task>::get_variable_importance_mode() const {
    return impl_->variable_importance_mode_value;
}

template <typename Task>
infer_mode descriptor_base<Task>::get_infer_mode_impl() const {
    return impl_->infer_mode_value;
}

template <typename Task>
std::int64_t descriptor_base<Task>::get_class_count_impl() const {
    return impl_->class_count;
}

template <typename Task>
voting_mode descriptor_base<Task>::get_voting_mode_impl() const {
    return impl_->voting_mode_value;
}

template <typename Task>
std::int64_t descriptor_base<Task>::get_seed() const {
    return impl_->seed;
}

template <typename Task>
engine_type descriptor_base<Task>::get_engine_type() const {
    return impl_->df_engine_type;
}

template <typename Task>
void descriptor_base<Task>::set_observations_per_tree_fraction_impl(double value) {
    check_domain_cond((value > 0.0 && value <= 1.0),
                      "observations_per_tree_fraction should be > 0.0 and <= 1.0");
    impl_->observations_per_tree_fraction = value;
}

template <typename Task>
void descriptor_base<Task>::set_impurity_threshold_impl(double value) {
    check_domain_cond((value >= 0.0), "impurity_threshold should be >= 0.0");
    impl_->impurity_threshold = value;
}

template <typename Task>
void descriptor_base<Task>::set_min_weight_fraction_in_leaf_node_impl(double value) {
    check_domain_cond((value >= 0.0 && value <= 0.5),
                      "min_weight_fraction_in_leaf_node should be >= 0.0 and <= 0.5");
    impl_->min_weight_fraction_in_leaf_node = value;
}

template <typename Task>
void descriptor_base<Task>::set_min_impurity_decrease_in_split_node_impl(double value) {
    check_domain_cond((value >= 0.0), "min_impurity_decrease_in_split_node should be >= 0.0");
    impl_->min_impurity_decrease_in_split_node = value;
}

template <typename Task>
void descriptor_base<Task>::set_tree_count_impl(std::int64_t value) {
    check_domain_cond((value > 0), "tree_count should be > 0");
    impl_->tree_count = value;
}

template <typename Task>
void descriptor_base<Task>::set_features_per_node_impl(std::int64_t value) {
    check_domain_cond((value >= 0), "features_per_node should be >= 0");
    impl_->features_per_node = value;
}

template <typename Task>
void descriptor_base<Task>::set_max_tree_depth_impl(std::int64_t value) {
    check_domain_cond((value >= 0), "max_tree_depth should be >= 0");
    impl_->max_tree_depth = value;
}

template <typename Task>
void descriptor_base<Task>::set_min_observations_in_leaf_node_impl(std::int64_t value) {
    check_domain_cond((value > 0), "min_observations_in_leaf_node should be > 0");
    impl_->min_observations_in_leaf_node = value;
}

template <typename Task>
void descriptor_base<Task>::set_min_observations_in_split_node_impl(std::int64_t value) {
    check_domain_cond((value > 0), "min_observations_in_split_node should be > 0");
    impl_->min_observations_in_split_node = value;
}

template <typename Task>
void descriptor_base<Task>::set_max_leaf_nodes_impl(std::int64_t value) {
    check_domain_cond((value >= 0), "max_leaf_nodes should be >= 0");
    impl_->max_leaf_nodes = value;
}

template <typename Task>
void descriptor_base<Task>::set_max_bins_impl(std::int64_t value) {
    check_domain_cond((value >= 2), "max_bins should be >= 2");
    impl_->max_bins = value;
}

template <typename Task>
void descriptor_base<Task>::set_min_bin_size_impl(std::int64_t value) {
    check_domain_cond((value >= 1), "min_bin_size should be >= 1");
    impl_->min_bin_size = value;
}

template <typename Task>
void descriptor_base<Task>::set_error_metric_mode_impl(error_metric_mode value) {
    impl_->error_metric_mode_value = value;
}

template <typename Task>
void descriptor_base<Task>::set_infer_mode_impl(infer_mode value) {
    impl_->infer_mode_value = value;
}

template <typename Task>
void descriptor_base<Task>::set_local_trees_mode_impl(bool value) {
    impl_->local_trees_mode = value;
}

template <typename Task>
void descriptor_base<Task>::set_memory_saving_mode_impl(bool value) {
    impl_->memory_saving_mode = value;
}

template <typename Task>
void descriptor_base<Task>::set_bootstrap_impl(bool value) {
    impl_->bootstrap = value;
}

template <typename Task>
void descriptor_base<Task>::set_splitter_mode_impl(splitter_mode value) {
    impl_->splitter_mode_value = value;
}

template <typename Task>
void descriptor_base<Task>::set_variable_importance_mode_impl(variable_importance_mode value) {
    impl_->variable_importance_mode_value = value;
}

template <typename Task>
void descriptor_base<Task>::set_class_count_impl(std::int64_t value) {
    check_domain_cond((value > 1), "class_count should be > 1");
    impl_->class_count = value;
}

template <typename Task>
void descriptor_base<Task>::set_voting_mode_impl(voting_mode value) {
    impl_->voting_mode_value = value;
}

template <typename Task>
void descriptor_base<Task>::set_seed_impl(std::int64_t value) {
    impl_->seed = value;
}

template <typename Task>
void descriptor_base<Task>::set_engine_type_impl(engine_type value) {
    impl_->df_engine_type = value;
}

template class ONEDAL_EXPORT descriptor_base<task::classification>;
template class ONEDAL_EXPORT descriptor_base<task::regression>;

} // namespace v1
} // namespace detail

namespace v1 {

using detail::v1::model_impl;

template <typename Task>
model<Task>::model() : impl_(new model_impl<Task>{}) {}

template <typename Task>
model<Task>::model(const std::shared_ptr<model_impl<Task>>& impl) : impl_(impl) {}

template <typename Task>
std::int64_t model<Task>::get_tree_count() const {
    return impl_->tree_count;
}

template <typename Task>
std::int64_t model<Task>::get_class_count_impl() const {
    return impl_->class_count;
}

template <typename Task>
void model<Task>::traverse_depth_first_impl(std::int64_t tree_idx,
                                            dtree_visitor_iface_t&& visitor) const {
    impl_->traverse_depth_first_impl(tree_idx, std::move(visitor));
}

template <typename Task>
void model<Task>::traverse_breadth_first_impl(std::int64_t tree_idx,
                                              dtree_visitor_iface_t&& visitor) const {
    impl_->traverse_breadth_first_impl(tree_idx, std::move(visitor));
}

template <typename Task>
void model<Task>::serialize(dal::detail::output_archive& ar) const {
    dal::detail::serialize_polymorphic_shared(impl_, ar);
}

template <typename Task>
void model<Task>::deserialize(dal::detail::input_archive& ar) {
    dal::detail::deserialize_polymorphic_shared(impl_, ar);
}

template class ONEDAL_EXPORT model<task::classification>;
template class ONEDAL_EXPORT model<task::regression>;

ONEDAL_REGISTER_SERIALIZABLE(model_impl<task::classification>)
ONEDAL_REGISTER_SERIALIZABLE(model_impl<task::regression>)
ONEDAL_REGISTER_SERIALIZABLE(backend::model_interop_cls)
ONEDAL_REGISTER_SERIALIZABLE(backend::model_interop_reg)

} // namespace v1
} // namespace oneapi::dal::decision_forest
