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

#pragma once

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/util/common.hpp"
#include "oneapi/dal/algo/decision_tree/detail/node_visitor.hpp"
#include "oneapi/dal/detail/serialization.hpp"
#include "oneapi/dal/detail/threading.hpp"

namespace oneapi::dal::decision_forest {

namespace task {
namespace v1 {
/// Tag-type that parameterizes entities used for solving
/// :capterm:`classification problem <classification>`.
struct classification {};

/// Tag-type that parameterizes entities used for solving
/// :capterm:`regression problem <regression>`.
struct regression {};

/// Alias tag-type for classification task.
using by_default = classification;
} // namespace v1

using v1::classification;
using v1::regression;
using v1::by_default;

} // namespace task

namespace method {
namespace v1 {
/// Tag-type that denotes :ref:`dense <df_t_math_dense>` computational
/// method.
struct dense {};

/// Tag-type that denotes :ref:`hist <df_t_math_hist>` computational
/// method.
struct hist {};

/// Alias tag-type for :ref:`dense <df_t_math_dense>` computational
/// method.
using by_default = dense;
} // namespace v1

using v1::dense;
using v1::hist;
using v1::by_default;

} // namespace method

namespace v1 {

/// Available identifiers to specify the variable importance mode
enum class variable_importance_mode {
    /// Do not compute variable importance
    none,

    /// Mean Decrease Impurity.
    /// Computed as the sum of weighted impurity decreases for all nodes where the variable is used,
    /// averaged over all trees in the forest
    mdi,
    /// Mean Decrease Accuracy (permutation importance).
    /// For each tree, the prediction error on the out-of-bag portion of the data is computed
    /// (error rate for classification, MSE for regression).
    /// The same is done after permuting each predictor variable.
    /// The difference between the two are then averaged over all trees.
    mda_raw,

    /// Mean Decrease Accuracy (permutation importance).
    /// This is MDA_Raw value scaled by its standard deviation.
    mda_scaled
};

/// Available identifiers to specify the error metric mode
enum class error_metric_mode : std::uint64_t {
    /// Do not compute error metric
    none = 0x00000000ULL,
    /// Train produces $1 \\times 1$ table with cumulative prediction error for out of bag observations
    out_of_bag_error = 0x00000001ULL,
    /// Train produces $n \\times 1$ table with prediction error for out-of-bag observations
    out_of_bag_error_per_observation = 0x00000002ULL,
    /// Train produces $1 \\times 1$ table with cumulative prediction error (accuracy) for out of bag observations
    out_of_bag_error_accuracy = 0x00000004ULL,
    /// Train produces $1 \\times 1$ table with cumulative prediction error (R2) for out of bag observations
    out_of_bag_error_r2 = 0x00000008ULL,
    /// Train produces $n \\times c$ table with decision function for out-of-bag observations
    out_of_bag_error_decision_function = 0x00000010ULL,
    /// Train produces $n \\times 1$ table with prediction for out-of-bag observations
    out_of_bag_error_prediction = 0x00000020ULL
};

/// Available identifiers to specify the infer mode
enum class infer_mode : std::uint64_t {
    /// Infer produces a $n \\times 1$  table with the predicted labels
    class_labels = 0x00000001ULL, /// deprecated
    /// Infer produces a $n \\times 1$  table with the predicted responses
    class_responses = class_labels,
    /// Infer produces $n \\times c$ table with the predicted class probabilities for each observation
    class_probabilities = 0x00000002ULL
};

/// Available voting modes for averaging trees predictions
enum class voting_mode {
    /// The final prediction is combined through a weighted majority voting
    weighted,
    /// The final prediction is combined through a simple majority voting
    unweighted
};

/// Available splitting strategies for building trees
enum class splitter_mode {
    /// Threshold for a node is chosen as the best among all bins
    best,
    /// Threshold for a node is the best for a set chosen at random
    random
};

inline infer_mode operator|(infer_mode value_left, infer_mode value_right) {
    return bitwise_or(value_left, value_right);
}

inline infer_mode& operator|=(infer_mode& value_left, infer_mode value_right) {
    value_left = value_left | value_right;
    return value_left;
}

inline infer_mode operator&(infer_mode value_left, infer_mode value_right) {
    return bitwise_and(value_left, value_right);
}

inline infer_mode& operator&=(infer_mode& value_left, infer_mode value_right) {
    value_left = value_left & value_right;
    return value_left;
}

inline error_metric_mode operator&(error_metric_mode value_left, error_metric_mode value_right) {
    return bitwise_and(value_left, value_right);
}

inline error_metric_mode& operator&=(error_metric_mode& value_left, error_metric_mode value_right) {
    value_left = value_left & value_right;
    return value_left;
}

inline error_metric_mode operator|(error_metric_mode value_left, error_metric_mode value_right) {
    return bitwise_or(value_left, value_right);
}

inline error_metric_mode& operator|=(error_metric_mode& value_left, error_metric_mode value_right) {
    value_left = value_left | value_right;
    return value_left;
}

} // namespace v1

using v1::variable_importance_mode;
using v1::error_metric_mode;
using v1::infer_mode;
using v1::voting_mode;
using v1::splitter_mode;

namespace detail {
namespace v1 {
struct descriptor_tag {};

template <typename Task>
class descriptor_impl;

template <typename Task>
class model_impl;

template <typename T>
using enable_if_classification_t =
    std::enable_if_t<std::is_same_v<std::decay_t<T>, task::classification>>;

template <typename Float>
constexpr bool is_valid_float_v = dal::detail::is_one_of_v<Float, float, double>;

template <typename Method>
constexpr bool is_valid_method_v = dal::detail::is_one_of_v<Method, method::dense, method::hist>;

template <typename Task>
constexpr bool is_valid_task_v =
    dal::detail::is_one_of_v<Task, task::classification, task::regression>;

/// @tparam Task   Tag-type that specifies the type of the problem to solve. Can
///                be :expr:`task::classification` or :expr:`task::regression`.
template <typename Task = task::by_default>
class descriptor_base : public base {
    static_assert(is_valid_task_v<Task>);
    friend dal::detail::pimpl_accessor;

public:
    using tag_t = descriptor_tag;
    using float_t = float;
    using method_t = method::by_default;
    using task_t = Task;

    descriptor_base();

    double get_observations_per_tree_fraction() const;
    double get_impurity_threshold() const;
    double get_min_weight_fraction_in_leaf_node() const;
    double get_min_impurity_decrease_in_split_node() const;
    std::int64_t get_tree_count() const;
    std::int64_t get_features_per_node() const;
    std::int64_t get_max_tree_depth() const;
    std::int64_t get_min_observations_in_leaf_node() const;
    std::int64_t get_min_observations_in_split_node() const;
    std::int64_t get_max_leaf_nodes() const;
    std::int64_t get_max_bins() const;
    std::int64_t get_min_bin_size() const;
    bool get_memory_saving_mode() const;
    bool get_bootstrap() const;
    splitter_mode get_splitter_mode() const;
    error_metric_mode get_error_metric_mode() const;
    variable_importance_mode get_variable_importance_mode() const;

    template <typename T = Task, typename = enable_if_classification_t<T>>
    std::int64_t get_class_count() const {
        return get_class_count_impl();
    }

    template <typename T = Task, typename = enable_if_classification_t<T>>
    infer_mode get_infer_mode() const {
        return get_infer_mode_impl();
    }

    template <typename T = Task, typename = enable_if_classification_t<T>>
    voting_mode get_voting_mode() const {
        return get_voting_mode_impl();
    }

    std::int64_t get_seed() const;

protected:
    void set_observations_per_tree_fraction_impl(double value);
    void set_impurity_threshold_impl(double value);
    void set_min_weight_fraction_in_leaf_node_impl(double value);
    void set_min_impurity_decrease_in_split_node_impl(double value);
    void set_tree_count_impl(std::int64_t value);
    void set_features_per_node_impl(std::int64_t value);
    void set_max_tree_depth_impl(std::int64_t value);
    void set_min_observations_in_leaf_node_impl(std::int64_t value);
    void set_min_observations_in_split_node_impl(std::int64_t value);
    void set_max_leaf_nodes_impl(std::int64_t value);
    void set_max_bins_impl(std::int64_t value);
    void set_min_bin_size_impl(std::int64_t value);
    void set_memory_saving_mode_impl(bool value);
    void set_bootstrap_impl(bool value);
    void set_splitter_mode_impl(splitter_mode value);
    void set_error_metric_mode_impl(error_metric_mode value);
    void set_variable_importance_mode_impl(variable_importance_mode value);
    void set_class_count_impl(std::int64_t value);
    void set_infer_mode_impl(infer_mode value);
    void set_voting_mode_impl(voting_mode value);

    std::int64_t get_class_count_impl() const;
    infer_mode get_infer_mode_impl() const;
    voting_mode get_voting_mode_impl() const;

    void set_seed_impl(std::int64_t value);

private:
    dal::detail::pimpl<descriptor_impl<Task>> impl_;
};

template <typename Task>
struct decision_tree_task_map;

template <>
struct decision_tree_task_map<task::classification> {
    using tree_task_t = decision_tree::task::classification;
};

template <>
struct decision_tree_task_map<task::regression> {
    using tree_task_t = decision_tree::task::regression;
};

template <typename Task>
using decision_tree_task_map_t = typename decision_tree_task_map<Task>::tree_task_t;

template <typename Task>
using decision_tree_visitor_ptr =
    decision_tree::detail::node_visitor_ptr<decision_tree_task_map_t<Task>>;

} // namespace v1

using v1::descriptor_tag;
using v1::descriptor_impl;
using v1::model_impl;
using v1::descriptor_base;

using v1::enable_if_classification_t;
using v1::is_valid_float_v;
using v1::is_valid_method_v;
using v1::is_valid_task_v;

using v1::decision_tree_task_map;
using v1::decision_tree_task_map_t;
using v1::decision_tree_visitor_ptr;

} // namespace detail

namespace v1 {
/// @tparam Float  The floating-point type that the algorithm uses for
///                intermediate computations. Can be :expr:`float` or
///                :expr:`double`.
/// @tparam Method Tag-type that specifies an implementation of algorithm. Can
///                be :expr:`method::dense` or :expr:`method::hist`.
/// @tparam Task   Tag-type that specifies type of the problem to solve. Can
///                be :expr:`task::classification` or :expr:`task::regression`.
template <typename Float = float,
          typename Method = method::by_default,
          typename Task = task::by_default>
class descriptor : public detail::descriptor_base<Task> {
    static_assert(detail::is_valid_float_v<Float>);
    static_assert(detail::is_valid_method_v<Method>);
    static_assert(detail::is_valid_task_v<Task>);

    using base_t = detail::descriptor_base<Task>;

public:
    using float_t = Float;
    using method_t = Method;
    using task_t = Task;

    /// Creates a new instance of the class with the default property values.
    descriptor() = default;

    /// The fraction of observations per tree
    /// @invariant :expr:`observations_per_tree_fraction > 0.0`
    /// @invariant :expr:`observations_per_tree_fraction <= 1.0`
    /// @remark default = 1.0
    double get_observations_per_tree_fraction() const {
        return base_t::get_observations_per_tree_fraction();
    }

    auto& set_observations_per_tree_fraction(double value) {
        base_t::set_observations_per_tree_fraction_impl(value);
        return *this;
    }

    /// The impurity threshold, a node will be split if this split
    /// induces a decrease of the impurity greater than or equal to the input value
    /// @invariant :expr:`impurity_threshold >= 0.0`
    /// @remark default = 0.0
    double get_impurity_threshold() const {
        return base_t::get_impurity_threshold();
    }

    auto& set_impurity_threshold(double value) {
        base_t::set_impurity_threshold_impl(value);
        return *this;
    }

    /// The min weight fraction in a leaf node. The minimum weighted fraction of the
    /// total sum of weights (of all input observations) required to be at a leaf node
    /// @invariant :expr:`min_weight_fraction_in_leaf_node >= 0.0`
    /// @invariant :expr:`min_weight_fraction_in_leaf_node <= 0.5`
    /// @remark default = 0.0
    double get_min_weight_fraction_in_leaf_node() const {
        return base_t::get_min_weight_fraction_in_leaf_node();
    }

    auto& set_min_weight_fraction_in_leaf_node(double value) {
        base_t::set_min_weight_fraction_in_leaf_node_impl(value);
        return *this;
    }

    /// The min impurity decrease in a split node is a threshold for stopping the tree
    /// growth early. A node will be split if its impurity is above the threshold, otherwise
    /// it is a leaf.
    /// @invariant :expr:`min_impurity_decrease_in_split_node >= 0.0`
    /// @remark default = 0.0
    double get_min_impurity_decrease_in_split_node() const {
        return base_t::get_min_impurity_decrease_in_split_node();
    }

    auto& set_min_impurity_decrease_in_split_node(double value) {
        base_t::set_min_impurity_decrease_in_split_node_impl(value);
        return *this;
    }

    /// The number of trees in the forest.
    /// @invariant :expr:`tree_count > 0`
    /// @remark default = 100
    std::int64_t get_tree_count() const {
        return base_t::get_tree_count();
    }

    auto& set_tree_count(std::int64_t value) {
        base_t::set_tree_count_impl(value);
        return *this;
    }

    /// The number of features to consider when looking for the best split for a node.
    /// @remark default = task::classification ? sqrt(p) : p/3, where p is the total number of features
    std::int64_t get_features_per_node() const {
        return base_t::get_features_per_node();
    }

    auto& set_features_per_node(std::int64_t value) {
        base_t::set_features_per_node_impl(value);
        return *this;
    }

    /// The maximal depth of the tree. If 0, then nodes are expanded
    /// until all leaves are pure or until all leaves contain less or
    /// equal to min observations in leaf node samples.
    /// @remark default = 0
    std::int64_t get_max_tree_depth() const {
        return base_t::get_max_tree_depth();
    }

    auto& set_max_tree_depth(std::int64_t value) {
        base_t::set_max_tree_depth_impl(value);
        return *this;
    }

    /// The minimal number of observations in a leaf node.
    /// @invariant :expr:`min_observations_in_leaf_node > 0`
    /// @remark default = 1 for classification, 5 for regression
    std::int64_t get_min_observations_in_leaf_node() const {
        return base_t::get_min_observations_in_leaf_node();
    }

    auto& set_min_observations_in_leaf_node(std::int64_t value) {
        base_t::set_min_observations_in_leaf_node_impl(value);
        return *this;
    }

    /// The minimal number of observations in a split node.
    /// @invariant :expr:`min_observations_in_split_node > 1`
    /// @remark default = 2
    std::int64_t get_min_observations_in_split_node() const {
        return base_t::get_min_observations_in_split_node();
    }

    auto& set_min_observations_in_split_node(std::int64_t value) {
        base_t::set_min_observations_in_split_node_impl(value);
        return *this;
    }

    /// The maximal number of the leaf nodes. If 0, the number of leaf nodes is not limited.
    /// @remark default = 0
    std::int64_t get_max_leaf_nodes() const {
        return base_t::get_max_leaf_nodes();
    }

    auto& set_max_leaf_nodes(std::int64_t value) {
        base_t::set_max_leaf_nodes_impl(value);
        return *this;
    }

    /// The maximal number of discrete bins to bucket continuous features.
    /// Used with :expr:`method::hist` split-finding method only. Increasing
    /// the number results in higher computation costs.
    /// @invariant :expr:`max_bins > 1`
    /// @remark default = 256
    std::int64_t get_max_bins() const {
        return base_t::get_max_bins();
    }

    auto& set_max_bins(std::int64_t value) {
        base_t::set_max_bins_impl(value);
        return *this;
    }

    /// The minimal number of observations in a bin. Used with
    /// :expr:`method::hist` split-finding method only.
    /// @invariant :expr:`min_bin_size > 0`
    /// @remark default = 5
    std::int64_t get_min_bin_size() const {
        return base_t::get_min_bin_size();
    }

    auto& set_min_bin_size(std::int64_t value) {
        base_t::set_min_bin_size_impl(value);
        return *this;
    }

    /// The memory saving mode.
    /// @remark default = false
    bool get_memory_saving_mode() const {
        return base_t::get_memory_saving_mode();
    }

    auto& set_memory_saving_mode(bool value) {
        base_t::set_memory_saving_mode_impl(value);
        return *this;
    }

    /// The bootstrap mode, if true, the training set for a tree
    /// is a bootstrap of the whole training set, if False, the whole
    /// dataset is used to build each tree.
    /// @remark default = true
    bool get_bootstrap() const {
        return base_t::get_bootstrap();
    }

    auto& set_bootstrap(bool value) {
        base_t::set_bootstrap_impl(value);
        return *this;
    }

    /// Splitter strategy: if 'best', best threshold for each is
    /// selected. If 'random', threshold is selected randomly.
    /// @remark default = splitter_mode::best
    splitter_mode get_splitter_mode() const {
        return base_t::get_splitter_mode();
    }

    auto& set_splitter_mode(splitter_mode value) {
        base_t::set_splitter_mode_impl(value);
        return *this;
    }

    /// The error metric mode
    /// @remark default = error_metric_mode::none
    error_metric_mode get_error_metric_mode() const {
        return base_t::get_error_metric_mode();
    }

    auto& set_error_metric_mode(error_metric_mode value) {
        base_t::set_error_metric_mode_impl(value);
        return *this;
    }

    /// The variable importance mode
    /// @remark default = variable_importance_mode::none
    variable_importance_mode get_variable_importance_mode() const {
        return base_t::get_variable_importance_mode();
    }

    auto& set_variable_importance_mode(variable_importance_mode value) {
        base_t::set_variable_importance_mode_impl(value);
        return *this;
    }

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    /// The class count. Used with :expr:`task::classification` only.
    /// @remark default = 2
    std::int64_t get_class_count() const {
        return base_t::get_class_count_impl();
    }

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    auto& set_class_count(std::int64_t value) {
        base_t::set_class_count_impl(value);
        return *this;
    }

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    /// The infer mode. Used with :expr:`task::classification` only.
    infer_mode get_infer_mode() const {
        return base_t::get_infer_mode_impl();
    }

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    auto& set_infer_mode(infer_mode value) {
        base_t::set_infer_mode_impl(value);
        return *this;
    }

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    /// The voting mode. Used with :expr:`task::classification` only.
    voting_mode get_voting_mode() const {
        return base_t::get_voting_mode_impl();
    }

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    auto& set_voting_mode(voting_mode value) {
        base_t::set_voting_mode_impl(value);
        return *this;
    }

    /// Seed for the random numbers generator used by the algorithm
    /// @invariant :expr:`tree_count > 0`
    std::int64_t get_seed() const {
        return base_t::get_seed();
    }

    auto& set_seed(std::int64_t value) {
        base_t::set_seed_impl(value);
        return *this;
    }
};

/// @tparam Task   Tag-type that specifies the type of the problem to solve. Can
///                be :expr:`task::classification` or :expr:`task::regression`.
template <typename Task = task::by_default>
class model : public base {
    static_assert(detail::is_valid_task_v<Task>);
    friend dal::detail::pimpl_accessor;
    friend dal::detail::serialization_accessor;

    using dtree_task_t = detail::decision_tree_task_map_t<Task>;
    using dtree_visitor_iface_t = detail::decision_tree_visitor_ptr<Task>;

public:
    using task_t = Task;

    /// Creates a new instance of the class with the default property values.
    model();

    /// The number of trees in the forest.
    /// @invariant :expr:`tree_count > 0`
    /// @remark default = 100
    std::int64_t get_tree_count() const;

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    /// The class count. Used with :expr:`oneapi::dal::decision_forest::task::classification` only.
    /// @remark default = 2
    std::int64_t get_class_count() const {
        return get_class_count_impl();
    }

    /// Performs Depth First Traversal of i-th tree
    /// @param[in] tree_idx     Index of the tree to traverse
    /// @param[in] visitor      This functor gets notified when tree nodes are visited, via corresponding operators:
    ///                             bool operator()(const decision_forest::split_node_info<Task>&)
    ///                             bool operator()(const decision_forest::leaf_node_info<Task>&)
    template <typename Visitor>
    void traverse_depth_first(std::int64_t tree_idx, Visitor&& visitor) const {
        traverse_depth_first_impl(
            tree_idx,
            decision_tree::detail::make_node_visitor<dtree_task_t>(std::forward<Visitor>(visitor)));
    }

    /// Performs Depth First Traversal for all trees
    /// @param[in] visitor_array    This an array of functors which are notified when tree nodes are visited,
    ///                             via corresponding operators:
    ///                             bool operator()(const decision_forest::split_node_info<Task>&)
    ///                             bool operator()(const decision_forest::leaf_node_info<Task>&)
    template <typename T, typename Visitor>
    void traverse_depth_first(T&& visitor_array) const {
        dal::detail::threader_for(this->get_tree_count(),
                                  this->get_tree_count(),
                                  [&](std::int64_t i) {
                                      traverse_depth_first_impl(
                                          i,
                                          decision_tree::detail::make_node_visitor<dtree_task_t>(
                                              std::forward<Visitor>(visitor_array[i])));
                                  });
    }

    /// Performs Breadth First Traversal of i-th tree
    /// @param[in] tree_idx    Index of the tree to traverse
    /// @param[in] visitor      This functor gets notified when tree nodes are visited, via corresponding operators:
    ///                             bool operator()(const decision_forest::split_node_info<Task>&)
    ///                             bool operator()(const decision_forest::leaf_node_info<Task>&)
    template <typename Visitor>
    void traverse_breadth_first(std::int64_t tree_idx, Visitor&& visitor) const {
        traverse_breadth_first_impl(
            tree_idx,
            decision_tree::detail::make_node_visitor<dtree_task_t>(std::forward<Visitor>(visitor)));
    }

    /// Performs Breadth First Traversal for all trees
    /// @param[in] visitor_array    This an array of functors which are notified when tree nodes are visited,
    ///                             via corresponding operators:
    ///                             bool operator()(const decision_forest::split_node_info<Task>&)
    ///                             bool operator()(const decision_forest::leaf_node_info<Task>&)
    template <typename T, typename Visitor>
    void traverse_breadth_first(T&& visitor_array) const {
        dal::detail::threader_for(this->get_tree_count(),
                                  this->get_tree_count(),
                                  [&](std::int64_t i) {
                                      traverse_breadth_first_impl(
                                          i,
                                          decision_tree::detail::make_node_visitor<dtree_task_t>(
                                              std::forward<Visitor>(visitor_array[i])));
                                  });
    }

protected:
    std::int64_t get_class_count_impl() const;

    void traverse_depth_first_impl(std::int64_t tree_idx, dtree_visitor_iface_t&& visitor) const;
    void traverse_breadth_first_impl(std::int64_t tree_idx, dtree_visitor_iface_t&& visitor) const;

private:
    void serialize(dal::detail::output_archive& ar) const;
    void deserialize(dal::detail::input_archive& ar);

    explicit model(const std::shared_ptr<detail::model_impl<Task>>& impl);
    dal::detail::pimpl<detail::model_impl<Task>> impl_;
};

template <typename Task>
using node_info = decision_tree::node_info<detail::decision_tree_task_map_t<Task>>;

template <typename Task>
using leaf_node_info = decision_tree::leaf_node_info<detail::decision_tree_task_map_t<Task>>;

template <typename Task>
using split_node_info = decision_tree::split_node_info<detail::decision_tree_task_map_t<Task>>;

template <typename Task>
using is_leaf_node_info = decision_tree::is_leaf_node_info<detail::decision_tree_task_map_t<Task>>;

template <typename Task>
using is_split_node_info =
    decision_tree::is_split_node_info<detail::decision_tree_task_map_t<Task>>;

template <typename Task>
inline constexpr bool is_leaf_node_info_v = is_leaf_node_info<Task>::value;

template <typename Task>
inline constexpr bool is_split_node_info_v = is_split_node_info<Task>::value;
} // namespace v1

using v1::descriptor;
using v1::model;
using v1::node_info;
using v1::leaf_node_info;
using v1::split_node_info;
using v1::is_leaf_node_info;
using v1::is_leaf_node_info_v;
using v1::is_split_node_info;
using v1::is_split_node_info_v;

} // namespace oneapi::dal::decision_forest
