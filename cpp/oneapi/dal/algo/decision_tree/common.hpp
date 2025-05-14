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

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/util/common.hpp"

namespace oneapi::dal::decision_tree {

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

namespace detail {
namespace v1 {

template <typename Task>
constexpr bool is_valid_task_v =
    dal::detail::is_one_of_v<Task, task::classification, task::regression>;

template <typename Task>
class node_info_impl;

template <typename Task>
class split_node_info_impl;

template <typename Task>
class leaf_node_info_impl;

} // namespace v1

using v1::node_info_impl;
using v1::split_node_info_impl;
using v1::leaf_node_info_impl;
using v1::is_valid_task_v;
} // namespace detail

namespace v1 {

/// Class containing base node info in decision tree
template <typename Task = task::by_default>
class ONEDAL_EXPORT node_info : public base {
    static_assert(detail::is_valid_task_v<Task>);
    friend dal::detail::pimpl_accessor;

public:
    using task_t = Task;
    node_info();
    ~node_info();
    node_info(const node_info<task_t>&);
    node_info(node_info<task_t>&&);
    node_info<task_t>& operator=(const node_info<task_t>&);
    node_info<task_t>& operator=(node_info<task_t>&&);

    /// Number of connections between the node and the root
    std::int64_t get_level() const;
    /// Measure of the homogeneity of the response variable at the node (i.e., the value of the criterion)
    double get_impurity() const;
    /// Number of samples at the node
    std::int64_t get_sample_count() const;

protected:
    using impl_t = detail::node_info_impl<task_t>;

    explicit node_info(impl_t* impl);

    impl_t* impl_;
};

/// Class containing description of split node in decision tree
template <typename Task = task::by_default>
class split_node_info : public node_info<Task> {
    static_assert(detail::is_valid_task_v<Task>);
    friend dal::detail::pimpl_accessor;

    using impl_t = detail::split_node_info_impl<Task>;

public:
    using task_t = Task;
    split_node_info();
    split_node_info(const split_node_info<task_t>&);
    split_node_info(split_node_info<task_t>&&);
    split_node_info<task_t>& operator=(const split_node_info<task_t>&);
    split_node_info<task_t>& operator=(split_node_info<task_t>&&);

    /// Feature used for splitting the node
    std::int64_t get_feature_index() const;
    /// Threshold value at the node
    double get_feature_value() const;

private:
    explicit split_node_info(impl_t* impl);
};

template <typename Task>
class leaf_node_info;

/// Class containing description of leaf node in classification decision tree
template <>
class ONEDAL_EXPORT leaf_node_info<task::classification> : public node_info<task::classification> {
    using impl_t = detail::leaf_node_info_impl<task_t>;

public:
    explicit leaf_node_info(std::int64_t class_count);
    leaf_node_info(const leaf_node_info<task_t>&);
    leaf_node_info(leaf_node_info<task_t>&&);
    leaf_node_info<task_t>& operator=(const leaf_node_info<task_t>&);
    leaf_node_info<task_t>& operator=(leaf_node_info<task_t>&&);

    /// Label to be predicted when reaching the leaf
    [[deprecated]] std::int64_t get_label() const {
        return get_response();
    }

    /// Response to be predicted when reaching the leaf
    std::int64_t get_response() const;
    /// Probability estimation for the leaf for certain class
    double get_probability(std::int64_t class_idx) const;

private:
    explicit leaf_node_info(impl_t* impl);
};

/// Class containing description of leaf node in regression decision tree
template <>
class ONEDAL_EXPORT leaf_node_info<task::regression> : public node_info<task::regression> {
    using impl_t = detail::leaf_node_info_impl<task_t>;

public:
    leaf_node_info();
    leaf_node_info(const leaf_node_info<task_t>&);
    leaf_node_info(leaf_node_info<task_t>&&);
    leaf_node_info<task_t>& operator=(const leaf_node_info<task_t>&);
    leaf_node_info<task_t>& operator=(leaf_node_info<task_t>&&);

    /// Label to be predicted when reaching the leaf
    [[deprecated]] double get_label() const {
        return get_response();
    }

    /// Response to be predicted when reaching the leaf
    double get_response() const;

private:
    explicit leaf_node_info(impl_t* impl);
};

template <typename T>
struct is_leaf_node_info {
    static constexpr bool value = std::is_same_v<T, leaf_node_info<task::classification>> ||
                                  std::is_same_v<T, leaf_node_info<task::regression>>;
};

template <typename T>
struct is_split_node_info {
    static constexpr bool value = std::is_same_v<T, split_node_info<task::classification>> ||
                                  std::is_same_v<T, split_node_info<task::regression>>;
};

template <typename T>
inline constexpr bool is_leaf_node_info_v = is_leaf_node_info<T>::value;

template <typename T>
inline constexpr bool is_split_node_info_v = is_split_node_info<T>::value;

} // namespace v1

using v1::node_info;
using v1::split_node_info;
using v1::leaf_node_info;
using v1::is_leaf_node_info;
using v1::is_leaf_node_info_v;
using v1::is_split_node_info;
using v1::is_split_node_info_v;

} // namespace oneapi::dal::decision_tree
