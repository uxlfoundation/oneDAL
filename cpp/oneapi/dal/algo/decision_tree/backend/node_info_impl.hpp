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

#include "oneapi/dal/algo/decision_tree/common.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::decision_tree {

inline void check_domain_cond(bool value, const char* description) {
    if (!(value))
        throw dal::domain_error(description);
}

namespace detail {
namespace v1 {

template <typename Task>
class node_info_impl : public base {
public:
    std::int64_t level = 0;
    double impurity = 0.;
    std::int64_t sample_count = 0;
};

template <typename Task>
class split_node_info_impl : public node_info_impl<Task> {
public:
    std::int64_t feature_index = 0;
    double feature_value = 0.;
};

template <typename Task>
class leaf_node_info_impl;

template <>
class leaf_node_info_impl<task::classification> : public node_info_impl<task::classification> {
public:
    leaf_node_info_impl(std::int64_t class_count_) : class_count(class_count_) {
        check_domain_cond((class_count_ > 1), dal::detail::error_messages::class_count_leq_one());
    }
    std::int64_t response = 0;
    const double* prob = nullptr;
    std::int64_t class_count;
};

template <>
class leaf_node_info_impl<task::regression> : public node_info_impl<task::regression> {
public:
    double response = 0;
};

template <typename T>
using enable_if_classification_t =
    std::enable_if_t<std::is_same_v<std::decay_t<T>, task::classification>>;

template <typename T>
using enable_if_regression_t = std::enable_if_t<std::is_same_v<std::decay_t<T>, task::regression>>;

} // namespace v1

using v1::node_info_impl;
using v1::split_node_info_impl;
using v1::leaf_node_info_impl;
using v1::enable_if_classification_t;
using v1::enable_if_regression_t;

} // namespace detail

} // namespace oneapi::dal::decision_tree
