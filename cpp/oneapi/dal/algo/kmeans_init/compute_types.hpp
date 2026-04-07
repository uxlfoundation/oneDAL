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

#include "oneapi/dal/algo/kmeans_init/common.hpp"

namespace oneapi::dal::kmeans_init {

namespace detail {
namespace v1 {
template <typename Task>
class compute_input_impl;

template <typename Task>
class compute_result_impl;
} // namespace v1

using v1::compute_input_impl;
using v1::compute_result_impl;

} // namespace detail

namespace v1 {

/// @tparam Task Tag-type that specifies type of the problem to solve. Can
///              be :expr:`task::init`.
template <typename Task = task::by_default>
class compute_input : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the given :literal:`data`.
    compute_input(const table& data);

    /// Do not remove the destructor
    /// it is needed to properly handle the visibility of the class in the shared library
    /// while compiling with -fvisibility=hidden

    ~compute_input() override;

    /// Rule of five methods defined here due to the definition of the destructor.

    compute_input(const compute_input&);
    compute_input(compute_input&&) noexcept;
    compute_input& operator=(const compute_input&);
    compute_input& operator=(compute_input&&) noexcept;

    /// An $n \\times p$ table with the data to be clustered, where each row
    /// stores one feature vector.
    /// @remark default = table{}
    const table& get_data() const;

    auto& set_data(const table& data) {
        set_data_impl(data);
        return *this;
    }

protected:
    static void swap(compute_input<Task>& a, compute_input<Task>& b) noexcept;

private:
    void set_data_impl(const table& data);

    dal::detail::pimpl<detail::compute_input_impl<Task>> impl_;
};

/// @tparam Task Tag-type that specifies type of the problem to solve. Can
///              be :expr:`oneapi::dal::kmeans::task::clustering`.
template <typename Task = task::by_default>
class compute_result {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the default property values.
    compute_result();

    /// A $k \\times p$ table with the initial centroids. Each row of the
    /// table stores one centroid.
    /// @remark default = table{}
    const table& get_centroids() const;

    auto& set_centroids(const table& value) {
        set_centroids_impl(value);
        return *this;
    }

protected:
    void set_centroids_impl(const table&);

private:
    dal::detail::pimpl<detail::compute_result_impl<Task>> impl_;
};

} // namespace v1

using v1::compute_input;
using v1::compute_result;

} // namespace oneapi::dal::kmeans_init
