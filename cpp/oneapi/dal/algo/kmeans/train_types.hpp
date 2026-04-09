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

#include "oneapi/dal/algo/kmeans/common.hpp"

namespace oneapi::dal::kmeans {

namespace detail {
namespace v1 {
template <typename Task>
class train_input_impl;

template <typename Task>
class train_result_impl;
} // namespace v1

using v1::train_input_impl;
using v1::train_result_impl;

} // namespace detail

namespace v1 {

/// @tparam Task Tag-type that specifies type of the problem to solve. Can
///              be :expr:`task::clustering`.
template <typename Task = task::by_default>
class train_input : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    train_input(const table& data);

    /// Creates a new instance of the class with the given :literal:`data` and
    /// :literal:`initial_centroids`
    train_input(const table& data, const table& initial_centroids);

    // Do not remove the destructor.
    // It is needed to properly handle the visibility of the class in the shared library
    // while compiling with -fvisibility=hidden
    ~train_input() override;

    // Rule of five methods defined here due to the definition of the destructor.
    train_input(const train_input&);
    train_input(train_input&&) noexcept;
    train_input& operator=(const train_input&);
    train_input& operator=(train_input&&) noexcept;

    /// An $n \\times p$ table with the data to be clustered, where each row
    /// stores one feature vector.
    const table& get_data() const;

    auto& set_data(const table& data) {
        set_data_impl(data);
        return *this;
    }

    /// A $k \\times p$ table with the initial centroids, where each row
    /// stores one centroid.
    const table& get_initial_centroids() const;

    auto& set_initial_centroids(const table& data) {
        set_initial_centroids_impl(data);
        return *this;
    }

protected:
    void set_data_impl(const table& data);
    void set_initial_centroids_impl(const table& data);
    static void swap(train_input<Task>& a, train_input<Task>& b) noexcept;

private:
    dal::detail::pimpl<detail::train_input_impl<Task>> impl_;
};

/// @tparam Task Tag-type that specifies type of the problem to solve. Can
///              be :expr:`task::clustering`.
template <typename Task = task::by_default>
class train_result {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the default property values.
    train_result();

    /// The trained K-means model
    /// @remark default = model<Task>{}
    const model<Task>& get_model() const;

    auto& set_model(const model<Task>& value) {
        set_model_impl(value);
        return *this;
    }

    /// An $n \\times 1$ table with the labels $y_i$ assigned to the
    /// samples $x_i$ in the input data, $1 \\leq 1 \\leq n$.
    /// @remark default = table{}
    [[deprecated]] const table& get_labels() const {
        return get_responses();
    }
    [[deprecated]] auto& set_labels(const table& value) {
        return set_responses(value);
    }

    /// An $n \\times 1$ table with the responses $y_i$ assigned to the
    /// samples $x_i$ in the input data, $1 \\leq 1 \\leq n$.
    /// @remark default = table{}
    const table& get_responses() const;

    auto& set_responses(const table& value) {
        set_responses_impl(value);
        return *this;
    }

    /// The number of iterations performed by the algorithm.
    /// @remark default = 0
    /// @invariant :expr:`iteration_count >= 0`
    std::int64_t get_iteration_count() const;

    auto& set_iteration_count(std::int64_t value) {
        set_iteration_count_impl(value);
        return *this;
    }

    /// The value of the objective function $\\Phi_X(C)$, where $C$ is
    /// :expr:`model.centroids`.
    /// @invariant :expr:`objective_function_value >= 0.0`
    double get_objective_function_value() const;

    auto& set_objective_function_value(double value) {
        set_objective_function_value_impl(value);
        return *this;
    }

protected:
    void set_model_impl(const model<Task>&);
    void set_responses_impl(const table&);
    void set_iteration_count_impl(std::int64_t);
    void set_objective_function_value_impl(double);

private:
    dal::detail::pimpl<detail::train_result_impl<Task>> impl_;
};

} // namespace v1

using v1::train_input;
using v1::train_result;

} // namespace oneapi::dal::kmeans
