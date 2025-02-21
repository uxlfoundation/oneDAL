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

#include "oneapi/dal/algo/basic_statistics/common.hpp"

namespace oneapi::dal::basic_statistics {

namespace detail {
namespace v1 {
template <typename Task>
class compute_input_impl;

template <typename Task>
class compute_result_impl;

template <typename Task>
class partial_compute_result_impl;
} // namespace v1

using v1::compute_input_impl;
using v1::compute_result_impl;
using v1::partial_compute_result_impl;

} // namespace detail

namespace v1 {

/// @tparam Task Tag-type that specifies the type of the problem to solve. Can
///              be :expr:`task::compute`.
template <typename Task = task::by_default>
class compute_input : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;
    compute_input();
    /// Creates a new instance of the class with the given :literal:`data`
    /// property value
    compute_input(const table& data);
    compute_input(const table& data, const table& weights);

    /// An $n \\times p$ table with the training data, where each row stores one
    /// feature vector.
    /// @remark default = table{}
    const table& get_data() const;

    auto& set_data(const table& data) {
        set_data_impl(data);
        return *this;
    }

    const table& get_weights() const;

    auto& set_weights(const table& weights) {
        set_weights_impl(weights);
        return *this;
    }

protected:
    void set_data_impl(const table& data);
    void set_weights_impl(const table& weights);

private:
    dal::detail::pimpl<detail::compute_input_impl<Task>> impl_;
};

/// @tparam Task Tag-type that specifies the type of the problem to solve. Can
///              be :expr:`task::compute`.
template <typename Task = task::by_default>
class compute_result : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the default property values.
    compute_result();

    /// A $1 \\times p$ table, where element $j$ is the minimum result for feature $j$.
    /// @remark default = table{}
    const table& get_min() const;

    /// A $1 \\times p$ table, where element $j$ is the maximum result for feature $j$.
    /// @remark default = table{}
    const table& get_max() const;

    /// A $1 \\times p$ table, where element $j$ is the sum result for feature $j$.
    /// @remark default = table{}
    const table& get_sum() const;

    /// A $1 \\times p$ table, where element $j$ is the sum_squares result for feature $j$.
    /// @remark default = table{}
    const table& get_sum_squares() const;

    /// A $1 \\times p$ table, where element $j$ is the sum_squares_centered result for feature $j$.
    /// @remark default = table{}
    const table& get_sum_squares_centered() const;

    /// A $1 \\times p$ table, where element $j$ is the mean result for feature $j$.
    /// @remark default = table{}
    const table& get_mean() const;

    /// A $1 \\times p$ table, where element $j$ is the second_order_raw_moment result for feature $j$.
    /// @remark default = table{}
    const table& get_second_order_raw_moment() const;

    /// A $1 \\times p$ table, where element $j$ is the variance result for feature $j$.
    /// @remark default = table{}
    const table& get_variance() const;

    /// A $1 \\times p$ table, where element $j$ is the standard_deviation result for feature $j$.
    /// @remark default = table{}
    const table& get_standard_deviation() const;

    /// A $1 \\times p$ table, where element $j$ is the variation result for feature $j$.
    /// @remark default = table{}
    const table& get_variation() const;

    auto& set_min(const table& value) {
        set_min_impl(value);
        return *this;
    }

    auto& set_max(const table& value) {
        set_max_impl(value);
        return *this;
    }

    auto& set_sum(const table& value) {
        set_sum_impl(value);
        return *this;
    }

    auto& set_sum_squares(const table& value) {
        set_sum_squares_impl(value);
        return *this;
    }

    auto& set_sum_squares_centered(const table& value) {
        set_sum_squares_centered_impl(value);
        return *this;
    }

    auto& set_mean(const table& value) {
        set_mean_impl(value);
        return *this;
    }

    auto& set_second_order_raw_moment(const table& value) {
        set_second_order_raw_moment_impl(value);
        return *this;
    }

    auto& set_variance(const table& value) {
        set_variance_impl(value);
        return *this;
    }

    auto& set_standard_deviation(const table& value) {
        set_standard_deviation_impl(value);
        return *this;
    }

    auto& set_variation(const table& value) {
        set_variation_impl(value);
        return *this;
    }

    /// Result options that indicates availability of the properties
    /// @remark default = full set of result_options
    const result_option_id& get_result_options() const;

    auto& set_result_options(const result_option_id& value) {
        set_result_options_impl(value);
        return *this;
    }

protected:
    void set_min_impl(const table& value);

    void set_max_impl(const table& value);

    void set_sum_impl(const table& value);

    void set_sum_squares_impl(const table& value);

    void set_sum_squares_centered_impl(const table& value);

    void set_mean_impl(const table& value);

    void set_second_order_raw_moment_impl(const table& value);

    void set_variance_impl(const table& value);

    void set_standard_deviation_impl(const table& value);

    void set_variation_impl(const table& value);

    void set_result_options_impl(const result_option_id&);

private:
    dal::detail::pimpl<detail::compute_result_impl<Task>> impl_;
};

template <typename Task = task::by_default>
class partial_compute_result : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    partial_compute_result();

    /// The nobs value.
    /// @remark default = table{}
    const table& get_partial_n_rows() const;

    auto& set_partial_n_rows(const table& value) {
        set_partial_n_rows_impl(value);
        return *this;
    }

    /// A $1 \\times p$ table, where element $j$ is the minimum current result for feature $j$.
    /// @remark default = table{}
    const table& get_partial_min() const;

    auto& set_partial_min(const table& value) {
        set_partial_min_impl(value);
        return *this;
    }

    /// A $1 \\times p$ table, where element $j$ is the maximum current result for feature $j$.
    /// @remark default = table{}
    const table& get_partial_max() const;

    auto& set_partial_max(const table& value) {
        set_partial_max_impl(value);
        return *this;
    }

    /// A $1 \\times p$ table, where element $j$ is the sum result of current blocks for feature $j$.
    /// @remark default = table{}
    const table& get_partial_sum() const;

    auto& set_partial_sum(const table& value) {
        set_partial_sum_impl(value);
        return *this;
    }

    /// A $1 \\times p$ table, where element $j$ is the sum_squares result of current blocks for feature $j$.
    /// @remark default = table{}
    const table& get_partial_sum_squares() const;

    auto& set_partial_sum_squares(const table& value) {
        set_partial_sum_squares_impl(value);
        return *this;
    }

    /// A $1 \\times p$ table, where element $j$ is the sum_squares_centered result of current blocks for feature $j$.
    /// @remark default = table{}
    const table& get_partial_sum_squares_centered() const;

    auto& set_partial_sum_squares_centered(const table& value) {
        set_partial_sum_squares_centered_impl(value);
        return *this;
    }

protected:
    void set_partial_n_rows_impl(const table&);
    void set_partial_min_impl(const table&);
    void set_partial_max_impl(const table&);
    void set_partial_sum_impl(const table&);
    void set_partial_sum_squares_impl(const table&);
    void set_partial_sum_squares_centered_impl(const table&);

private:
    dal::detail::pimpl<detail::partial_compute_result_impl<Task>> impl_;
};

template <typename Task = task::by_default>
class partial_compute_input : protected compute_input<Task> {
public:
    using task_t = Task;

    partial_compute_input();

    partial_compute_input(const table& data);

    partial_compute_input(const partial_compute_result<Task>& prev, const table& data);

    partial_compute_input(const partial_compute_result<Task>& prev,
                          const table& data,
                          const table& weights);

    const table& get_data() const {
        return compute_input<Task>::get_data();
    }

    auto& set_data(const table& value) {
        compute_input<Task>::set_data(value);
        return *this;
    }

    const table& get_weights() const {
        return compute_input<Task>::get_weights();
    }

    auto& set_weights(const table& value) {
        compute_input<Task>::set_weights(value);
        return *this;
    }

    const partial_compute_result<Task>& get_prev() const {
        return prev_;
    }

    auto& set_prev(const partial_compute_result<Task>& value) {
        prev_ = value;
        return *this;
    }

private:
    partial_compute_result<Task> prev_;
};

} // namespace v1

using v1::compute_input;
using v1::compute_result;
using v1::partial_compute_input;
using v1::partial_compute_result;

} // namespace oneapi::dal::basic_statistics
