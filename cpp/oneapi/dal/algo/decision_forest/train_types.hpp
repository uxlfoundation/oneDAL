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

#include "oneapi/dal/algo/decision_forest/common.hpp"
#include "oneapi/dal/detail/parameters/system_parameters.hpp"

namespace oneapi::dal::decision_forest {

namespace detail {
namespace v1 {
template <typename Task>
class train_input_impl;

template <typename Task>
class train_result_impl;

template <typename Task>
struct train_parameters_impl;

template <typename Task = task::classification>
class train_parameters {};

template <>
class ONEDAL_EXPORT train_parameters<task::regression> : public dal::detail::system_parameters {
public:
    explicit train_parameters();

    /// Multiplier that defines the minimum work size for a thread to be assigned a separate task.
    /// For example, the value 4 means that a thread will be assigned a separate task
    /// that processes at least 4 topmost levels of a tree.
    std::int64_t get_min_part_coefficient() const;
    auto& set_min_part_coefficient(std::int64_t val) {
        set_min_part_coefficient_impl(val);
        return *this;
    }

    /// Multiplier that defines the minimum work size for a thread to be assigned a separate task.
    /// For example, the value 24000 means that a thread will be assigned a separate task
    /// that processes a 8-level tree built on at least 1000 observations and 10 features.
    std::int64_t get_min_size_coefficient() const;
    auto& set_min_size_coefficient(std::int64_t val) {
        set_min_size_coefficient_impl(val);
        return *this;
    }

    /// Check that the hyperparameters of the algorithm belong to the expected ranges
    void check_ranges() const;

private:
    void set_min_part_coefficient_impl(std::int64_t val);
    void set_min_size_coefficient_impl(std::int64_t val);
    dal::detail::pimpl<train_parameters_impl<task::regression>> impl_;
};

template <>
class ONEDAL_EXPORT train_parameters<task::classification> : public dal::detail::system_parameters {
public:
    explicit train_parameters();

    /// The threshold value to determine if the number of classes is small enough to switch
    /// to the implementation that is optimized for small number of classes.
    /// @remark The maximum value is 8.
    std::int64_t get_small_classes_threshold() const;
    auto& set_small_classes_threshold(std::int64_t val) {
        set_small_classes_threshold_impl(val);
        return *this;
    }

    std::int64_t get_min_part_coefficient() const;
    auto& set_min_part_coefficient(std::int64_t val) {
        set_min_part_coefficient_impl(val);
        return *this;
    }

    std::int64_t get_min_size_coefficient() const;
    auto& set_min_size_coefficient(std::int64_t val) {
        set_min_size_coefficient_impl(val);
        return *this;
    }

    /// Check that the hyperparameters of the algorithm belong to the expected ranges
    void check_ranges() const;

private:
    void set_small_classes_threshold_impl(std::int64_t val);
    void set_min_part_coefficient_impl(std::int64_t val);
    void set_min_size_coefficient_impl(std::int64_t val);
    dal::detail::pimpl<train_parameters_impl<task::classification>> impl_;
};

} // namespace v1

using v1::train_input_impl;
using v1::train_result_impl;
using v1::train_parameters;

} // namespace detail

namespace v1 {

/// @tparam Task   Tag-type that specifies type of the problem to solve. Can
///                be :expr:`task::classification` or :expr:`task::regression`.
template <typename Task = task::by_default>
class train_result {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the default property values.
    train_result();

    /// The trained Decision Forest model
    /// @remark default = model<Task>{}
    const model<Task>& get_model() const;

    auto& set_model(const model<Task>& value) {
        set_model_impl(value);
        return *this;
    }

    /// A $1 \\times 1$ table containing cumulative out-of-bag error value.
    /// Computed when :literal:`error_metric_mode` set with :literal:`error_metric_mode::out_of_bag_error`
    /// @remark default = table{}
    const table& get_oob_err() const;

    auto& set_oob_err(const table& value) {
        set_oob_err_impl(value);
        return *this;
    }

    /// A $n \\times 1$ table containing out-of-bag error value per observation.
    /// Computed when :literal:`error_metric_mode` set with :literal:`error_metric_mode::out_of_bag_error_per_observation`
    /// @remark default = table{}
    const table& get_oob_err_per_observation() const;

    auto& set_oob_err_per_observation(const table& value) {
        set_oob_err_per_observation_impl(value);
        return *this;
    }

    /// A $1 \\times 1$ table containing cumulative out-of-bag error (accuracy) value.
    /// Computed when :literal:`error_metric_mode` set with :literal:`error_metric_mode::out_of_bag_error_accuracy`
    /// @remark default = table{}
    const table& get_oob_err_accuracy() const;

    auto& set_oob_err_accuracy(const table& value) {
        set_oob_err_accuracy_impl(value);
        return *this;
    }

    /// A $1 \\times 1$ table containing cumulative out-of-bag error (R2) value.
    /// Computed when :literal:`error_metric_mode` set with :literal:`error_metric_mode::out_of_bag_error_r2`
    /// @remark default = table{}
    const table& get_oob_err_r2() const;

    auto& set_oob_err_r2(const table& value) {
        set_oob_err_r2_impl(value);
        return *this;
    }

    /// A $n \\times c$ table containing decision function value per observation.
    /// Computed when :literal:`error_metric_mode` set with :literal:`error_metric_mode::out_of_bag_error_decision_function`
    /// @remark default = table{}
    const table& get_oob_err_decision_function() const;

    auto& set_oob_err_decision_function(const table& value) {
        set_oob_err_decision_function_impl(value);
        return *this;
    }

    /// A $n \\times 1$ table containing prediction value per observation.
    /// Computed when :literal:`error_metric_mode` set with :literal:`error_metric_mode::out_of_bag_error_prediction`
    /// @remark default = table{}
    const table& get_oob_err_prediction() const;

    auto& set_oob_err_prediction(const table& value) {
        set_oob_err_prediction_impl(value);
        return *this;
    }

    /// A $1 \\times p$ table containing variable importance value for each feature.
    /// Computed when :expr:`variable_importance_mode != variable_importance_mode::none`
    /// @remark default = table{}
    const table& get_var_importance() const;

    auto& set_var_importance(const table& value) {
        set_var_importance_impl(value);
        return *this;
    }

private:
    void set_model_impl(const model<Task>&);
    void set_oob_err_impl(const table&);
    void set_oob_err_per_observation_impl(const table&);
    void set_oob_err_accuracy_impl(const table&);
    void set_oob_err_r2_impl(const table&);
    void set_oob_err_decision_function_impl(const table&);
    void set_oob_err_prediction_impl(const table&);
    void set_var_importance_impl(const table&);

    dal::detail::pimpl<detail::train_result_impl<Task>> impl_;
};
} // namespace v1

namespace v2 {

/// @tparam Task   Tag-type that specifies type of the problem to solve. Can
///                be :expr:`task::classification` or :expr:`task::regression`.
template <typename Task = task::by_default>
class train_input : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the given :literal:`data`,
    /// :literal:`responses` and :literal:`weights` property values
    train_input(const table& data, const table& responses, const table& weights = table{});

    /// The training set $X$
    /// @remark default = table{}
    const table& get_data() const;

    auto& set_data(const table& value) {
        set_data_impl(value);
        return *this;
    }

    /// Vector of labels $y$ for the training set $X$
    /// @remark default = table{}
    [[deprecated]] const table& get_labels() const {
        return get_responses();
    }
    [[deprecated]] auto& set_labels(const table& value) {
        return set_responses(value);
    }

    /// Vector of responses $y$ for the training set $X$
    /// @remark default = table{}
    const table& get_responses() const;

    auto& set_responses(const table& value) {
        set_responses_impl(value);
        return *this;
    }

    /// The vector of weights $w$ for the training set $X$
    /// @remark default = table{}
    const table& get_weights() const;

    auto& set_weights(const table& value) {
        set_weights_impl(value);
        return *this;
    }

private:
    void set_data_impl(const table& value);
    void set_responses_impl(const table& value);
    void set_weights_impl(const table& value);

    dal::detail::pimpl<detail::train_input_impl<Task>> impl_;
};
} // namespace v2

using v1::train_result;
using v2::train_input;

} // namespace oneapi::dal::decision_forest
