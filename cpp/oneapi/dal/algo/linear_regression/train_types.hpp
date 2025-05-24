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

#include "oneapi/dal/algo/linear_regression/common.hpp"
#include "oneapi/dal/detail/parameters/system_parameters.hpp"

namespace oneapi::dal::linear_regression {

namespace detail {
namespace v1 {
template <typename Task>
class train_input_impl;

template <typename Task>
class train_result_impl;

template <typename Task>
struct train_parameters_impl;

template <typename Task>
class partial_train_result_impl;

template <typename Task = task::by_default>
class train_parameters : public dal::detail::system_parameters {
public:
    explicit train_parameters();
    train_parameters(train_parameters&&) = default;
    train_parameters(const train_parameters&) = default;

    /// Linear regression training algorithm splits input data table into blocks of rows
    /// to speedup the computations.
    /// These API define the number of rows in the data block to be processed on CPU.
    std::int64_t get_cpu_macro_block() const;
    auto& set_cpu_macro_block(std::int64_t val) {
        set_cpu_macro_block_impl(val);
        return *this;
    }

    /// Linear regression training algorithm splits input data table into blocks of rows
    /// to speedup the computations.
    /// These API define the number of rows in the data block to be processed on GPU.
    std::int64_t get_gpu_macro_block() const;
    auto& set_gpu_macro_block(std::int64_t val) {
        set_gpu_macro_block_impl(val);
        return *this;
    }

    /// Maximum number of columns in the dataset to use blocked data processing on CPU.
    /// Otherwise, the whole dataset is processed at once to compute the coefficients.
    std::int64_t get_cpu_max_cols_batched() const;
    auto& set_cpu_max_cols_batched(std::int64_t val) {
        set_cpu_max_cols_batched_impl(val);
        return *this;
    }

    /// Minimum number of rows in the dataset to use blocked data processing on CPU.
    /// Otherwise, the whole dataset is processed at once to compute the coefficients.
    std::int64_t get_cpu_small_rows_threshold() const;
    auto& set_cpu_small_rows_threshold(std::int64_t val) {
        set_cpu_small_rows_threshold_impl(val);
        return *this;
    }

    std::int64_t get_cpu_small_rows_max_cols_batched() const;
    auto& set_cpu_small_rows_max_cols_batched(std::int64_t val) {
        set_cpu_small_rows_max_cols_batched_impl(val);
        return *this;
    }

private:
    void set_cpu_macro_block_impl(std::int64_t val);
    void set_gpu_macro_block_impl(std::int64_t val);
    void set_cpu_max_cols_batched_impl(std::int64_t val);
    void set_cpu_small_rows_threshold_impl(std::int64_t val);
    void set_cpu_small_rows_max_cols_batched_impl(std::int64_t val);
    dal::detail::pimpl<train_parameters_impl<Task>> impl_;
};

} // namespace v1

using v1::train_parameters;
using v1::train_input_impl;
using v1::train_result_impl;
using v1::partial_train_result_impl;

} // namespace detail

namespace v1 {

/// @tparam Task Tag-type that specifies type of the problem to solve. Can
///              be :expr:`task::regression`.
template <typename Task = task::by_default>
class train_input : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    train_input();

    /// Creates a new instance of the class with the given :literal:`data`
    /// and :literal:`responses` property values
    train_input(const table& data, const table& responses);

    train_input(const table& data);

    /// The training set X
    /// @remark default = table{}
    const table& get_data() const;

    auto& set_data(const table& data) {
        set_data_impl(data);
        return *this;
    }

    /// Vector of responses y for the training set X
    /// @remark default = table{}
    const table& get_responses() const;

    auto& set_responses(const table& responses) {
        set_data_impl(responses);
        return *this;
    }

protected:
    void set_data_impl(const table& data);
    void set_responses_impl(const table& responses);

private:
    dal::detail::pimpl<detail::train_input_impl<Task>> impl_;
};

/// @tparam Task Tag-type that specifies type of the problem to solve. Can
///              be :expr:`task::classification` or :expr:`task::search`.
template <typename Task = task::by_default>
class train_result {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the default property values.
    train_result();

    /// The trained Linear Regression model
    /// @remark default = model<Task>{}
    const model<Task>& get_model() const;

    auto& set_model(const model<Task>& value) {
        set_model_impl(value);
        return *this;
    }

    /// Table of Linear regression intercept
    const table& get_intercept() const;

    auto& set_intercept(const table& value) {
        set_intercept_impl(value);
        return *this;
    }

    /// Table of Linear regression coefficients
    const table& get_coefficients() const;

    auto& set_coefficients(const table& value) {
        set_coefficients_impl(value);
        return *this;
    }

    /// Table of Linear regression coefficients with intercept
    const table& get_packed_coefficients() const;

    auto& set_packed_coefficients(const table& value) {
        set_packed_coefficients_impl(value);
        return *this;
    }

    /// Result options that indicates availability of the properties
    const result_option_id& get_result_options() const;

    auto& set_result_options(const result_option_id& value) {
        set_result_options_impl(value);
        return *this;
    }

protected:
    void set_model_impl(const model<Task>&);

    void set_intercept_impl(const table&);
    void set_coefficients_impl(const table&);
    void set_packed_coefficients_impl(const table&);

    void set_result_options_impl(const result_option_id&);

private:
    dal::detail::pimpl<detail::train_result_impl<Task>> impl_;
};

template <typename Task = task::by_default>
class partial_train_result : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    partial_train_result();

    /// The partial_xtx matrix.
    /// @remark default = table{}
    const table& get_partial_xtx() const;

    auto& set_partial_xtx(const table& value) {
        set_partial_xtx_impl(value);
        return *this;
    }

    /// The partial_xty matrix.
    /// @remark default = table{}
    const table& get_partial_xty() const;

    auto& set_partial_xty(const table& value) {
        set_partial_xty_impl(value);
        return *this;
    }

protected:
    void set_partial_xtx_impl(const table&);
    void set_partial_xty_impl(const table&);

private:
    dal::detail::pimpl<detail::partial_train_result_impl<Task>> impl_;
};

template <typename Task = task::by_default>
class partial_train_input : protected train_input<Task> {
public:
    using task_t = Task;

    partial_train_input();

    partial_train_input(const table& data);

    partial_train_input(const table& data, const table& responses);

    partial_train_input(const partial_train_result<Task>& prev, const table& data);

    partial_train_input(const partial_train_result<Task>& prev,
                        const table& data,
                        const table& responses);

    partial_train_input(const partial_train_result<Task>& prev,
                        const partial_train_input<Task>& input);

    const table& get_data() const {
        return train_input<Task>::get_data();
    }

    auto& set_data(const table& value) {
        train_input<Task>::set_data(value);
        return *this;
    }

    /// Vector of responses y for the training set X
    /// @remark default = table{}
    const table& get_responses() const {
        return train_input<Task>::get_responses();
    }

    auto& set_responses(const table& responses) {
        train_input<Task>::set_responses(responses);
        return *this;
    }

    const partial_train_result<Task>& get_prev() const {
        return prev_;
    }

    auto& set_prev(const partial_train_result<Task>& value) {
        prev_ = value;
        return *this;
    }

private:
    partial_train_result<Task> prev_;
};

} // namespace v1

using v1::train_input;
using v1::train_result;
using v1::partial_train_input;
using v1::partial_train_result;

} // namespace oneapi::dal::linear_regression
