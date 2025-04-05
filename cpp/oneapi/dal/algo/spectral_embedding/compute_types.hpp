/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#include "oneapi/dal/algo/spectral_embedding/common.hpp"

namespace oneapi::dal::spectral_embedding {

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

/// @tparam Task Tag-type that specifies the type of the problem to solve. Can
///              be :expr:`task::compute`.
template <typename Task = task::by_default>
class compute_input : public base {
    static_assert(detail::is_valid_task_v<Task>);

public:
    using task_t = Task;

    /// Creates a new instance of the class with the given :literal:`data`
    compute_input(const table& data);

    /// An $n \\times p$ table with the training data, where each row stores one
    /// feature vector.
    /// @remark default = table{}
    const table& get_data() const;

    auto& set_data(const table& value) {
        set_data_impl(value);
        return *this;
    }

protected:
    void set_data_impl(const table& value);

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

    /// The matrix of size $n \\times k$ with
    /// spectral embeddings.
    /// @remark default = table{}
    const table& get_embedding() const;

    auto& set_embedding(const table& value) {
        set_embedding_impl(value);
        return *this;
    }

    /// The matrix of size $n \\times 1$ with
    /// eigen values of Laplassian matrix.
    /// @remark default = table{}
    const table& get_eigen_values() const;

    auto& set_eigen_values(const table& value) {
        set_eigen_values_impl(value);
        return *this;
    }

    /// Result options that indicates availability of the properties
    /// @remark default = default_result_options<Task>
    const result_option_id& get_result_options() const;

    auto& set_result_options(const result_option_id& value) {
        set_result_options_impl(value);
        return *this;
    }

protected:
    void set_embedding_impl(const table&);
    void set_eigen_values_impl(const table&);
    void set_result_options_impl(const result_option_id&);

private:
    dal::detail::pimpl<detail::compute_result_impl<Task>> impl_;
};

} // namespace v1

using v1::compute_input;
using v1::compute_result;

} // namespace oneapi::dal::spectral_embedding
