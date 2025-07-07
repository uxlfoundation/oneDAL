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

#include "oneapi/dal/algo/correlation_distance/common.hpp"

namespace oneapi::dal::correlation_distance {

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

    /// Creates a new instance of the class with the given :literal:`x` and :literal:`y`.
    compute_input(const table& x, const table& y);

    /// An $n \\times p$ table with the data x, where each row
    /// stores one feature vector.
    /// @remark default = table{}
    const table& get_x() const;

    auto& set_x(const table& data) {
        set_x_impl(data);
        return *this;
    }

    /// An $m \\times p$ table with the data y, where each row
    /// stores one feature vector.
    /// @remark default = table{}
    const table& get_y() const;

    auto& set_y(const table& data) {
        set_y_impl(data);
        return *this;
    }

protected:
    void set_x_impl(const table& data);
    void set_y_impl(const table& data);

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

    /// A $n \\times m$ table with the result kernel functions.
    /// @remark default = table{}
    const table& get_values() const;

    auto& set_values(const table& value) {
        set_values_impl(value);
        return *this;
    }

protected:
    void set_values_impl(const table&);

private:
    dal::detail::pimpl<detail::compute_result_impl<Task>> impl_;
};

} // namespace v1

using v1::compute_input;
using v1::compute_result;

} // namespace oneapi::dal::correlation_distance
