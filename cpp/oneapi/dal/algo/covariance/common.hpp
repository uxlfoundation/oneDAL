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

#include "oneapi/dal/util/result_option_id.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/serialization.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/common.hpp"

namespace oneapi::dal::covariance {

namespace task {
namespace v1 {

/// Tag-type that parameterizes entities that are used to compute statistics.
struct compute {};
/// Alias tag-type for the compute task.
using by_default = compute;
} // namespace v1

using v1::compute;
using v1::by_default;

} // namespace task

namespace method {
namespace v1 {
/// Tag-type that denotes dense computational method.
struct dense {};

/// Alias tag-type for the dense computational method.
using by_default = dense;

} // namespace v1

using v1::dense;
using v1::by_default;

} // namespace method

/// Represents result option flag
/// Behaves like a regular :expr`enum`.
class result_option_id : public result_option_id_base {
public:
    constexpr result_option_id() = default;
    constexpr explicit result_option_id(const result_option_id_base& base)
            : result_option_id_base{ base } {}
};

namespace detail {

ONEDAL_EXPORT result_option_id get_cov_matrix_id();
ONEDAL_EXPORT result_option_id get_cor_matrix_id();
ONEDAL_EXPORT result_option_id get_means_id();

} // namespace detail

/// Result options are used to define
/// what should an algorithm return
namespace result_options {

/// Return covariance matrix
const inline result_option_id cov_matrix = detail::get_cov_matrix_id();
/// Return correlation matrix
const inline result_option_id cor_matrix = detail::get_cor_matrix_id();
/// Return means
const inline result_option_id means = detail::get_means_id();

} // namespace result_options

namespace detail {
namespace v1 {
struct descriptor_tag {};

template <typename Task>
class descriptor_impl;

template <typename Float>
constexpr bool is_valid_float_v = dal::detail::is_one_of_v<Float, float, double>;

template <typename Method>
constexpr bool is_valid_method_v = dal::detail::is_one_of_v<Method, method::dense>;

template <typename Task>
constexpr bool is_valid_task_v = dal::detail::is_one_of_v<Task, task::compute>;

template <typename Task = task::by_default>
class descriptor_base : public base {
    static_assert(is_valid_task_v<Task>);

public:
    using tag_t = descriptor_tag;
    using float_t = float;
    using method_t = method::by_default;
    using task_t = Task;

    descriptor_base();

    bool get_assume_centered() const;
    bool get_bias() const;
    result_option_id get_result_options() const;

protected:
    void set_assume_centered_impl(const bool& value);
    void set_bias_impl(const bool& value);
    void set_result_options_impl(const result_option_id& value);

private:
    dal::detail::pimpl<descriptor_impl<Task>> impl_;
};

} // namespace v1

using v1::descriptor_tag;
using v1::descriptor_impl;
using v1::descriptor_base;

using v1::is_valid_float_v;
using v1::is_valid_method_v;
using v1::is_valid_task_v;

} // namespace detail

namespace v1 {

/// @tparam Float  The floating-point type that the algorithm uses for
///                intermediate computations. Can be :expr:`float` or
///                :expr:`double`.
/// @tparam Method Tag-type that specifies an implementation of algorithm. Can
///                be :expr:`method::dense`.
/// @tparam Task   Tag-type that specifies the type of the problem to solve. Can
///                be :expr:`task::compute`.
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

    /// Indicates whether the input data is assumed to be centered.
    /// @remark default = false
    bool get_assume_centered() const {
        return base_t::get_assume_centered();
    }

    auto& set_assume_centered(const bool& value) {
        base_t::set_assume_centered_impl(value);
        return *this;
    }

    /// Choose if result biased or not
    bool get_bias() const {
        return base_t::get_bias();
    }

    auto& set_bias(const bool& value) {
        base_t::set_bias_impl(value);
        return *this;
    }

    /// Choose which results should be computed and returned.
    result_option_id get_result_options() const {
        return base_t::get_result_options();
    }

    auto& set_result_options(const result_option_id& value) {
        base_t::set_result_options_impl(value);
        return *this;
    }
};

} // namespace v1

using v1::descriptor;

} // namespace oneapi::dal::covariance
