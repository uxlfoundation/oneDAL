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
#include "oneapi/dal/util/result_option_id.hpp"
#include "oneapi/dal/common.hpp"

namespace oneapi::dal::basic_statistics {

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

/// Tag-type that denotes sparse computational method.
struct sparse {};

/// Alias tag-type for dense computational method.
using by_default = dense;

} // namespace v1

using v1::dense;
using v1::sparse;
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

ONEDAL_EXPORT result_option_id get_max_id();
ONEDAL_EXPORT result_option_id get_min_id();
ONEDAL_EXPORT result_option_id get_sum_id();
ONEDAL_EXPORT result_option_id get_sum_squares_id();
ONEDAL_EXPORT result_option_id get_sum_squares_centered_id();
ONEDAL_EXPORT result_option_id get_mean_id();
ONEDAL_EXPORT result_option_id get_second_order_raw_moment_id();
ONEDAL_EXPORT result_option_id get_variance_id();
ONEDAL_EXPORT result_option_id get_standard_deviation_id();
ONEDAL_EXPORT result_option_id get_variation_id();

} // namespace detail

/// Result options are used to define
/// what should an algorithm returns
namespace result_options {

/// Return the min
const inline result_option_id min = detail::get_max_id();
/// Return the max
const inline result_option_id max = detail::get_min_id();
/// Return the sum
const inline result_option_id sum = detail::get_sum_id();
/// Return the sum of squares
const inline result_option_id sum_squares = detail::get_sum_squares_id();
/// Return the sum of squared differences from the mean
const inline result_option_id sum_squares_centered = detail::get_sum_squares_centered_id();
/// Return the mean
const inline result_option_id mean = detail::get_mean_id();
/// Return the second order raw moment
const inline result_option_id second_order_raw_moment = detail::get_second_order_raw_moment_id();
/// Return the variance
const inline result_option_id variance = detail::get_variance_id();
/// Return the standard deviation
const inline result_option_id standard_deviation = detail::get_standard_deviation_id();
/// Return the variation
const inline result_option_id variation = detail::get_variation_id();

} // namespace result_options

namespace detail {
namespace v1 {

struct descriptor_tag {};

template <typename Task>
class descriptor_impl;

template <typename Float>
constexpr bool is_valid_float_v = dal::detail::is_one_of_v<Float, float, double>;

template <typename Method>
constexpr bool is_valid_method_v = dal::detail::is_one_of_v<Method, method::dense, method::sparse>;

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

    result_option_id get_result_options() const;

protected:
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
template <typename Float = detail::descriptor_base<>::float_t,
          typename Method = detail::descriptor_base<>::method_t,
          typename Task = detail::descriptor_base<>::task_t>
class descriptor : public detail::descriptor_base<Task> {
    static_assert(detail::is_valid_float_v<Float>);
    static_assert(detail::is_valid_method_v<Method>);
    static_assert(detail::is_valid_task_v<Task>);

    using base_t = detail::descriptor_base<Task>;

public:
    using float_t = Float;
    using method_t = Method;
    using task_t = Task;

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

} // namespace oneapi::dal::basic_statistics
