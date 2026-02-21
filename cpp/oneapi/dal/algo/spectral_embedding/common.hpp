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

#include "oneapi/dal/util/result_option_id.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/serialization.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/common.hpp"

namespace oneapi::dal::spectral_embedding {

namespace task {
namespace v1 {

/// Tag-type that parameterizes entities that are used to compute statistics.
struct compute {};

/// Alias tag-type for compute task.
using by_default = compute;
} // namespace v1

using v1::compute;
using v1::by_default;

} // namespace task

namespace method {
namespace v1 {

/// Tag-type that denotes dense_batch computational method.
struct dense_batch {};

/// Alias tag-type for the dense_batch computational method.
using by_default = dense_batch;

} // namespace v1

using v1::dense_batch;
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

ONEDAL_EXPORT result_option_id get_embedding_id();
ONEDAL_EXPORT result_option_id get_eigen_values_id();

} // namespace detail

/// Result options are used to define
/// what should algorithm return
namespace result_options {

/// Return spectral embedding
const inline auto embedding = detail::get_embedding_id();

/// Return eigen values of Laplassian matrix
const inline auto eigen_values = detail::get_eigen_values_id();

} // namespace result_options

namespace detail {

namespace v1 {

struct descriptor_tag {};

template <typename Task>
class descriptor_impl;

template <typename Float>
constexpr bool is_valid_float_v = dal::detail::is_one_of_v<Float, float, double>;

template <typename Method>
constexpr bool is_valid_method_v = dal::detail::is_one_of_v<Method, method::dense_batch>;

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

    std::int64_t get_component_count() const;
    std::int64_t get_neighbor_count() const;
    result_option_id get_result_options() const;

protected:
    void set_component_count_impl(std::int64_t component_count);
    void set_neighbor_count_impl(std::int64_t neighbor_count);
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

/// @tparam Float       The floating-point type that the algorithm uses for
///                     intermediate computations. Can be :expr:`float` or
///                     :expr:`double`.
/// @tparam Method      Tag-type that specifies an implementation of algorithm. Can
///                     be :expr:`method::dense_batch`.
/// @tparam Task        Tag-type that specifies the type of the problem to solve. Can
///                     be :expr:`task::compute`.

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
    explicit descriptor() : base_t() {}

    std::int64_t get_component_count() const {
        return base_t::get_component_count();
    }

    std::int64_t get_neighbor_count() const {
        return base_t::get_neighbor_count();
    }

    auto& set_component_count(std::int64_t component_count) {
        base_t::set_component_count_impl(component_count);
        return *this;
    }

    auto& set_neighbor_count(std::int64_t neighbor_count) {
        base_t::set_neighbor_count_impl(neighbor_count);
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

} // namespace oneapi::dal::spectral_embedding
