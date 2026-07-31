/*******************************************************************************
* Copyright 2023 Intel Corporation
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
#include "oneapi/dal/detail/serialization.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/common.hpp"
#include "oneapi/dal/algo/logistic_regression/detail/optimizer.hpp"

namespace oneapi::dal::logistic_regression {

namespace task {
namespace v1 {
/// Tag-type that parameterizes entities used for solving
/// :capterm:`classification problem <classification>`.
struct classification {};

/// Alias tag-type for classification task
using by_default = classification;
} // namespace v1

using v1::classification;
using v1::by_default;

} // namespace task

namespace method {
namespace v1 {
/// Tag-type that denotes :ref:`dense_batch <logreg_t_math_dense_batch>` computational method.
struct dense_batch {};
/// Tag-type that denotes :ref:`sparse <logreg_t_math_sparse>` computational method.
struct sparse {};

/// Alias tag-type for the dense_batch method
using by_default = dense_batch;
} // namespace v1

using v1::dense_batch;
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

ONEDAL_EXPORT result_option_id get_intercept_id();
ONEDAL_EXPORT result_option_id get_coefficients_id();
ONEDAL_EXPORT result_option_id get_iterations_count_id();
ONEDAL_EXPORT result_option_id get_inner_iterations_count_id();

} // namespace detail

/// Result options are used to define
/// what should algorithm return
namespace result_options {

/// Return the indices the intercept term in logistic regression
const inline result_option_id intercept = detail::get_intercept_id();

/// Return the coefficients to use in logistic regression
const inline result_option_id coefficients = detail::get_coefficients_id();

/// Return the number of iterations made by optimizer
const inline result_option_id iterations_count = detail::get_iterations_count_id();

/// Return the number of subiterations made by optimizer. Only available for newton-cg optimizer
const inline result_option_id inner_iterations_count = detail::get_inner_iterations_count_id();

} // namespace result_options

namespace detail {
namespace v1 {

struct descriptor_tag {};

template <typename Task>
class descriptor_impl;

template <typename Task>
class model_impl;

template <typename Float>
constexpr bool is_valid_float_v = dal::detail::is_one_of_v<Float, float, double>;

template <typename Method>
constexpr bool is_valid_method_v =
    dal::detail::is_one_of_v<Method, method::dense_batch, method::sparse>;

template <typename Task>
constexpr bool is_valid_task_v = dal::detail::is_one_of_v<Task, task::classification>;

template <typename Optimizer>
constexpr bool is_valid_optimizer_v =
    dal::detail::is_tag_one_of_v<Optimizer, newton_cg::detail::descriptor_tag>;

template <typename Task = task::by_default>
class descriptor_base : public base {
    static_assert(is_valid_task_v<Task>);
    friend detail::optimizer_accessor;

public:
    using tag_t = descriptor_tag;
    using float_t = float;
    using optimizer_t = oneapi::dal::newton_cg::descriptor<float_t>;
    descriptor_base();

    bool get_compute_intercept() const;
    double get_inverse_regularization() const;
    std::int64_t get_class_count() const;
    result_option_id get_result_options() const;

protected:
    explicit descriptor_base(bool compute_intercept,
                             double C,
                             const detail::optimizer_ptr& optimizer);

    void set_compute_intercept_impl(bool compute_intercept);
    void set_inverse_regularization_impl(double C);
    void set_class_count_impl(std::int64_t class_count);

    void set_optimizer_impl(const detail::optimizer_ptr& opt);
    void set_result_options_impl(const result_option_id& value);

    const detail::optimizer_ptr& get_optimizer_impl() const;

private:
    dal::detail::pimpl<descriptor_impl<Task>> impl_;
};

} // namespace v1

using v1::descriptor_tag;
using v1::descriptor_impl;
using v1::model_impl;
using v1::descriptor_base;

using v1::is_valid_float_v;
using v1::is_valid_method_v;
using v1::is_valid_task_v;
using v1::is_valid_optimizer_v;

} // namespace detail

namespace v1 {

/// @tparam Float       The floating-point type that the algorithm uses for
///                     intermediate computations. Can be :expr:`float` or
///                     :expr:`double`.
/// @tparam Method      Tag-type that specifies an implementation of algorithm. Can
///                     be :expr:`method::dense_batch` or :expr:`method::sparse`.
/// @tparam Task        Tag-type that specifies type of the problem to solve. Can
///                     be :expr:`task::classification`.
/// @tparam Optimizer   The descriptor of the optimizer used for minimization. Can
///                     be :expr:`newton_cg::descriptor`
template <typename Float = float,
          typename Method = method::by_default,
          typename Task = task::by_default,
          typename Optimizer = oneapi::dal::newton_cg::descriptor<Float>>
class descriptor : public detail::descriptor_base<Task> {
    static_assert(detail::is_valid_float_v<Float>);
    static_assert(detail::is_valid_method_v<Method>);
    static_assert(detail::is_valid_task_v<Task>);
    static_assert(detail::is_valid_optimizer_v<Optimizer>);

    using base_t = detail::descriptor_base<Task>;

public:
    using float_t = Float;
    using method_t = Method;
    using task_t = Task;
    using optimizer_t = Optimizer;

    /// Creates a new instance of the class with the given :literal:`compute_intercept`
    /// and :literal:`C`
    explicit descriptor(bool compute_intercept = true, double C = 1.0)
            : base_t(compute_intercept,
                     C,
                     std::make_shared<detail::optimizer<optimizer_t>>(optimizer_t{})) {}

    /// Creates a new instance of the class with the given :literal:`compute_intercept`,
    /// :literal:`C` and :literal:`optimizer`
    explicit descriptor(bool compute_intercept, double C, const optimizer_t& optimizer)
            : base_t(compute_intercept,
                     C,
                     std::make_shared<detail::optimizer<optimizer_t>>(optimizer)) {}

    /// Defines should intercept be taken into consideration.
    bool get_compute_intercept() const {
        return base_t::get_compute_intercept();
    }

    /// Defines inverse regularization factor.
    double get_inverse_regularization() const {
        return base_t::get_inverse_regularization();
    }

    /// Defines number of classes.
    std::int64_t get_class_count() const {
        return base_t::get_class_count();
    }

    auto& set_compute_intercept(bool compute_intercept) const {
        base_t::set_compute_intercept_impl(compute_intercept);
        return *this;
    }

    auto& set_inverse_regularization(double C) const {
        base_t::set_inverse_regularization_impl(C);
        return *this;
    }

    auto& set_class_count(std::int64_t class_count) const {
        base_t::set_class_count_impl(class_count);
        return *this;
    }

    /// The optimizer descriptor.
    /// @return The descriptor of the optimizer used for minimization.
    const optimizer_t& get_optimizer() const {
        using optimizer_t = detail::optimizer<optimizer_t>;
        const auto opt = std::static_pointer_cast<optimizer_t>(base_t::get_optimizer_impl());
        return opt;
    }

    auto& set_optimizer(const optimizer_t& opt) {
        base_t::set_optimizer_impl(std::make_shared<detail::optimizer<optimizer_t>>(opt));
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

/// @tparam Task Tag-type that specifies type of the problem to solve.
template <typename Task = task::by_default>
class model : public base {
    static_assert(detail::is_valid_task_v<Task>);
    friend dal::detail::pimpl_accessor;
    friend dal::detail::serialization_accessor;

public:
    /// Creates a new instance of the class with the default property values.
    model();

    /// The packed coefficients table.
    const table& get_packed_coefficients() const;
    model& set_packed_coefficients(const table& t);

private:
    void serialize(dal::detail::output_archive& ar) const;
    void deserialize(dal::detail::input_archive& ar);

    explicit model(const std::shared_ptr<detail::model_impl<Task>>& impl);
    dal::detail::pimpl<detail::model_impl<Task>> impl_;
};

} // namespace v1

using v1::descriptor;
using v1::model;

} // namespace oneapi::dal::logistic_regression
