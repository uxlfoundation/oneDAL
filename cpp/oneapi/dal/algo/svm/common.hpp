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

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/detail/serialization.hpp"
#include "oneapi/dal/algo/svm/detail/kernel_function.hpp"

namespace oneapi::dal::svm {

namespace task {
namespace v1 {

/// Tag-type that parameterizes entities that are used for solving
/// :capterm:`classification problem <classification>`.
struct classification {};

/// Tag-type that parameterizes entities used for solving
/// :capterm:`regression problem <regression>`.
struct regression {};

/// Tag-type that parameterizes entities that are used for solving
/// :capterm:`nu-classification problem <nu-classification>`.
struct nu_classification {};

/// Tag-type that parameterizes entities used for solving
/// :capterm:`nu-regression problem <nu-regression>`.
struct nu_regression {};

/// Alias tag-type for classification task.
using by_default = classification;
} // namespace v1

using v1::classification;
using v1::regression;
using v1::nu_classification;
using v1::nu_regression;
using v1::by_default;

} // namespace task

namespace method {
namespace v1 {

/// Tag-type that denotes :ref:`Thunder <svm_t_math_thunder>` computational
/// method.
struct thunder {};

/// Tag-type that denotes :ref:`SMO <svm_t_math_smo>` computational
/// method.
struct smo {};

/// Alias tag-type for :ref:`Thunder <svm_t_math_thunder>` computational
/// method.
using by_default = thunder;
} // namespace v1

using v1::thunder;
using v1::smo;
using v1::by_default;

} // namespace method

namespace detail {
namespace v1 {
struct descriptor_tag {};

template <typename Task>
class descriptor_impl;

template <typename Task>
class model_impl;

template <typename T>
using enable_if_classification_t =
    std::enable_if_t<dal::detail::is_one_of_v<T, task::classification, task::nu_classification>>;

template <typename T>
using enable_if_regression_t =
    std::enable_if_t<dal::detail::is_one_of_v<T, task::regression, task::nu_regression>>;

template <typename T>
using enable_if_nu_task_t =
    std::enable_if_t<dal::detail::is_one_of_v<T, task::nu_classification, task::nu_regression>>;

template <typename T>
using enable_if_c_available_t = std::enable_if_t<
    dal::detail::is_one_of_v<T, task::classification, task::regression, task::nu_regression>>;

template <typename T>
using enable_if_epsilon_available_t =
    std::enable_if_t<std::is_same_v<std::decay_t<T>, task::regression>>;

template <typename Float>
constexpr bool is_valid_float_v = dal::detail::is_one_of_v<Float, float, double>;

template <typename Method>
constexpr bool is_valid_method_v = dal::detail::is_one_of_v<Method, method::smo, method::thunder>;

template <typename Task>
constexpr bool is_valid_task_v = dal::detail::is_one_of_v<Task,
                                                          task::classification,
                                                          task::regression,
                                                          task::nu_classification,
                                                          task::nu_regression>;

template <typename Method, typename Task>
constexpr bool is_valid_method_task_combination = dal::detail::is_one_of_v<Method, method::smo>
    &&dal::detail::is_one_of_v<Task, task::regression>;

template <typename Method, typename Task>
constexpr bool is_valid_method_nu_task_combination = dal::detail::is_one_of_v<Method, method::smo>
    &&dal::detail::is_one_of_v<Task, task::nu_classification, task::nu_regression>;

template <typename Kernel>
constexpr bool is_valid_kernel_v =
    dal::detail::is_tag_one_of_v<Kernel,
                                 linear_kernel::detail::descriptor_tag,
                                 polynomial_kernel::detail::descriptor_tag,
                                 rbf_kernel::detail::descriptor_tag,
                                 sigmoid_kernel::detail::descriptor_tag>;

template <typename Task = task::by_default>
class descriptor_base : public base {
    static_assert(is_valid_task_v<Task>);
    friend detail::kernel_function_accessor;

public:
    using tag_t = descriptor_tag;
    using float_t = float;
    using method_t = method::by_default;
    using task_t = Task;
    using kernel_t = linear_kernel::descriptor<float_t>;

    double get_c() const;
    std::int64_t get_max_iteration_count() const;
    double get_accuracy_threshold() const;
    double get_cache_size() const;
    double get_tau() const;
    bool get_shrinking() const;

    std::int64_t get_class_count() const {
        return get_class_count_impl();
    }

    double get_epsilon() const {
        return get_epsilon_impl();
    }

    double get_nu() const {
        return get_nu_impl();
    }

protected:
    explicit descriptor_base(const detail::kernel_function_ptr &kernel);

    void set_c_impl(double);
    void set_accuracy_threshold_impl(double);
    void set_max_iteration_count_impl(std::int64_t);
    void set_cache_size_impl(double);
    void set_tau_impl(double);
    void set_shrinking_impl(bool);
    void set_kernel_impl(const detail::kernel_function_ptr &);
    void set_class_count_impl(std::int64_t);
    void set_epsilon_impl(double);
    void set_nu_impl(double);

    std::int64_t get_class_count_impl() const;
    double get_epsilon_impl() const;
    double get_nu_impl() const;
    const detail::kernel_function_ptr &get_kernel_impl() const;

private:
    dal::detail::pimpl<descriptor_impl<Task>> impl_;
};

} // namespace v1

using v1::descriptor_tag;
using v1::descriptor_impl;
using v1::model_impl;
using v1::descriptor_base;

using v1::enable_if_classification_t;
using v1::enable_if_regression_t;
using v1::enable_if_c_available_t;
using v1::enable_if_epsilon_available_t;
using v1::enable_if_nu_task_t;
using v1::is_valid_float_v;
using v1::is_valid_method_v;
using v1::is_valid_task_v;
using v1::is_valid_method_task_combination;
using v1::is_valid_method_nu_task_combination;
using v1::is_valid_kernel_v;

} // namespace detail

namespace v1 {

/// @tparam Float  The floating-point type that the algorithm uses for
///                intermediate computations. Can be :expr:`float` or
///                :expr:`double`.
/// @tparam Method Tag-type that specifies an implementation of algorithm. Can
///                be :expr:`method::thunder` or :expr:`method::smo`.
/// @tparam Task   Tag-type that specifies the type of the problem to solve.
///                Can be :expr:`task::classification`,
///                :expr:`task::nu_classification`,
///                :expr:`task::regression`, or :expr:`task::nu_regression`.
template <typename Float = float,
          typename Method = method::by_default,
          typename Task = task::by_default,
          typename Kernel = linear_kernel::descriptor<Float>>
class descriptor : public detail::descriptor_base<Task> {
    static_assert(detail::is_valid_float_v<Float>);
    static_assert(detail::is_valid_method_v<Method>);
    static_assert(detail::is_valid_task_v<Task>);
    static_assert(!detail::is_valid_method_task_combination<Method, Task>,
                  "Regression SVM not supported with SMO method");
    static_assert(!detail::is_valid_method_nu_task_combination<Method, Task>,
                  "nuSVM is not supported with SMO method");
    static_assert(detail::is_valid_kernel_v<Kernel>,
                  "Custom kernel for SVM is not supported. "
                  "Use one of the predefined kernels.");

    using base_t = detail::descriptor_base<Task>;

public:
    using float_t = Float;
    using method_t = Method;
    using task_t = Task;
    using kernel_t = Kernel;

    /// Creates a new instance of the class with the given descriptor of the
    /// kernel function
    /// @remark default = :literal:`kernel`
    explicit descriptor(const Kernel &kernel = kernel_t{})
            : base_t(std::make_shared<detail::kernel_function<Kernel>>(kernel)) {}

    /// The descriptor of kernel function $K(x, y)$. Can be
    /// :expr:`linear_kernel::descriptor` or
    /// :expr:`polynomial_kernel::descriptor` or
    /// :expr:`rbf_kernel::descriptor` or
    /// :expr:`sigmoid_kernel::descriptor`.
    /// @remark default = :literal:`kernel`
    const Kernel &get_kernel() const {
        using kf_t = detail::kernel_function<Kernel>;
        const auto kf = std::static_pointer_cast<kf_t>(base_t::get_kernel_impl());
        return kf->get_kernel();
    }

    auto &set_kernel(const Kernel &kernel) {
        base_t::set_kernel_impl(std::make_shared<detail::kernel_function<Kernel>>(kernel));
        return *this;
    }

    template <typename T = Task, typename = detail::enable_if_c_available_t<T>>
    /// The upper bound $C$ in constraints of the quadratic optimization
    /// problem.
    /// Used with :expr:`task::classification`, :expr:`task::regression`, and
    /// :expr:`task::nu_regression`.
    /// @invariant :expr:`c > 0`
    /// @remark default = 1.0
    double get_c() const {
        return base_t::get_c();
    }

    template <typename T = Task, typename = detail::enable_if_c_available_t<T>>
    auto &set_c(double value) {
        base_t::set_c_impl(value);
        return *this;
    }

    /// The maximum number of iterations $T$
    /// @invariant :expr:`max_iteration_count >= 0`
    /// @remark default = 100000
    std::int64_t get_max_iteration_count() const {
        return base_t::get_max_iteration_count();
    }

    auto &set_max_iteration_count(std::int64_t value) {
        base_t::set_max_iteration_count_impl(value);
        return *this;
    }

    /// The threshold $\\varepsilon$ for the stop condition
    /// @invariant :expr:`accuracy_threshold >= 0.0`
    /// @remark default = 0.0
    double get_accuracy_threshold() const {
        return base_t::get_accuracy_threshold();
    }

    auto &set_accuracy_threshold(double value) {
        base_t::set_accuracy_threshold_impl(value);
        return *this;
    }

    /// The size of cache (in megabytes) for storing the values of the kernel
    /// matrix.
    /// @invariant :expr:`cache_size >= 0.0`
    /// @remark default = 200.0
    double get_cache_size() const {
        return base_t::get_cache_size();
    }

    auto &set_cache_size(double value) {
        base_t::set_cache_size_impl(value);
        return *this;
    }

    /// The threshold parameter $\\tau$ for computing the quadratic coefficient.
    /// @invariant :expr:`tau > 0.0`
    /// @remark default = 1e-6
    double get_tau() const {
        return base_t::get_tau();
    }

    auto &set_tau(double value) {
        base_t::set_tau_impl(value);
        return *this;
    }

    /// A flag that enables the use of a shrinking optimization technique.
    /// Used with :expr:`method::smo` split-finding method only.
    /// @remark default = true
    bool get_shrinking() const {
        return base_t::get_shrinking();
    }

    auto &set_shrinking(bool value) {
        base_t::set_shrinking_impl(value);
        return *this;
    }

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    /// The number of classes. Used with :expr:`task::classification`
    /// and :expr:`task::nu_classification`.
    /// @invariant :expr:`class_count >= 2`
    /// @remark default = 2
    std::int64_t get_class_count() const {
        return base_t::get_class_count_impl();
    }

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    auto &set_class_count(std::int64_t value) {
        base_t::set_class_count_impl(value);
        return *this;
    }

    template <typename T = Task, typename = detail::enable_if_epsilon_available_t<T>>
    /// The epsilon. Used with :expr:`task::regression` only.
    /// @invariant :expr:`epsilon >= 0`
    /// @remark default = 0.1
    double get_epsilon() const {
        return base_t::get_epsilon_impl();
    }

    template <typename T = Task, typename = detail::enable_if_epsilon_available_t<T>>
    auto &set_epsilon(double value) {
        base_t::set_epsilon_impl(value);
        return *this;
    }

    template <typename T = Task, typename = detail::enable_if_nu_task_t<T>>
    /// The nu. Used with :expr:`task::nu_classification` and
    /// :expr:`task::nu_regression`.
    /// @invariant :expr:`0 < nu <= 1`
    /// @remark default = 0.5
    double get_nu() const {
        return base_t::get_nu_impl();
    }

    template <typename T = Task, typename = detail::enable_if_nu_task_t<T>>
    auto &set_nu(double value) {
        base_t::set_nu_impl(value);
        return *this;
    }
};

/// @tparam Task Tag-type that specifies the type of the problem to solve. Can
///              be :expr:`task::classification`,
///              :expr:`task::nu_classification`,
///              :expr:`task::regression`, or :expr:`task::nu_regression`.
template <typename Task = task::by_default>
class model : public base {
    static_assert(detail::is_valid_task_v<Task>);
    friend dal::detail::pimpl_accessor;
    friend dal::detail::serialization_accessor;

public:
    using task_t = Task;

    /// Creates a new instance of the class with the default property values.
    model();

    /// The number of support vectors
    /// @remark default = 0
    /// @invariant :expr:`support_vector_count >= 0`
    std::int64_t get_support_vector_count() const;

    /// A $nsv \\times p$ table containing support vectors.
    /// Where $nsv$ - number of support vectors.
    /// @remark default = table{}
    const table &get_support_vectors() const;

    auto &set_support_vectors(const table &value) {
        set_support_vectors_impl(value);
        return *this;
    }

    /// A $nsv \\times class_count - 1$ table for :expr:`task::classification`
    /// and :expr:`task::nu_classification`
    /// and a $nsv \\times 1$ table for :expr:`task::regression`
    /// and :expr:`task::nu_regression`
    /// containing coefficients of Lagrange multiplier
    /// @remark default = table{}
    const table &get_coeffs() const;

    auto &set_coeffs(const table &value) {
        set_coeffs_impl(value);
        return *this;
    }

    /// The bias
    /// @remark default = 0.0
    [[deprecated("Use get_biases() instead.")]] double get_bias() const;

    [[deprecated("Use set_biases() instead.")]] auto &set_bias(double value) {
        set_bias_impl(value);
        return *this;
    }

    /// A $class_count*(class_count-1)/2 \\times 1$ table for
    /// :expr:`task::classification` and :expr:`task::nu_classification`
    /// and a $1 \\times 1$ table for :expr:`task::regression` and
    /// :expr:`task::nu_regression` containing constants in decision function
    const table &get_biases() const;

    auto &set_biases(const table &value) {
        set_biases_impl(value);
        return *this;
    }

    /// The number of iterations performed by the SVM training algorithm
    const table &get_iteration_counts() const;

    auto &set_iteration_counts(const table &value) {
        set_iteration_counts_impl(value);
        return *this;
    }

    /// The first unique value in class labels.
    /// Used with :expr:`task::classification` and
    /// :expr:`task::nu_classification`.
    [[deprecated]] std::int64_t get_first_class_label() const {
        return get_first_class_response();
    }

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    [[deprecated]] auto &set_first_class_label(std::int64_t value) {
        return set_first_class_response(value);
    }

    /// The first unique value in class responses.
    /// Used with :expr:`task::classification` and
    /// :expr:`task::nu_classification`.
    std::int64_t get_first_class_response() const;

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    auto &set_first_class_response(std::int64_t value) {
        set_first_class_response_impl(value);
        return *this;
    }

    /// The second unique value in class labels.
    /// Used with :expr:`task::classification` and
    /// :expr:`task::nu_classification`.
    [[deprecated]] std::int64_t get_second_class_label() const {
        return get_second_class_response();
    }

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    [[deprecated]] auto &set_second_class_label(std::int64_t value) {
        return set_second_class_response(value);
    }

    /// The second unique value in class responses.
    /// Used with :expr:`task::classification` and
    /// :expr:`task::nu_classification`.
    std::int64_t get_second_class_response() const;

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    auto &set_second_class_response(std::int64_t value) {
        set_second_class_response_impl(value);
        return *this;
    }

protected:
    void set_support_vectors_impl(const table &);
    void set_coeffs_impl(const table &);
    void set_bias_impl(double);
    void set_biases_impl(const table &);
    void set_iteration_counts_impl(const table &);
    void set_first_class_response_impl(std::int64_t);
    void set_second_class_response_impl(std::int64_t);

private:
    void serialize(dal::detail::output_archive &ar) const;
    void deserialize(dal::detail::input_archive &ar);

    explicit model(const std::shared_ptr<detail::model_impl<Task>> &impl);
    dal::detail::pimpl<detail::model_impl<Task>> impl_;
};

} // namespace v1

using v1::descriptor;
using v1::model;

} // namespace oneapi::dal::svm
