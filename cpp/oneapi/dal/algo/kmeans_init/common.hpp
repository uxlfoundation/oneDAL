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

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/rng.hpp"

namespace oneapi::dal::kmeans_init {

namespace task {
namespace v1 {
/// Tag-type that parameterizes entities used for obtaining the initial K-Means centroids.
struct init {};

/// Alias tag-type for the initialization task.
using by_default = init;
} // namespace v1

using v1::init;
using v1::by_default;

} // namespace task

namespace method {
namespace v1 {
/// Tag-type that denotes :ref:`dense <kmeans_init_c_math_dense>`
/// computational method.
struct dense {};
/// Tag-type that denotes :ref:`random_dense <kmeans_init_c_math_random_dense>`
/// computational method.
struct random_dense {};

struct random_csr {};
/// Tag-type that denotes :ref:`plus_plus_dense <kmeans_init_c_math_plus_plus_dense>`
/// computational method.
struct plus_plus_dense {};

struct plus_plus_csr {};
/// Tag-type that denotes :ref:`parallel_plus_dense <kmeans_init_c_math_parallel_plus_dense>`
/// computational method.
struct parallel_plus_dense {};

struct parallel_plus_csr {};
using by_default = dense;
} // namespace v1

using v1::dense;
using v1::random_dense;
using v1::random_csr;
using v1::plus_plus_dense;
using v1::plus_plus_csr;
using v1::parallel_plus_dense;
using v1::parallel_plus_csr;
using v1::by_default;

} // namespace method

namespace detail {
namespace v1 {

struct descriptor_tag {};
template <typename Task>
class descriptor_impl;

template <typename Float>
constexpr bool is_valid_float_v = dal::detail::is_one_of_v<Float, float, double>;

template <typename Method>
constexpr bool is_valid_method_v = dal::detail::is_one_of_v<Method,
                                                            method::dense,
                                                            method::random_dense,
                                                            method::random_csr,
                                                            method::plus_plus_dense,
                                                            method::plus_plus_csr,
                                                            method::parallel_plus_dense,
                                                            method::parallel_plus_csr>;

template <typename Method>
constexpr bool is_plus_plus_dense_or_csr_v =
    dal::detail::is_one_of_v<Method, method::plus_plus_dense, method::plus_plus_csr>;

template <typename Task>
constexpr bool is_valid_task_v = dal::detail::is_one_of_v<Task, task::init>;

template <typename M>
constexpr bool is_not_default_dense = !std::is_same_v<M, method::dense>;

template <typename M>
using enable_if_not_default_dense = std::enable_if_t<is_not_default_dense<M>>;

template <typename M>
using enable_if_plus_plus = std::enable_if_t<is_plus_plus_dense_or_csr_v<M>>;

template <typename Task = task::by_default>
class descriptor_base : public base {
    static_assert(is_valid_task_v<Task>);

public:
    using tag_t = descriptor_tag;
    using float_t = float;
    using method_t = method::by_default;
    using task_t = Task;

    descriptor_base();

    std::int64_t get_local_trials_count() const;
    std::int64_t get_cluster_count() const;
    engine_type get_engine_type() const;
    std::int64_t get_seed() const;

protected:
    void set_local_trials_count_impl(std::int64_t);
    void set_cluster_count_impl(std::int64_t);
    void set_seed_impl(std::int64_t value);

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
/// @tparam Method Tag-type that specifies an implementation
///                of K-Means Initialization algorithm.
/// @tparam Task   Tag-type that specifies the type of the problem to solve. Can
///                be :expr:`task::init`.
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

    /// Creates a new instance of the class with the given :literal:`cluster_count`
    explicit descriptor(std::int64_t cluster_count = 2) {
        set_cluster_count(cluster_count);
    }

    /// The number of clusters k
    /// @invariant :expr:`cluster_count > 0`
    /// @remark default = 2
    std::int64_t get_cluster_count() const {
        return base_t::get_cluster_count();
    }

    auto& set_cluster_count(std::int64_t value) {
        base_t::set_cluster_count_impl(value);
        return *this;
    }

    /// Engine method for the random numbers generator used by the algorithm
    /// @remark default = engine_method::philox4x32x10
    engine_type get_engine_type() const {
        return base_t::get_engine_type();
    }

    auto& set_engine_type(engine_type value) {
        base_t::set_engine_type_impl(value);
        return *this;
    }

    template <typename M = Method, typename = detail::v1::enable_if_not_default_dense<M>>
    auto& get_seed() const {
        return base_t::get_seed();
    }

    template <typename M = Method, typename = detail::v1::enable_if_not_default_dense<M>>
    auto& set_seed(std::int64_t value) {
        base_t::set_seed_impl(value);
        return *this;
    }

    /// Number of attempts to find the best
    /// sample in terms of potential value
    /// If the value is equal to -1, the number
    /// of trials is 2 + int(log(cluster_count))
    /// @invariant :expr:`local_trials > 0` or :expr`local_trials = -1`
    /// @remark default = -1
    template <typename M = Method, typename = detail::v1::enable_if_plus_plus<M>>
    auto& get_local_trials_count() const {
        return base_t::get_local_trials_count();
    }

    template <typename M = Method, typename = detail::v1::enable_if_plus_plus<M>>
    auto& set_local_trials_count(std::int64_t value = -1) {
        base_t::set_local_trials_count_impl(value);
        return *this;
    }
};

} // namespace v1

using v1::descriptor;

} // namespace oneapi::dal::kmeans_init
