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

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/util/result_option_id.hpp"

namespace oneapi::dal::hdbscan {

namespace v1 {

/// Distance metric used for computing pairwise distances in HDBSCAN.
enum class distance_metric {
    euclidean, ///< Euclidean (L2) distance
    manhattan, ///< Manhattan (L1) distance
    minkowski, ///< Minkowski (Lp) distance with configurable degree
    chebyshev, ///< Chebyshev (L-infinity) distance
    cosine ///< Cosine distance (1 - cosine_similarity). Brute-force only.
};

/// Method used to select clusters from the condensed tree.
enum class cluster_selection_method {
    eom, ///< Excess of Mass — selects clusters maximizing stability (default)
    leaf ///< Select all leaf nodes in the condensed tree (finer-grained clusters)
};

/// Method for computing and storing cluster centers.
enum class store_centers_method {
    none, ///< Do not compute cluster centers (default)
    centroid, ///< Compute centroids (weighted mean of member points)
    medoid, ///< Compute medoids (member point minimizing intra-cluster distance)
    both ///< Compute both centroids and medoids
};

} // namespace v1

using v1::distance_metric;
using v1::cluster_selection_method;
using v1::store_centers_method;

namespace task {
namespace v1 {
/// Tag-type that parameterizes entities used for solving
/// :capterm:`clustering problem <clustering>`.
struct clustering {};

/// Alias tag-type for the clustering task.
using by_default = clustering;
} // namespace v1

using v1::clustering;
using v1::by_default;

} // namespace task

namespace method {
namespace v1 {
struct brute_force {};
struct kd_tree {};
struct ball_tree {};

using by_default = brute_force;
} // namespace v1

using v1::brute_force;
using v1::kd_tree;
using v1::ball_tree;
using v1::by_default;

} // namespace method

/// Represents result option flag
/// Behaves like a regular :expr`enum`.
class result_option_id : public result_option_id_base {
public:
    result_option_id() : result_option_id_base{} {}
    result_option_id(const result_option_id_base& base) : result_option_id_base{ base } {}
};

namespace detail {

ONEDAL_EXPORT result_option_id get_responses_id();
ONEDAL_EXPORT result_option_id get_core_observation_indices_id();
ONEDAL_EXPORT result_option_id get_core_observations_id();
ONEDAL_EXPORT result_option_id get_core_flags_id();
ONEDAL_EXPORT result_option_id get_cluster_centers_id();
ONEDAL_EXPORT result_option_id get_medoid_centers_id();

} // namespace detail

/// Result options are used to define
/// what should algorithm return
namespace result_options {

const inline result_option_id responses = detail::get_responses_id();
const inline result_option_id core_observation_indices = detail::get_core_observation_indices_id();
const inline result_option_id core_observations = detail::get_core_observations_id();
const inline result_option_id core_flags = detail::get_core_flags_id();
const inline result_option_id cluster_centers = detail::get_cluster_centers_id();
const inline result_option_id medoid_centers = detail::get_medoid_centers_id();

} // namespace result_options

namespace detail {
namespace v1 {
struct descriptor_tag {};

template <typename Task>
class descriptor_impl;

template <typename Float>
constexpr bool is_valid_float_v = dal::detail::is_one_of_v<Float, float, double>;

template <typename Method>
constexpr bool is_valid_method_v =
    dal::detail::is_one_of_v<Method, method::brute_force, method::kd_tree, method::ball_tree>;

template <typename Task>
constexpr bool is_valid_task_v = dal::detail::is_one_of_v<Task, task::clustering>;

template <typename Task = task::by_default>
class descriptor_base : public base {
    static_assert(is_valid_task_v<Task>);

public:
    using tag_t = descriptor_tag;
    using method_t = method::by_default;
    using task_t = Task;

    descriptor_base();

    std::int64_t get_min_cluster_size() const;
    std::int64_t get_min_samples() const;
    result_option_id get_result_options() const;
    distance_metric get_metric() const;
    double get_degree() const;
    cluster_selection_method get_cluster_selection() const;
    bool get_allow_single_cluster() const;
    double get_cluster_selection_epsilon() const;
    std::int64_t get_max_cluster_size() const;
    double get_alpha() const;
    store_centers_method get_store_centers() const;
    std::int64_t get_leaf_size() const;

protected:
    void set_min_cluster_size_impl(std::int64_t);
    void set_min_samples_impl(std::int64_t);
    void set_result_options_impl(const result_option_id& value);
    void set_metric_impl(distance_metric);
    void set_degree_impl(double);
    void set_cluster_selection_impl(cluster_selection_method);
    void set_allow_single_cluster_impl(bool);
    void set_cluster_selection_epsilon_impl(double);
    void set_max_cluster_size_impl(std::int64_t);
    void set_alpha_impl(double);
    void set_store_centers_impl(store_centers_method);
    void set_leaf_size_impl(std::int64_t);

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
///                be :expr:`method::brute_force` or :expr:`method::kd_tree`.
/// @tparam Task   Tag-type that specifies the type of the problem to solve. Can
///                be :expr:`task::clustering`.
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

    /// Creates a new instance of the class with the given :literal:`min_cluster_size`, :literal:`min_samples`
    explicit descriptor(std::int64_t min_cluster_size = 5, std::int64_t min_samples = 5) {
        set_min_cluster_size(min_cluster_size);
        set_min_samples(min_samples);
    }

    /// The minimum cluster size
    /// @invariant :expr:`min_cluster_size >= 2`
    std::int64_t get_min_cluster_size() const {
        return base_t::get_min_cluster_size();
    }

    auto& set_min_cluster_size(std::int64_t value) {
        base_t::set_min_cluster_size_impl(value);
        return *this;
    }

    /// The number of samples in a neighborhood for a point to be considered a core point
    /// @invariant :expr:`min_samples >= 1`
    std::int64_t get_min_samples() const {
        return base_t::get_min_samples();
    }

    auto& set_min_samples(std::int64_t value) {
        base_t::set_min_samples_impl(value);
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

    /// The distance metric used for pairwise distance computation.
    /// @invariant Cosine metric is only supported with brute_force method.
    distance_metric get_metric() const {
        return base_t::get_metric();
    }

    auto& set_metric(distance_metric value) {
        base_t::set_metric_impl(value);
        return *this;
    }

    /// The degree parameter for Minkowski distance.
    /// @invariant :expr:`degree > 0`
    double get_degree() const {
        return base_t::get_degree();
    }

    auto& set_degree(double value) {
        base_t::set_degree_impl(value);
        return *this;
    }

    /// The cluster selection method: eom (Excess of Mass) or leaf.
    cluster_selection_method get_cluster_selection() const {
        return base_t::get_cluster_selection();
    }

    auto& set_cluster_selection(cluster_selection_method value) {
        base_t::set_cluster_selection_impl(value);
        return *this;
    }

    /// Whether to allow a single cluster result.
    /// When false (default), if only the root cluster is selected,
    /// its children are selected instead.
    bool get_allow_single_cluster() const {
        return base_t::get_allow_single_cluster();
    }

    auto& set_allow_single_cluster(bool value) {
        base_t::set_allow_single_cluster_impl(value);
        return *this;
    }

    /// Distance threshold below which clusters are merged after selection.
    /// @invariant :expr:`cluster_selection_epsilon >= 0`
    double get_cluster_selection_epsilon() const {
        return base_t::get_cluster_selection_epsilon();
    }

    auto& set_cluster_selection_epsilon(double value) {
        base_t::set_cluster_selection_epsilon_impl(value);
        return *this;
    }

    /// Maximum cluster size for EOM selection. 0 means no limit.
    /// @invariant :expr:`max_cluster_size >= 0`
    std::int64_t get_max_cluster_size() const {
        return base_t::get_max_cluster_size();
    }

    auto& set_max_cluster_size(std::int64_t value) {
        base_t::set_max_cluster_size_impl(value);
        return *this;
    }

    /// Distance scaling parameter for robust single linkage.
    /// @invariant :expr:`alpha > 0`
    double get_alpha() const {
        return base_t::get_alpha();
    }

    auto& set_alpha(double value) {
        base_t::set_alpha_impl(value);
        return *this;
    }

    /// Which cluster centers to compute and store: none, centroid, medoid, or both.
    store_centers_method get_store_centers() const {
        return base_t::get_store_centers();
    }

    auto& set_store_centers(store_centers_method value) {
        base_t::set_store_centers_impl(value);
        return *this;
    }

    /// Leaf size for kd_tree nearest neighbor queries.
    /// @invariant :expr:`leaf_size >= 1`
    std::int64_t get_leaf_size() const {
        return base_t::get_leaf_size();
    }

    auto& set_leaf_size(std::int64_t value) {
        base_t::set_leaf_size_impl(value);
        return *this;
    }
};

} // namespace v1

using v1::descriptor;

} // namespace oneapi::dal::hdbscan
