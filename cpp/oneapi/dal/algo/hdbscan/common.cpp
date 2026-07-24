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

#include "oneapi/dal/algo/hdbscan/common.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::hdbscan {
namespace detail {

result_option_id get_responses_id() {
    return result_option_id::make_by_index(0);
}

result_option_id get_core_observation_indices_id() {
    return result_option_id::make_by_index(1);
}

result_option_id get_core_observations_id() {
    return result_option_id::make_by_index(2);
}

result_option_id get_core_flags_id() {
    return result_option_id::make_by_index(3);
}

result_option_id get_cluster_centers_id() {
    return result_option_id::make_by_index(4);
}

result_option_id get_medoid_centers_id() {
    return result_option_id::make_by_index(5);
}

template <typename Task>
const result_option_id default_result_options = result_options::responses;

namespace v1 {

template <typename Task>
class descriptor_impl : public base {
public:
    std::int64_t min_cluster_size = 5;
    std::int64_t min_samples = 5;
    result_option_id result_options = default_result_options<Task>;
    distance_metric metric = distance_metric::euclidean;
    double degree = 2.0;
    cluster_selection_method cluster_selection = cluster_selection_method::eom;
    bool allow_single_cluster = false;
    double cluster_selection_epsilon = 0.0;
    std::int64_t max_cluster_size = 0;
    double alpha = 1.0;
    store_centers_method store_centers = store_centers_method::none;
    std::int64_t leaf_size = 40;
};

template <typename Task>
descriptor_base<Task>::descriptor_base() : impl_(new descriptor_impl<Task>{}) {}

template <typename Task>
std::int64_t descriptor_base<Task>::get_min_cluster_size() const {
    return impl_->min_cluster_size;
}

template <typename Task>
std::int64_t descriptor_base<Task>::get_min_samples() const {
    return impl_->min_samples;
}

template <typename Task>
void descriptor_base<Task>::set_min_cluster_size_impl(std::int64_t value) {
    if (value < 2) {
        throw domain_error(dal::detail::error_messages::hdbscan_min_cluster_size_lt_two());
    }
    impl_->min_cluster_size = value;
}

template <typename Task>
void descriptor_base<Task>::set_min_samples_impl(std::int64_t value) {
    if (value < 1) {
        throw domain_error(dal::detail::error_messages::hdbscan_min_samples_lt_one());
    }
    impl_->min_samples = value;
}

template <typename Task>
result_option_id descriptor_base<Task>::get_result_options() const {
    return impl_->result_options;
}

template <typename Task>
distance_metric descriptor_base<Task>::get_metric() const {
    return impl_->metric;
}

template <typename Task>
double descriptor_base<Task>::get_degree() const {
    return impl_->degree;
}

template <typename Task>
void descriptor_base<Task>::set_metric_impl(distance_metric value) {
    impl_->metric = value;
}

template <typename Task>
void descriptor_base<Task>::set_degree_impl(double value) {
    if (value <= 0.0) {
        throw domain_error(dal::detail::error_messages::hdbscan_minkowski_degree_leq_zero());
    }
    impl_->degree = value;
}

template <typename Task>
cluster_selection_method descriptor_base<Task>::get_cluster_selection() const {
    return impl_->cluster_selection;
}

template <typename Task>
bool descriptor_base<Task>::get_allow_single_cluster() const {
    return impl_->allow_single_cluster;
}

template <typename Task>
void descriptor_base<Task>::set_cluster_selection_impl(cluster_selection_method value) {
    impl_->cluster_selection = value;
}

template <typename Task>
void descriptor_base<Task>::set_allow_single_cluster_impl(bool value) {
    impl_->allow_single_cluster = value;
}

template <typename Task>
double descriptor_base<Task>::get_cluster_selection_epsilon() const {
    return impl_->cluster_selection_epsilon;
}

template <typename Task>
void descriptor_base<Task>::set_cluster_selection_epsilon_impl(double value) {
    if (value < 0.0) {
        throw domain_error(
            dal::detail::error_messages::hdbscan_cluster_selection_epsilon_lt_zero());
    }
    impl_->cluster_selection_epsilon = value;
}

template <typename Task>
std::int64_t descriptor_base<Task>::get_max_cluster_size() const {
    return impl_->max_cluster_size;
}

template <typename Task>
void descriptor_base<Task>::set_max_cluster_size_impl(std::int64_t value) {
    if (value < 0) {
        throw domain_error(dal::detail::error_messages::hdbscan_max_cluster_size_lt_zero());
    }
    impl_->max_cluster_size = value;
}

template <typename Task>
double descriptor_base<Task>::get_alpha() const {
    return impl_->alpha;
}

template <typename Task>
void descriptor_base<Task>::set_alpha_impl(double value) {
    if (value <= 0.0) {
        throw domain_error(dal::detail::error_messages::hdbscan_alpha_leq_zero());
    }
    impl_->alpha = value;
}

template <typename Task>
store_centers_method descriptor_base<Task>::get_store_centers() const {
    return impl_->store_centers;
}

template <typename Task>
void descriptor_base<Task>::set_store_centers_impl(store_centers_method value) {
    impl_->store_centers = value;
}

template <typename Task>
std::int64_t descriptor_base<Task>::get_leaf_size() const {
    return impl_->leaf_size;
}

template <typename Task>
void descriptor_base<Task>::set_leaf_size_impl(std::int64_t value) {
    if (value < 1) {
        throw domain_error(dal::detail::error_messages::hdbscan_leaf_size_lt_one());
    }
    impl_->leaf_size = value;
}

template <typename Task>
void descriptor_base<Task>::set_result_options_impl(const result_option_id& value) {
    using msg = dal::detail::error_messages;
    if (!bool(value)) {
        throw domain_error(msg::empty_set_of_result_options());
    }
    impl_->result_options = value;
}

template class ONEDAL_EXPORT descriptor_base<task::clustering>;

} // namespace v1
} // namespace detail

} // namespace oneapi::dal::hdbscan
