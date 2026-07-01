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

#include "oneapi/dal/algo/pca/common.hpp"
#include "oneapi/dal/exceptions.hpp"
#include "oneapi/dal/backend/serialization.hpp"

namespace oneapi::dal::pca {
namespace detail {

result_option_id get_eigenvectors_id() {
    return result_option_id{ result_option_id::make_by_index(0) };
}

result_option_id get_eigenvalues_id() {
    return result_option_id{ result_option_id::make_by_index(1) };
}

result_option_id get_variances_id() {
    return result_option_id{ result_option_id::make_by_index(2) };
}

result_option_id get_means_id() {
    return result_option_id{ result_option_id::make_by_index(3) };
}

result_option_id get_singular_values_id() {
    return result_option_id{ result_option_id::make_by_index(4) };
}

result_option_id get_noise_variance_id() {
    return result_option_id{ result_option_id::make_by_index(5) };
}

result_option_id get_explained_variances_ratio_id() {
    return result_option_id{ result_option_id::make_by_index(6) };
}

template <typename Task>
result_option_id get_default_result_options() {
    return result_option_id{};
}

template <>
result_option_id get_default_result_options<task::dim_reduction>() {
    return get_eigenvectors_id() | get_eigenvalues_id() | get_variances_id() | get_means_id() |
           get_singular_values_id() | get_noise_variance_id() | get_explained_variances_ratio_id();
}

namespace v1 {

template <typename Task>
class descriptor_impl : public base {
public:
    std::int64_t component_count = -1;
    bool deterministic = false;
    bool whiten = false;
    normalization normalization_mode = normalization::zscore;
    normalization data_normalization = normalization::none;
    result_option_id result_options = get_default_result_options<Task>();
};

template <typename Task>
class model_impl : public ONEDAL_SERIALIZABLE(pca_dim_reduction_model_impl_id) {
public:
    table eigenvectors;
    table pMeans;
    table pVariances;
    table eigenvalues;
    void serialize(dal::detail::output_archive& ar) const override {
        ar(eigenvectors);
        ar(pMeans);
        ar(pVariances);
        ar(eigenvalues);
    }

    void deserialize(dal::detail::input_archive& ar) override {
        ar(eigenvectors);
        ar(pMeans);
        ar(pVariances);
        ar(eigenvalues);
    }
};

template <typename Task>
descriptor_base<Task>::descriptor_base() : impl_(new descriptor_impl<Task>{}) {}

template <typename Task>
std::int64_t descriptor_base<Task>::get_component_count() const {
    return impl_->component_count;
}

template <typename Task>
bool descriptor_base<Task>::get_deterministic() const {
    return impl_->deterministic;
}

template <typename Task>
bool descriptor_base<Task>::whiten() const {
    return impl_->whiten;
}

template <typename Task>
normalization descriptor_base<Task>::get_normalization_mode() const {
    return impl_->normalization_mode;
}

template <typename Task>
normalization descriptor_base<Task>::get_data_normalization() const {
    return impl_->data_normalization;
}

template <typename Task>
void descriptor_base<Task>::set_component_count_impl(std::int64_t value) {
    if (value < 0) {
        throw domain_error(dal::detail::error_messages::component_count_lt_zero());
    }
    impl_->component_count = value;
}

template <typename Task>
void descriptor_base<Task>::set_deterministic_impl(bool value) {
    impl_->deterministic = value;
}

template <typename Task>
void descriptor_base<Task>::set_normalization_mode_impl(normalization value) {
    impl_->normalization_mode = value;
}
template <typename Task>
void descriptor_base<Task>::set_data_normalization_impl(normalization value) {
    impl_->data_normalization = value;
}

template <typename Task>
void descriptor_base<Task>::set_whiten_impl(bool value) {
    impl_->whiten = value;
}
template <typename Task>
result_option_id descriptor_base<Task>::get_result_options() const {
    return impl_->result_options;
}

template <typename Task>
void descriptor_base<Task>::set_result_options_impl(const result_option_id& value) {
    using msg = dal::detail::error_messages;
    if (!bool(value)) {
        throw domain_error(msg::empty_set_of_result_options());
    }
    impl_->result_options = value;
}

template class ONEDAL_EXPORT descriptor_base<task::dim_reduction>;

} // namespace v1
} // namespace detail

namespace v1 {

using detail::v1::model_impl;

template <typename Task>
model<Task>::model() : impl_(new model_impl<Task>{}) {}

template <typename Task>
const table& model<Task>::get_eigenvectors() const {
    return impl_->eigenvectors;
}

template <typename Task>
void model<Task>::set_eigenvectors_impl(const table& value) {
    impl_->eigenvectors = value;
}
template <typename Task>
const table& model<Task>::get_means() const {
    return impl_->pMeans;
}

template <typename Task>
void model<Task>::set_means_impl(const table& value) {
    impl_->pMeans = value;
}

template <typename Task>
const table& model<Task>::get_variances() const {
    return impl_->pVariances;
}

template <typename Task>
void model<Task>::set_variances_impl(const table& value) {
    impl_->pVariances = value;
}

template <typename Task>
const table& model<Task>::get_eigenvalues() const {
    return impl_->eigenvalues;
}

template <typename Task>
void model<Task>::set_eigenvalues_impl(const table& value) {
    impl_->eigenvalues = value;
}
template <typename Task>
void model<Task>::serialize(dal::detail::output_archive& ar) const {
    dal::detail::serialize_polymorphic_shared(impl_, ar);
}

template <typename Task>
void model<Task>::deserialize(dal::detail::input_archive& ar) {
    dal::detail::deserialize_polymorphic_shared(impl_, ar);
}

template class ONEDAL_EXPORT model<task::dim_reduction>;
ONEDAL_REGISTER_SERIALIZABLE(model_impl<task::dim_reduction>)

} // namespace v1
} // namespace oneapi::dal::pca
