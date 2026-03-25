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

#include "oneapi/dal/algo/svm/common.hpp"
#include "oneapi/dal/algo/svm/backend/model_impl.hpp"
#include "oneapi/dal/algo/svm/backend/kernel_function_impl.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::svm {

namespace detail {
namespace v1 {

template <typename Task>
class descriptor_impl : public base {
public:
    explicit descriptor_impl(const detail::kernel_function_ptr& kernel) : kernel(kernel) {}

    detail::kernel_function_ptr kernel;
    double c = 1.0;
    double accuracy_threshold = 0.001;
    std::int64_t max_iteration_count = 100000;
    double cache_size = 200.0;
    double tau = 1e-6;
    bool shrinking = true;
    std::int64_t class_count = 2;
    double epsilon = 0.1;
    double nu = 0.5;
};

template <typename Task>
descriptor_base<Task>::descriptor_base(const detail::kernel_function_ptr& kernel)
        : impl_(new descriptor_impl<Task>{ kernel }) {}

template <typename Task>
double descriptor_base<Task>::get_c() const {
    return impl_->c;
}

template <typename Task>
double descriptor_base<Task>::get_accuracy_threshold() const {
    return impl_->accuracy_threshold;
}

template <typename Task>
std::int64_t descriptor_base<Task>::get_max_iteration_count() const {
    return impl_->max_iteration_count;
}

template <typename Task>
double descriptor_base<Task>::get_cache_size() const {
    return impl_->cache_size;
}

template <typename Task>
double descriptor_base<Task>::get_tau() const {
    return impl_->tau;
}

template <typename Task>
bool descriptor_base<Task>::get_shrinking() const {
    return impl_->shrinking;
}

template <typename Task>
void descriptor_base<Task>::set_c_impl(double value) {
    if (value <= 0.0) {
        throw domain_error(dal::detail::error_messages::c_leq_zero());
    }
    impl_->c = value;
}

template <typename Task>
void descriptor_base<Task>::set_accuracy_threshold_impl(double value) {
    if (value < 0.0) {
        throw domain_error(dal::detail::error_messages::accuracy_threshold_lt_zero());
    }
    impl_->accuracy_threshold = value;
}

template <typename Task>
void descriptor_base<Task>::set_max_iteration_count_impl(std::int64_t value) {
    if (value <= 0) {
        throw domain_error(dal::detail::error_messages::max_iteration_count_leq_zero());
    }
    impl_->max_iteration_count = value;
}

template <typename Task>
void descriptor_base<Task>::set_cache_size_impl(double value) {
    if (value < 0.0) {
        throw domain_error(dal::detail::error_messages::cache_size_lt_zero());
    }
    impl_->cache_size = value;
}

template <typename Task>
void descriptor_base<Task>::set_tau_impl(double value) {
    if (value <= 0.0) {
        throw domain_error(dal::detail::error_messages::tau_leq_zero());
    }
    impl_->tau = value;
}

template <typename Task>
void descriptor_base<Task>::set_shrinking_impl(bool value) {
    impl_->shrinking = value;
}

template <typename Task>
void descriptor_base<Task>::set_kernel_impl(const detail::kernel_function_ptr& kernel) {
    impl_->kernel = kernel;
}

template <typename Task>
void descriptor_base<Task>::set_class_count_impl(std::int64_t value) {
    if (value <= 1) {
        throw domain_error(dal::detail::error_messages::class_count_leq_one());
    }
    impl_->class_count = value;
}

template <typename Task>
std::int64_t descriptor_base<Task>::get_class_count_impl() const {
    return impl_->class_count;
}

template <typename Task>
void descriptor_base<Task>::set_epsilon_impl(double value) {
    if (value < 0.0) {
        throw domain_error(dal::detail::error_messages::epsilon_lt_zero());
    }
    impl_->epsilon = value;
}

template <typename Task>
double descriptor_base<Task>::get_epsilon_impl() const {
    return impl_->epsilon;
}

template <typename Task>
void descriptor_base<Task>::set_nu_impl(double value) {
    if (value <= 0.0) {
        throw domain_error(dal::detail::error_messages::nu_leq_zero());
    }
    if (value > 1.0) {
        throw domain_error(dal::detail::error_messages::nu_gt_one());
    }
    impl_->nu = value;
}

template <typename Task>
double descriptor_base<Task>::get_nu_impl() const {
    return impl_->nu;
}

template <typename Task>
const detail::kernel_function_ptr& descriptor_base<Task>::get_kernel_impl() const {
    return impl_->kernel;
}

template class ONEDAL_EXPORT descriptor_base<task::classification>;
template class ONEDAL_EXPORT descriptor_base<task::nu_classification>;
template class ONEDAL_EXPORT descriptor_base<task::regression>;
template class ONEDAL_EXPORT descriptor_base<task::nu_regression>;

} // namespace v1
} // namespace detail

namespace v1 {

using detail::v1::model_impl;

template <typename Task>
model<Task>::model() : impl_(new model_impl<Task>{}) {}

template <typename Task>
model<Task>::model(const std::shared_ptr<model_impl<Task>>& impl) : impl_(impl) {}

template <typename Task>
const table& model<Task>::get_support_vectors() const {
    return impl_->support_vectors;
}

template <typename Task>
const table& model<Task>::get_coeffs() const {
    return impl_->coeffs;
}

template <typename Task>
double model<Task>::get_bias() const {
    return impl_->bias;
}

template <typename Task>
const table& model<Task>::get_biases() const {
    return impl_->biases;
}

template <typename Task>
std::int64_t model<Task>::get_iteration_count() const {
    return impl_->iteration_count;
}

template <typename Task>
std::int64_t model<Task>::get_support_vector_count() const {
    return impl_->support_vectors.get_row_count();
}

template <typename Task>
std::int64_t model<Task>::get_first_class_response() const {
    return impl_->first_class_response;
}

template <typename Task>
std::int64_t model<Task>::get_second_class_response() const {
    return impl_->second_class_response;
}

template <typename Task>
void model<Task>::set_support_vectors_impl(const table& value) {
    impl_->support_vectors = value;
}

template <typename Task>
void model<Task>::set_coeffs_impl(const table& value) {
    impl_->coeffs = value;
}

template <typename Task>
void model<Task>::set_bias_impl(double value) {
    impl_->bias = value;
}

template <typename Task>
void model<Task>::set_biases_impl(const table& value) {
    impl_->biases = value;
}

template <typename Task>
void model<Task>::set_iteration_count_impl(std::int64_t value) {
    impl_->iteration_count = value;
}

template <typename Task>
void model<Task>::set_first_class_response_impl(std::int64_t value) {
    impl_->first_class_response = value;
}

template <typename Task>
void model<Task>::set_second_class_response_impl(std::int64_t value) {
    impl_->second_class_response = value;
}

template <typename Task>
void model<Task>::serialize(dal::detail::output_archive& ar) const {
    dal::detail::serialize_polymorphic_shared(impl_, ar);
}

template <typename Task>
void model<Task>::deserialize(dal::detail::input_archive& ar) {
    dal::detail::deserialize_polymorphic_shared(impl_, ar);
}

template class ONEDAL_EXPORT model<task::classification>;
template class ONEDAL_EXPORT model<task::nu_classification>;
template class ONEDAL_EXPORT model<task::regression>;
template class ONEDAL_EXPORT model<task::nu_regression>;

ONEDAL_REGISTER_SERIALIZABLE(model_impl<task::classification>)
ONEDAL_REGISTER_SERIALIZABLE(model_impl<task::regression>)
ONEDAL_REGISTER_SERIALIZABLE(model_impl<task::nu_classification>)
ONEDAL_REGISTER_SERIALIZABLE(model_impl<task::nu_regression>)
ONEDAL_REGISTER_SERIALIZABLE(backend::model_interop_cls)

} // namespace v1
} // namespace oneapi::dal::svm
