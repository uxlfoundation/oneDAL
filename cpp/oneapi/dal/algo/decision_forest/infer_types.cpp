/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/algo/decision_forest/infer_types.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::decision_forest {

namespace detail::v1 {

template <typename Task>
infer_parameters<Task>::infer_parameters() : impl_(new infer_parameters_impl<Task>{}) {}

template <typename Task>
std::int64_t infer_parameters<Task>::get_block_size() const {
    return impl_->block_size;
}

template <typename Task>
void infer_parameters<Task>::set_block_size_impl(std::int64_t val) {
    impl_->block_size = val;
}

template <typename Task>
std::int64_t infer_parameters<Task>::get_min_trees_for_threading() const {
    return impl_->min_trees_for_threading;
}

template <typename Task>
void infer_parameters<Task>::set_min_trees_for_threading_impl(std::int64_t val) {
    impl_->min_trees_for_threading = val;
}

template <typename Task>
std::int64_t infer_parameters<Task>::get_min_number_of_rows_for_vect_seq_compute() const {
    return impl_->min_number_of_rows_for_vect_seq_compute;
}

template <typename Task>
void infer_parameters<Task>::set_min_number_of_rows_for_vect_seq_compute_impl(std::int64_t val) {
    impl_->min_number_of_rows_for_vect_seq_compute = val;
}

template <typename Task>
double infer_parameters<Task>::get_scale_factor_for_vect_parallel_compute() const {
    return impl_->scale_factor_for_vect_parallel_compute;
}

template <typename Task>
void infer_parameters<Task>::set_scale_factor_for_vect_parallel_compute_impl(double val) {
    impl_->scale_factor_for_vect_parallel_compute = val;
}

template <typename Task>
struct infer_parameters_impl : public base {
    std::int64_t block_size = 32l;
    std::int64_t min_trees_for_threading = 100l;
    std::int64_t min_number_of_rows_for_vect_seq_compute = 32l;
    double scale_factor_for_vect_parallel_compute = 0.3f;
};

template <typename Task>
class infer_input_impl : public base {
public:
    infer_input_impl(const model<Task>& trained_model, const table& data)
            : trained_model(trained_model),
              data(data) {}
    model<Task> trained_model;
    table data;
};

template <typename Task>
class infer_result_impl : public base {
public:
    table responses;
    table probabilities;
};

template class ONEDAL_EXPORT infer_parameters<task::classification>;
template class ONEDAL_EXPORT infer_parameters<task::regression>;

} // namespace detail::v1

using detail::v1::infer_parameters;
using detail::v1::infer_input_impl;
using detail::v1::infer_result_impl;

namespace v1 {

template <typename Task>
infer_input<Task>::infer_input(const model<Task>& trained_model, const table& data)
        : impl_(new infer_input_impl<Task>(trained_model, data)) {}

template <typename Task>
const model<Task>& infer_input<Task>::get_model() const {
    return impl_->trained_model;
}

template <typename Task>
const table& infer_input<Task>::get_data() const {
    return impl_->data;
}

template <typename Task>
void infer_input<Task>::set_model_impl(const model<Task>& value) {
    impl_->trained_model = value;
}

template <typename Task>
void infer_input<Task>::set_data_impl(const table& value) {
    impl_->data = value;
}

template <typename Task>
infer_result<Task>::infer_result() : impl_(new infer_result_impl<Task>{}) {}

template <typename Task>
const table& infer_result<Task>::get_responses() const {
    return impl_->responses;
}

template <typename Task>
const table& infer_result<Task>::get_probabilities_impl() const {
    return impl_->probabilities;
}

template <typename Task>
void infer_result<Task>::set_responses_impl(const table& value) {
    impl_->responses = value;
}

template <typename Task>
void infer_result<Task>::set_probabilities_impl(const table& value) {
    impl_->probabilities = value;
}

template class ONEDAL_EXPORT infer_input<task::classification>;
template class ONEDAL_EXPORT infer_input<task::regression>;
template class ONEDAL_EXPORT infer_result<task::classification>;
template class ONEDAL_EXPORT infer_result<task::regression>;

} // namespace v1
} // namespace oneapi::dal::decision_forest
