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

#include "oneapi/dal/algo/kmeans_init/compute_types.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::kmeans_init {

template <typename Task>
class detail::v1::compute_input_impl : public base {
public:
    compute_input_impl(const table& data) : data(data) {}

    table data;
};

template <typename Task>
class detail::v1::compute_result_impl : public base {
public:
    table centroids;
};

using detail::v1::compute_input_impl;
using detail::v1::compute_result_impl;

namespace v1 {

template <typename Task>
compute_input<Task>::compute_input(const table& data) : impl_(new compute_input_impl<Task>(data)) {}

template <typename Task>
compute_input<Task>::~compute_input() {}

template <typename Task>
compute_input<Task>::compute_input(const compute_input& other) : impl_(other.impl_) {}

template <typename Task>
compute_input<Task>::compute_input(compute_input&& other) noexcept
        : impl_(std::move(other.impl_)) {}

template <typename Task>
compute_input<Task>& compute_input<Task>::operator=(const compute_input& other) {
    if (this != &other) {
        compute_input<Task> tmp(other);
        swap(*this, tmp);
    }
    return *this;
}

template <typename Task>
compute_input<Task>& compute_input<Task>::operator=(compute_input&& other) noexcept {
    if (this != &other) {
        swap(*this, other);
    }
    return *this;
}

template <typename Task>
const table& compute_input<Task>::get_data() const {
    return impl_->data;
}

template <typename Task>
void compute_input<Task>::set_data_impl(const table& value) {
    impl_->data = value;
}

template <typename Task>
void compute_input<Task>::swap(compute_input<Task>& a, compute_input<Task>& b) noexcept {
    std::swap(a.impl_, b.impl_);
}

template <typename Task>
compute_result<Task>::compute_result() : impl_(new compute_result_impl<Task>{}) {}

template <typename Task>
const table& compute_result<Task>::get_centroids() const {
    return impl_->centroids;
}

template <typename Task>
void compute_result<Task>::set_centroids_impl(const table& value) {
    impl_->centroids = value;
}

template class ONEDAL_EXPORT compute_input<task::init>;
template class ONEDAL_EXPORT compute_result<task::init>;

} // namespace v1
} // namespace oneapi::dal::kmeans_init
