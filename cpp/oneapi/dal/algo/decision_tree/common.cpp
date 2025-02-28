/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/algo/decision_tree/backend/node_info_impl.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::decision_tree {
namespace v1 {

namespace de = dal::detail;

template <typename Task>
node_info<Task>::node_info() : impl_(new impl_t{}) {}

template <typename Task>
node_info<Task>::~node_info() {
    delete impl_;
}

template <typename Task>
node_info<Task>::node_info(impl_t* impl) : impl_(impl) {}

template <typename Task>
node_info<Task>::node_info(const node_info<Task>& orig) : impl_(new impl_t(*orig.impl_)) {}

template <typename Task>
node_info<Task>::node_info(node_info<Task>&& orig) : impl_(orig.impl_) {
    orig.impl_ = nullptr;
}

template <typename Task>
node_info<Task>& node_info<Task>::operator=(const node_info<task_t>& orig) {
    if (&orig == this)
        return *this;

    delete impl_;

    impl_ = new impl_t(*orig.impl_);
    return *this;
}

template <typename Task>
node_info<Task>& node_info<Task>::operator=(node_info<task_t>&& orig) {
    if (&orig == this)
        return *this;

    delete impl_;

    impl_ = orig.impl_;
    orig.impl_ = nullptr;

    return *this;
}

template <typename Task>
std::int64_t node_info<Task>::get_level() const {
    return impl_->level;
}

template <typename Task>
double node_info<Task>::get_impurity() const {
    return impl_->impurity;
}

template <typename Task>
std::int64_t node_info<Task>::get_sample_count() const {
    return impl_->sample_count;
}

template <typename Task>
split_node_info<Task>::split_node_info() : node_info<Task>(new impl_t{}) {}

template <typename Task>
split_node_info<Task>::split_node_info(const split_node_info<Task>& orig)
        : node_info<Task>(new impl_t(de::cast_impl<impl_t>(orig))) {}

template <typename Task>
split_node_info<Task>::split_node_info(split_node_info<Task>&& orig) : node_info<Task>(orig.impl_) {
    orig.impl_ = nullptr;
}

template <typename Task>
split_node_info<Task>& split_node_info<Task>::operator=(const split_node_info<task_t>& orig) {
    if (&orig == this)
        return *this;

    delete this->impl_;

    this->impl_ = new impl_t(de::cast_impl<impl_t>(orig));
    return *this;
}

template <typename Task>
split_node_info<Task>& split_node_info<Task>::operator=(split_node_info<task_t>&& orig) {
    if (&orig == this)
        return *this;

    delete this->impl_;

    this->impl_ = orig.impl_;
    orig.impl_ = nullptr;

    return *this;
}

template <typename Task>
std::int64_t split_node_info<Task>::get_feature_index() const {
    return de::cast_impl<impl_t>(*this).feature_index;
}

template <typename Task>
double split_node_info<Task>::get_feature_value() const {
    return de::cast_impl<impl_t>(*this).feature_value;
}

leaf_node_info<task::classification>::leaf_node_info(std::int64_t class_count)
        : node_info<task::classification>(new impl_t(class_count)) {}

leaf_node_info<task::classification>::leaf_node_info(
    const leaf_node_info<task::classification>& orig)
        : node_info<task::classification>(new impl_t(de::cast_impl<impl_t>(orig))) {}

leaf_node_info<task::classification>::leaf_node_info(leaf_node_info<task::classification>&& orig)
        : node_info<task::classification>(orig.impl_) {
    orig.impl_ = nullptr;
}

leaf_node_info<task::classification>& leaf_node_info<task::classification>::operator=(
    const leaf_node_info<task::classification>& orig) {
    if (&orig == this)
        return *this;

    delete impl_;

    impl_ = new impl_t(de::cast_impl<impl_t>(orig));

    return *this;
}

leaf_node_info<task::classification>& leaf_node_info<task::classification>::operator=(
    leaf_node_info<task::classification>&& orig) {
    if (&orig == this)
        return *this;

    delete impl_;

    impl_ = orig.impl_;
    orig.impl_ = nullptr;

    return *this;
}

std::int64_t leaf_node_info<task::classification>::get_response() const {
    return de::cast_impl<impl_t>(*this).response;
}

double leaf_node_info<task::classification>::get_probability(std::int64_t class_idx) const {
    auto& impl = de::cast_impl<impl_t>(*this);
    return impl.prob && class_idx < impl.class_count ? impl.prob[class_idx] : 0.;
}

leaf_node_info<task::regression>::leaf_node_info() : node_info<task::regression>(new impl_t{}) {}

leaf_node_info<task::regression>::leaf_node_info(const leaf_node_info<task::regression>& orig)
        : node_info<task::regression>(new impl_t(de::cast_impl<impl_t>(orig))) {}

leaf_node_info<task::regression>::leaf_node_info(leaf_node_info<task::regression>&& orig)
        : node_info<task::regression>(orig.impl_) {
    orig.impl_ = nullptr;
}

leaf_node_info<task::regression>& leaf_node_info<task::regression>::operator=(
    const leaf_node_info<task::regression>& orig) {
    if (&orig == this)
        return *this;

    delete impl_;

    impl_ = new impl_t(de::cast_impl<impl_t>(orig));
    return *this;
}

leaf_node_info<task::regression>& leaf_node_info<task::regression>::operator=(
    leaf_node_info<task::regression>&& orig) {
    if (&orig == this)
        return *this;

    delete impl_;

    impl_ = orig.impl_;
    orig.impl_ = nullptr;

    return *this;
}

double leaf_node_info<task::regression>::get_response() const {
    return de::cast_impl<impl_t>(*this).response;
}

template class ONEDAL_EXPORT node_info<task::classification>;
template class ONEDAL_EXPORT node_info<task::regression>;
template class ONEDAL_EXPORT split_node_info<task::classification>;
template class ONEDAL_EXPORT split_node_info<task::regression>;

} // namespace v1
} // namespace oneapi::dal::decision_tree
