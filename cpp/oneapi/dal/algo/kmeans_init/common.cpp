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

#include "oneapi/dal/algo/kmeans_init/common.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::kmeans_init::detail {
namespace v1 {

template <typename Task>
class descriptor_impl : public base {
public:
    std::int64_t local_trials_count = -1;
    std::int64_t cluster_count = 2;
    // The default engine has been switched from mt2203 to philox for GPU,
    // as philox is more efficient in terms of performance on GPU architectures.
    // Note: Due to this change, some conformance(not critical) tests might fail as a result.
    engine_type df_engine_type = engine_type::philox4x32x10;
    std::int64_t seed = 777;
};

template <typename Task>
descriptor_base<Task>::descriptor_base() : impl_(new descriptor_impl<Task>{}) {}

template <typename Task>
std::int64_t descriptor_base<Task>::get_cluster_count() const {
    return impl_->cluster_count;
}

template <typename Task>
std::int64_t descriptor_base<Task>::get_seed() const {
    return impl_->seed;
}

template <typename Task>
engine_type descriptor_base<Task>::get_engine_type() const {
    return impl_->df_engine_type;
}

template <typename Task>
std::int64_t descriptor_base<Task>::get_local_trials_count() const {
    return impl_->local_trials_count;
}

template <typename Task>
void descriptor_base<Task>::set_cluster_count_impl(std::int64_t value) {
    if (value <= 0) {
        throw domain_error(dal::detail::error_messages::cluster_count_leq_zero());
    }
    impl_->cluster_count = value;
}

template <typename Task>
void descriptor_base<Task>::set_seed_impl(std::int64_t value) {
    impl_->seed = value;
}

template <typename Task>
void descriptor_base<Task>::set_local_trials_count_impl(std::int64_t value) {
    if (value <= 0) {
        throw domain_error(dal::detail::error_messages::cluster_count_leq_zero());
    }
    impl_->local_trials_count = value;
}

template class ONEDAL_EXPORT descriptor_base<task::init>;

} // namespace v1
} // namespace oneapi::dal::kmeans_init::detail
