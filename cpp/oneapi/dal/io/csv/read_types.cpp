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

#include "oneapi/dal/io/csv/read_types.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/memory.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/csr.hpp"

namespace oneapi::dal::csv {

template <>
class detail::v1::read_args_impl<table> : public base {
public:
    read_args_impl() {}
};

template <>
class detail::v1::read_args_impl<csr_table> : public base {
public:
    read_args_impl() {}

    std::int64_t feature_count = 0; // Used for csr_table to specify the number of features
};

namespace v1 {

read_args<table>::read_args() : impl_(new detail::read_args_impl<table>()) {}

read_args<csr_table>::read_args(std::int64_t feature_count)
        : impl_(new detail::read_args_impl<csr_table>()) {
    set_feature_count_impl(feature_count);
}

std::int64_t read_args<csr_table>::get_feature_count_impl() const {
    return impl_->feature_count;
}

void read_args<csr_table>::set_feature_count_impl(std::int64_t feature_count) {
    if (feature_count < 0) {
        throw dal::invalid_argument(dal::detail::error_messages::invalid_feature_count());
    }
    impl_->feature_count = feature_count;
}

} // namespace v1
} // namespace oneapi::dal::csv
