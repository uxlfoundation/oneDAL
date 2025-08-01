/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/algo/covariance/compute_types.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::covariance::detail {
namespace v1 {

template <typename Context, typename Float, typename Method, typename Task, typename... Options>
struct partial_compute_ops_dispatcher {
    partial_compute_result<Task> operator()(const Context&,
                                            const descriptor_base<Task>&,
                                            const partial_compute_input<Task>&) const;
};

template <typename Descriptor>
struct partial_compute_ops {
    using float_t = typename Descriptor::float_t;
    using method_t = typename Descriptor::method_t;
    using task_t = typename Descriptor::task_t;
    using input_t = partial_compute_input<task_t>;
    using result_t = partial_compute_result<task_t>;

    void check_preconditions(const Descriptor& params, const input_t& input) const {
        using msg = dal::detail::error_messages;

        if (!input.get_data().has_data()) {
            throw domain_error(msg::input_data_is_empty());
        }
    }

    void check_postconditions(const Descriptor& params,
                              const input_t& input,
                              const result_t& result) const {
        ONEDAL_ASSERT(result.get_partial_n_rows().has_data());
        ONEDAL_ASSERT(result.get_partial_n_rows().get_column_count() == 1);
        ONEDAL_ASSERT(result.get_partial_n_rows().get_row_count() == 1);
        ONEDAL_ASSERT(result.get_partial_crossproduct().has_data());
        ONEDAL_ASSERT(result.get_partial_crossproduct().get_column_count() ==
                      input.get_data().get_column_count());
        ONEDAL_ASSERT(result.get_partial_crossproduct().get_row_count() ==
                      input.get_data().get_column_count());
        ONEDAL_ASSERT(result.get_partial_sum().has_data());
        ONEDAL_ASSERT(result.get_partial_sum().get_column_count() ==
                      input.get_data().get_column_count());
        ONEDAL_ASSERT(result.get_partial_sum().get_row_count() == 1);
    }

    template <typename Context>
    auto operator()(const Context& ctx,
                    const Descriptor& desc,
                    const partial_compute_input<task_t>& input) const {
        check_preconditions(desc, input);
        const auto result =
            partial_compute_ops_dispatcher<Context, float_t, method_t, task_t>()(ctx, desc, input);
        check_postconditions(desc, input, result);
        return result;
    }
};

} // namespace v1

using v1::partial_compute_ops;

} // namespace oneapi::dal::covariance::detail
