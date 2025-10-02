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

#pragma once

#include "oneapi/dal/algo/basic_statistics/compute_types.hpp"
#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::basic_statistics::detail {
namespace v1 {

template <typename Context, typename Float, typename Method, typename Task, typename... Options>
struct compute_ops_dispatcher {
    using input_t = compute_input<Task>;
    compute_result<Task> operator()(const Context&,
                                    const descriptor_base<Task>&,
                                    const input_t&) const;
};

template <typename Descriptor>
struct compute_ops {
    using float_t = typename Descriptor::float_t;
    using method_t = typename Descriptor::method_t;
    using task_t = typename Descriptor::task_t;
    using input_t = compute_input<task_t>;
    using result_t = compute_result<task_t>;

    void check_preconditions(const Descriptor& params, const input_t& input) const {
        using msg = dal::detail::error_messages;

        const auto& data = input.get_data();
        if (!data.has_data()) {
            throw domain_error(msg::input_data_is_empty());
        }

        const auto& weights = input.get_weights();
        if (weights.has_data()) {
            const auto r_count = weights.get_row_count();
            if (r_count != data.get_row_count())
                throw domain_error(msg::weight_dimension_doesnt_match_data_dimension());

            const auto c_count = weights.get_column_count();
            if (c_count != std::int64_t(1))
                throw domain_error(msg::weights_column_count_ne_1());
        }
    }

    void check_postconditions(const Descriptor& params,
                              const input_t& input,
                              const result_t& result) const {}

    template <typename Context>
    auto operator()(const Context& ctx, const Descriptor& desc, const input_t& input) const {
        check_preconditions(desc, input);
        const auto result =
            compute_ops_dispatcher<Context, float_t, method_t, task_t>()(ctx, desc, input);
        check_postconditions(desc, input, result);
        return result;
    }
};

} // namespace v1

using v1::compute_ops;

} // namespace oneapi::dal::basic_statistics::detail
