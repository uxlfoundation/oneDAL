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

#include "oneapi/dal/algo/objective_function/compute_types.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::objective_function::detail {
namespace v1 {

template <typename Context, typename Float, typename Method, typename Task, typename... Options>
struct compute_ops_dispatcher {
    compute_result<Task> operator()(const Context&,
                                    const descriptor_base<Task>&,
                                    const compute_input<Task>&) const;
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

        if (!input.get_data().has_data()) {
            throw domain_error(msg::input_data_is_empty());
        }
        if (!input.get_parameters().has_data()) {
            throw domain_error(msg::input_data_is_empty());
        }
        if (!input.get_responses().has_data()) {
            throw domain_error(msg::input_data_is_empty());
        }
        if (input.get_data().get_row_count() != input.get_responses().get_row_count()) {
            throw domain_error(msg::input_data_rc_neq_input_responses_rc());
        }
        if (input.get_data().get_column_count() + 1 != input.get_parameters().get_row_count()) {
            throw domain_error(msg::input_data_rc_neq_input_weights_rc());
        }
        if (input.get_responses().get_column_count() != 1) {
            throw domain_error(msg::resp_column_count_is_not_eq_to_one());
        }
        if (input.get_parameters().get_column_count() != 1) {
            throw domain_error(msg::params_column_count_is_not_eq_to_one());
        }
    }

    void check_postconditions(const Descriptor& params,
                              const input_t& input,
                              const result_t& result) const {
        using msg = dal::detail::error_messages;
        std::int64_t p = input.get_data().get_column_count();
        if (result.get_result_options().test(result_options::value)) {
            if (!result.get_value().has_data()) {
                throw domain_error(msg::value_is_not_provided());
            }
            if (result.get_value().get_row_count() != 1 ||
                result.get_value().get_column_count() != 1) {
                throw domain_error(msg::incorrect_output_table_size());
            }
        }

        if (result.get_result_options().test(result_options::gradient)) {
            if (!result.get_gradient().has_data()) {
                throw domain_error(msg::gradient_is_not_provided());
            }
            if (result.get_gradient().get_row_count() != (p + 1) ||
                result.get_gradient().get_column_count() != 1) {
                throw domain_error(msg::incorrect_output_table_size());
            }
        }

        if (result.get_result_options().test(result_options::hessian)) {
            if (!result.get_hessian().has_data()) {
                throw domain_error(msg::hessian_is_not_provided());
            }
            if (result.get_hessian().get_row_count() != (p + 1) ||
                result.get_hessian().get_column_count() != (p + 1)) {
                throw domain_error(msg::incorrect_output_table_size());
            }
        }
    }

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

} // namespace oneapi::dal::objective_function::detail
