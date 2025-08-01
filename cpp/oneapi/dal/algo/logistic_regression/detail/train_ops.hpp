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

#include "oneapi/dal/algo/logistic_regression/train_types.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::logistic_regression::detail {
namespace v1 {

template <typename Context, typename Float, typename Method, typename Task, typename... Options>
struct train_ops_dispatcher {
    train_result<Task> operator()(const Context&,
                                  const descriptor_base<Task>&,
                                  const train_parameters<Task>&,
                                  const train_input<Task>&) const;
    train_parameters<Task> select_parameters(const Context&,
                                             const descriptor_base<Task>&,
                                             const train_input<Task>&) const;
    train_result<Task> operator()(const Context&,
                                  const descriptor_base<Task>&,
                                  const train_input<Task>&) const;
};

template <typename Descriptor>
struct train_ops {
    using float_t = typename Descriptor::float_t;
    using method_t = typename Descriptor::method_t;
    using task_t = typename Descriptor::task_t;

    using input_t = train_input<task_t>;
    using result_t = train_result<task_t>;
    using param_t = train_parameters<task_t>;

    void check_preconditions(const Descriptor& params, const input_t& input) const {
        using msg = dal::detail::error_messages;

        const auto& data = input.get_data();
        const auto& responses = input.get_responses();

        if (params.get_class_count() != 2) {
            throw domain_error(msg::class_count_neq_two());
        }
        if (params.get_inverse_regularization() <= 0.0) {
            throw domain_error(msg::inverse_regularization_leq_zero());
        }

        if (!data.has_data()) {
            throw domain_error(msg::input_data_is_empty());
        }
        if (!responses.has_data()) {
            throw domain_error(msg::input_responses_are_empty());
        }
        if (data.get_row_count() != responses.get_row_count()) {
            throw domain_error(msg::input_data_rc_neq_input_responses_rc());
        }

        if (responses.get_column_count() != 1) {
            throw domain_error(msg::input_responses_table_has_wrong_cc_expect_one());
        }
    }

    void check_postconditions(const Descriptor& params,
                              const input_t& input,
                              const result_t& result) const {
        const auto& res = params.get_result_options();

        [[maybe_unused]] const std::int64_t f_count = //
            input.get_data().get_column_count();
        [[maybe_unused]] const std::int64_t r_count = //
            input.get_responses().get_column_count();

        ONEDAL_ASSERT(r_count == 1);

        if (res.test(result_options::coefficients)) {
            [[maybe_unused]] const table& coefficients = //
                result.get_coefficients();
            ONEDAL_ASSERT(coefficients.has_data());
            ONEDAL_ASSERT(coefficients.get_row_count() == r_count);
            ONEDAL_ASSERT(coefficients.get_column_count() == f_count);
        }

        if (res.test(result_options::intercept)) {
            [[maybe_unused]] const table& intercept = //
                result.get_intercept();

            ONEDAL_ASSERT(intercept.has_data());
            ONEDAL_ASSERT(intercept.get_row_count() == r_count);
            ONEDAL_ASSERT(intercept.get_column_count() == 1);
        }

        {
            [[maybe_unused]] const table& betas = //
                result.get_packed_coefficients();

            ONEDAL_ASSERT(betas.has_data());
            ONEDAL_ASSERT(betas.get_row_count() == r_count);
            ONEDAL_ASSERT(betas.get_column_count() == f_count + 1);
        }
    }

    template <typename Context>
    auto select_parameters(const Context& ctx, const Descriptor& desc, const input_t& input) const {
        check_preconditions(desc, input);
        return train_ops_dispatcher<Context, float_t, method_t, task_t>{}.select_parameters(ctx,
                                                                                            desc,
                                                                                            input);
    }

    template <typename Context>
    auto operator()(const Context& ctx,
                    const Descriptor& desc,
                    const param_t& params,
                    const input_t& input) const {
        const auto result =
            train_ops_dispatcher<Context, float_t, method_t, task_t>{}(ctx, desc, params, input);
        check_postconditions(desc, input, result);
        return result;
    }

    template <typename Context>
    auto operator()(const Context& ctx, const Descriptor& desc, const input_t& input) const {
        const auto params = select_parameters(ctx, desc, input);
        return this->operator()(ctx, desc, params, input);
    }
};

} // namespace v1

using v1::train_ops;

} // namespace oneapi::dal::logistic_regression::detail
