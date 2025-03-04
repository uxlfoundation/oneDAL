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

#pragma once

#include "oneapi/dal/algo/decision_forest/train_types.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::decision_forest::detail {
namespace v1 {

template <typename Context, typename Float, typename Task, typename Method, typename... Options>
struct train_ops_dispatcher {
    using input_t = oneapi::dal::decision_forest::v2::train_input<Task>;
    train_result<Task> operator()(const Context&,
                                  const descriptor_base<Task>&,
                                  const train_parameters<Task>&,
                                  const input_t&) const;
    train_result<Task> operator()(const Context&,
                                  const descriptor_base<Task>&,
                                  const input_t&) const;
    train_parameters<Task> select_parameters(const Context&,
                                             const descriptor_base<Task>&,
                                             const input_t&) const;
};

template <typename Descriptor>
struct train_ops {
    using float_t = typename Descriptor::float_t;
    using task_t = typename Descriptor::task_t;
    using method_t = typename Descriptor::method_t;
    using input_t = train_input<task_t>;
    using result_t = train_result<task_t>;
    using param_t = train_parameters<task_t>;
    using descriptor_base_t = descriptor_base<task_t>;

    void check_preconditions(const Descriptor& params, const input_t& input) const {
        using msg = dal::detail::error_messages;

        if (!(input.get_data().has_data())) {
            throw domain_error(msg::input_data_is_empty());
        }
        if (!(input.get_responses().has_data())) {
            throw domain_error(msg::input_responses_are_empty());
        }
        if (input.get_responses().get_column_count() != 1) {
            throw domain_error(msg::input_responses_table_has_wrong_cc_expect_one());
        }
        if (input.get_data().get_row_count() != input.get_responses().get_row_count()) {
            throw invalid_argument(msg::input_data_rc_neq_input_responses_rc());
        }
        if (input.get_weights().has_data() &&
            input.get_data().get_row_count() != input.get_weights().get_row_count()) {
            throw invalid_argument(msg::input_data_rc_neq_input_weights_rc());
        }
        if (!params.get_bootstrap() &&
            (params.get_variable_importance_mode() == variable_importance_mode::mda_raw ||
             params.get_variable_importance_mode() == variable_importance_mode::mda_scaled)) {
            throw invalid_argument(msg::bootstrap_is_incompatible_with_variable_importance_mode());
        }

        if (!params.get_bootstrap() &&
            (check_mask_flag(params.get_error_metric_mode(), error_metric_mode::out_of_bag_error) ||
             check_mask_flag(params.get_error_metric_mode(),
                             error_metric_mode::out_of_bag_error_per_observation))) {
            throw invalid_argument(msg::bootstrap_is_incompatible_with_error_metric());
        }
    }

    void check_postconditions(const Descriptor& params,
                              const input_t& input,
                              const result_t& result) const {}

    template <typename Context>
    auto select_parameters(const Context& ctx, const Descriptor& desc, const input_t& input) const {
        check_preconditions(desc, input);
        return train_ops_dispatcher<Context, float_t, task_t, method_t>{}.select_parameters(ctx,
                                                                                            desc,
                                                                                            input);
    }

    template <typename Context>
    auto operator()(const Context& ctx,
                    const Descriptor& desc,
                    const param_t& params,
                    const input_t& input) const {
        /// Check that the hyperparameters of the algorithm belong to the expected ranges
        params.check_ranges();
        const auto result = train_ops_dispatcher<Context, float_t, task_t, method_t>{}(
            ctx,
            dynamic_cast<const descriptor_base_t&>(desc),
            params,
            input);
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

} // namespace oneapi::dal::decision_forest::detail
