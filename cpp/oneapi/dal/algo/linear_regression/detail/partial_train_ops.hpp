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

#include "oneapi/dal/algo/linear_regression/train_types.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::linear_regression::detail {
namespace v1 {

template <typename Context, typename Float, typename Method, typename Task, typename... Options>
struct partial_train_ops_dispatcher {
    partial_train_result<Task> operator()(const Context&,
                                          const descriptor_base<Task>&,
                                          const train_parameters<Task>&,
                                          const partial_train_input<Task>&) const;
    train_parameters<Task> select_parameters(const Context&,
                                             const descriptor_base<Task>&,
                                             const partial_train_input<Task>&) const;
    partial_train_result<Task> operator()(const Context&,
                                          const descriptor_base<Task>&,
                                          const partial_train_input<Task>&) const;
};

template <typename Descriptor>
struct partial_train_ops {
    using float_t = typename Descriptor::float_t;
    using method_t = typename Descriptor::method_t;
    using task_t = typename Descriptor::task_t;
    using param_t = train_parameters<task_t>;
    using input_t = partial_train_input<task_t>;
    using result_t = partial_train_result<task_t>;

    void check_preconditions(const Descriptor& params, const input_t& input) const {
        using msg = dal::detail::error_messages;

        const auto& data = input.get_data();
        const auto& responses = input.get_responses();

        if (!data.has_data()) {
            throw domain_error(msg::input_data_is_empty());
        }
        if (!responses.has_data()) {
            throw domain_error(msg::input_responses_are_empty());
        }
        if (data.get_row_count() != responses.get_row_count()) {
            throw domain_error(msg::input_data_rc_neq_input_responses_rc());
        }
    }

    void check_postconditions(const Descriptor& params,
                              const input_t& input,
                              const result_t& result) const {
        using msg = dal::detail::error_messages;

        const auto& partial_xtx = result.get_partial_xtx();
        const auto& partial_xty = result.get_partial_xty();

        if (!partial_xtx.has_data()) {
            throw domain_error(msg::input_data_is_empty());
        }
        if (!partial_xty.has_data()) {
            throw domain_error(msg::input_data_is_empty());
        }
    }

    /// Check that the hyperparameters of the algorithm belong to the expected ranges
    void check_parameters_ranges(const param_t& params, const input_t& input) const {
        ONEDAL_ASSERT(params.get_cpu_macro_block() > 0);
        ONEDAL_ASSERT(params.get_cpu_macro_block() <= 0x10000l);
        ONEDAL_ASSERT(params.get_gpu_macro_block() > 0);
        ONEDAL_ASSERT(params.get_gpu_macro_block() <= 0x100000l);
    }

    template <typename Context>
    auto select_parameters(const Context& ctx, const Descriptor& desc, const input_t& input) const {
        check_preconditions(desc, input);
        return partial_train_ops_dispatcher<Context, float_t, method_t, task_t>{}.select_parameters(
            ctx,
            desc,
            input);
    }

    template <typename Context>
    auto operator()(const Context& ctx,
                    const Descriptor& desc,
                    const param_t& params,
                    const input_t& input) const {
        check_parameters_ranges(params, input);
        const auto result =
            partial_train_ops_dispatcher<Context, float_t, method_t, task_t>{}(ctx,
                                                                               desc,
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

using v1::partial_train_ops;

} // namespace oneapi::dal::linear_regression::detail
