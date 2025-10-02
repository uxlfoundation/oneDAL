/*******************************************************************************
* Copyright 2020 Intel Corporation
* Copyright contributors to the oneDAL project
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

#include "oneapi/dal/algo/decision_forest/infer_types.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::decision_forest::detail {
namespace v1 {

template <typename Context, typename Float, typename Task, typename Method, typename... Options>
struct infer_ops_dispatcher {
    infer_result<Task> operator()(const Context&,
                                  const descriptor_base<Task>&,
                                  const infer_parameters<Task>&,
                                  const infer_input<Task>&) const;
    infer_result<Task> operator()(const Context&,
                                  const descriptor_base<Task>&,
                                  const infer_input<Task>&) const;
    infer_parameters<Task> select_parameters(const Context&,
                                             const descriptor_base<Task>&,
                                             const infer_input<Task>&) const;
};

template <typename Descriptor>
struct infer_ops {
    using float_t = typename Descriptor::float_t;
    using task_t = typename Descriptor::task_t;
    using method_t = method::by_default;
    using input_t = infer_input<task_t>;
    using result_t = infer_result<task_t>;
    using param_t = infer_parameters<task_t>;

    void check_preconditions(const Descriptor& params, const input_t& input) const {
        using msg = dal::detail::error_messages;

        if (!(input.get_data().has_data())) {
            throw domain_error(msg::input_data_is_empty());
        }
    }

    void check_postconditions(const Descriptor& params,
                              const input_t& input,
                              const result_t& result) const {}

    /// Check that the hyperparameters of the algorithm belong to the expected ranges
    void check_parameters_ranges(const param_t& params, const input_t& input) const {
        ONEDAL_ASSERT(params.get_block_size() > 0x0l);
        ONEDAL_ASSERT(params.get_block_size() <= 0x10000l);
        ONEDAL_ASSERT(params.get_min_trees_for_threading() > 0x0l);
        ONEDAL_ASSERT(params.get_min_number_of_rows_for_vect_seq_compute() >= 0x0l);
        ONEDAL_ASSERT(params.get_scale_factor_for_vect_parallel_compute() > 0.0f);
        ONEDAL_ASSERT(params.get_scale_factor_for_vect_parallel_compute() < 1.0f);
    }

    template <typename Context>
    auto select_parameters(const Context& ctx, const Descriptor& desc, const input_t& input) const {
        check_preconditions(desc, input);
        return infer_ops_dispatcher<Context, float_t, task_t, method_t>{}.select_parameters(ctx,
                                                                                            desc,
                                                                                            input);
    }

    template <typename Context>
    auto operator()(const Context& ctx,
                    const Descriptor& desc,
                    const param_t& params,
                    const input_t& input) const {
        check_preconditions(desc, input);
        const auto result =
            infer_ops_dispatcher<Context, float_t, task_t, method_t>{}(ctx, desc, params, input);
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

using v1::infer_ops;

} // namespace oneapi::dal::decision_forest::detail
