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

#include "oneapi/dal/algo/covariance/compute_types.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::covariance::detail {
namespace v1 {

template <typename Context, typename Float, typename Method, typename Task, typename... Options>
struct compute_ops_dispatcher {
    compute_result<Task> operator()(const Context&,
                                    const descriptor_base<Task>&,
                                    const compute_parameters<Task>&,
                                    const compute_input<Task>&) const;
    compute_parameters<Task> select_parameters(const Context&,
                                               const descriptor_base<Task>&,
                                               const compute_input<Task>&) const;
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
    using param_t = compute_parameters<task_t>;
    using descriptor_base_t = descriptor_base<task_t>;

    void check_preconditions(const Descriptor& params, const input_t& input) const {
        using msg = dal::detail::error_messages;

        if (!input.get_data().has_data()) {
            throw domain_error(msg::input_data_is_empty());
        }
    }

    void check_postconditions(const Descriptor& params,
                              const input_t& input,
                              const result_t& result) const {
        if (result.get_result_options().test(result_options::means)) {
            ONEDAL_ASSERT(result.get_means().has_data());
            ONEDAL_ASSERT(result.get_means().get_column_count() ==
                          input.get_data().get_column_count());
            ONEDAL_ASSERT(result.get_means().get_row_count() == 1);
        }

        if (result.get_result_options().test(result_options::cov_matrix)) {
            ONEDAL_ASSERT(result.get_cov_matrix().has_data());
            ONEDAL_ASSERT(result.get_cov_matrix().get_column_count() ==
                          input.get_data().get_column_count());
            ONEDAL_ASSERT(result.get_cov_matrix().get_row_count() ==
                          input.get_data().get_column_count());
        }

        if (result.get_result_options().test(result_options::cor_matrix)) {
            ONEDAL_ASSERT(result.get_cor_matrix().has_data());
            ONEDAL_ASSERT(result.get_cor_matrix().get_column_count() ==
                          input.get_data().get_column_count());
            ONEDAL_ASSERT(result.get_cor_matrix().get_row_count() ==
                          input.get_data().get_column_count());
        }
    }

    /// Check that the hyperparameters of the algorithm belong to the expected ranges
    void check_parameters_ranges(const param_t& params, const input_t& input) const {
        ONEDAL_ASSERT(params.get_cpu_macro_block() > 0);
        ONEDAL_ASSERT(params.get_cpu_grain_size() > 0);
    }

    template <typename Context>
    auto select_parameters(const Context& ctx, const Descriptor& desc, const input_t& input) const {
        check_preconditions(desc, input);
        return compute_ops_dispatcher<Context, float_t, method_t, task_t>{}.select_parameters(
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
            compute_ops_dispatcher<Context, float_t, method_t, task_t>()(ctx, desc, params, input);
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

using v1::compute_ops;

} // namespace oneapi::dal::covariance::detail
