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

#include "oneapi/dal/algo/svm/infer_types.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::svm::detail {
namespace v1 {

template <typename Context, typename Float, typename Method, typename Task, typename... Options>
struct infer_ops_dispatcher {
    infer_result<Task> operator()(const Context&,
                                  const descriptor_base<Task>&,
                                  const infer_input<Task>&) const;
};

template <typename Descriptor>
struct infer_ops {
    using float_t = typename Descriptor::float_t;
    using method_t = method::by_default;
    using task_t = typename Descriptor::task_t;
    using input_t = infer_input<task_t>;
    using result_t = infer_result<task_t>;

    void check_preconditions(const Descriptor& params, const input_t& input) const {
        using msg = dal::detail::error_messages;

        if (!input.get_data().has_data()) {
            throw domain_error(msg::input_data_is_empty());
        }
        if (!input.get_model().get_support_vectors().has_data()) {
            throw domain_error(msg::input_model_support_vectors_are_empty());
        }
        if (!input.get_model().get_coeffs().has_data()) {
            throw domain_error(msg::input_model_coeffs_are_empty());
        }
        if (input.get_model().get_support_vectors().get_column_count() !=
            input.get_data().get_column_count()) {
            throw invalid_argument(msg::input_model_support_vectors_cc_neq_input_data_cc());
        }
        if (input.get_model().get_support_vectors().get_row_count() !=
            input.get_model().get_support_vector_count()) {
            throw invalid_argument(
                msg::input_model_support_vectors_rc_neq_input_model_support_vector_count());
        }
        if (input.get_model().get_coeffs().get_row_count() !=
            input.get_model().get_support_vector_count()) {
            throw invalid_argument(
                msg::input_model_coeffs_rc_neq_input_model_support_vector_count());
        }
    }

    void check_postconditions(const Descriptor& params,
                              const input_t& input,
                              const result_t& result) const {
        ONEDAL_ASSERT(result.get_responses().has_data());

        if constexpr (std::is_same_v<task_t, task::classification>) {
            ONEDAL_ASSERT(result.get_decision_function().has_data());

            ONEDAL_ASSERT(result.get_decision_function().get_row_count() ==
                          result.get_responses().get_row_count());
        }
    }

    template <typename Context>
    auto operator()(const Context& ctx, const Descriptor& desc, const input_t& input) const {
        check_preconditions(desc, input);
        const auto result =
            infer_ops_dispatcher<Context, float_t, method_t, task_t>()(ctx, desc, input);
        check_postconditions(desc, input, result);
        return result;
    }
};

} // namespace v1

using v1::infer_ops;

} // namespace oneapi::dal::svm::detail
