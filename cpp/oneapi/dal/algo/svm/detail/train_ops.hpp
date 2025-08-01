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

#include "oneapi/dal/algo/svm/train_types.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::svm::detail {
namespace v1 {

template <typename Context, typename Float, typename Method, typename Task, typename... Options>
struct train_ops_dispatcher {
    train_result<Task> operator()(const Context&,
                                  const descriptor_base<Task>&,
                                  const train_input<Task>&) const;
};

template <typename Descriptor>
struct train_ops {
    using float_t = typename Descriptor::float_t;
    using method_t = typename Descriptor::method_t;
    using task_t = typename Descriptor::task_t;
    using kernel_t = typename Descriptor::kernel_t;
    using input_t = train_input<task_t>;
    using result_t = train_result<task_t>;

    void check_preconditions(const Descriptor& params, const input_t& input) const {
        using msg = dal::detail::error_messages;

        if (!input.get_data().has_data()) {
            throw domain_error(msg::input_data_is_empty());
        }
        if (!input.get_responses().has_data()) {
            throw domain_error(msg::input_responses_are_empty());
        }
        if (input.get_data().get_row_count() != input.get_responses().get_row_count()) {
            throw invalid_argument(msg::input_data_rc_neq_input_responses_rc());
        }
        if (input.get_weights().has_data() &&
            input.get_data().get_row_count() != input.get_weights().get_row_count()) {
            throw invalid_argument(msg::input_data_rc_neq_input_weights_rc());
        }
    }

    void check_postconditions(const Descriptor& params,
                              const input_t& input,
                              const result_t& result) const {
        ONEDAL_ASSERT(result.get_support_vectors().has_data());
        ONEDAL_ASSERT(result.get_support_indices().has_data());
        ONEDAL_ASSERT(result.get_coeffs().has_data());
        ONEDAL_ASSERT(result.get_support_vector_count() <= input.get_data().get_row_count());
        ONEDAL_ASSERT(result.get_support_vectors().get_column_count() ==
                      input.get_data().get_column_count());
        ONEDAL_ASSERT(result.get_support_vectors().get_row_count() ==
                      result.get_support_vector_count());
        ONEDAL_ASSERT(result.get_support_indices().get_row_count() ==
                      result.get_support_vector_count());
        ONEDAL_ASSERT(result.get_coeffs().get_row_count() == result.get_support_vector_count());
    }

    template <typename Context>
    auto operator()(const Context& ctx, const Descriptor& desc, const input_t& input) const {
        check_preconditions(desc, input);
        const auto result =
            train_ops_dispatcher<Context, float_t, method_t, task_t>()(ctx, desc, input);
        check_postconditions(desc, input, result);
        return result;
    }
};

} // namespace v1

using v1::train_ops;

} // namespace oneapi::dal::svm::detail
