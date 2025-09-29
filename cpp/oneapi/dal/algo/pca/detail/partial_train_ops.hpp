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

#include "oneapi/dal/algo/pca/train_types.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::pca::detail {
namespace v1 {

template <typename Context, typename Float, typename Method, typename Task, typename... Options>
struct partial_train_ops_dispatcher {
    partial_train_result<Task> operator()(const Context&,
                                          const descriptor_base<Task>&,
                                          const partial_train_input<Task>&) const;
};

template <typename Descriptor>
struct partial_train_ops {
    using float_t = typename Descriptor::float_t;
    using method_t = typename Descriptor::method_t;
    using task_t = typename Descriptor::task_t;
    using input_t = partial_train_input<task_t>;
    using result_t = partial_train_result<task_t>;

    void check_preconditions(const Descriptor& params, const input_t& input) const {
        using msg = dal::detail::error_messages;

        if (!input.get_data().has_data()) {
            throw domain_error(msg::input_data_is_empty());
        }
    }

    void check_postconditions(const Descriptor& params,
                              const input_t& input,
                              const result_t& result) const {}

    template <typename Context>
    auto operator()(const Context& ctx,
                    const Descriptor& desc,
                    const partial_train_input<task_t>& input) const {
        check_preconditions(desc, input);
        const auto result =
            partial_train_ops_dispatcher<Context, float_t, method_t, task_t>()(ctx, desc, input);
        check_postconditions(desc, input, result);
        return result;
    }
};

} // namespace v1

using v1::partial_train_ops;

} // namespace oneapi::dal::pca::detail
