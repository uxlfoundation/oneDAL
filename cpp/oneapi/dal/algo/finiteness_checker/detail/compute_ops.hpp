/*******************************************************************************
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

#include "oneapi/dal/algo/finiteness_checker/compute_types.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/table/homogen.hpp"

namespace oneapi::dal::finiteness_checker::detail {
namespace v1 {

template <typename Context, typename Float, typename Method, typename Task, typename... Options>
struct compute_ops_dispatcher {
    compute_result<Task> operator()(const Context&,
                                    const descriptor_base<Task>& desc,
                                    const compute_input<Task>&) const;

#ifdef ONEDAL_DATA_PARALLEL
    void operator()(const Context&,
                    const descriptor_base<Task>& desc,
                    const table& data,
                    const bool&);
#endif
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
    }

    template <typename Context>
    auto operator()(const Context& ctx, const Descriptor& desc, const input_t& input) const {
        check_preconditions(desc, input);
        const auto result =
            compute_ops_dispatcher<Context, float_t, method_t, task_t>()(ctx, desc, input);
        return result;
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename Context>
    void operator()(const Context& ctx, const Descriptor& desc, const table& data, bool& res) {
        compute_ops_dispatcher<Context, float_t, method_t, task_t>()(ctx, desc, data, res);
    }
#endif
};

} // namespace v1

using v1::compute_ops;

} // namespace oneapi::dal::finiteness_checker::detail
