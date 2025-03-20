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

#include "oneapi/dal/algo/correlation_distance/compute_types.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/table/homogen.hpp"

namespace oneapi::dal::correlation_distance::detail {
namespace v1 {

template <typename Context, typename Float, typename Method, typename Task, typename... Options>
struct compute_ops_dispatcher {
    compute_result<Task> operator()(const Context&,
                                    const descriptor_base<Task>&,
                                    const compute_input<Task>&) const;

#ifdef ONEDAL_DATA_PARALLEL
    void operator()(const Context&,
                    const descriptor_base<Task>&,
                    const table& x,
                    const table& y,
                    homogen_table& res);
#endif
};

template <typename Descriptor>
struct compute_ops {
    using float_t = typename Descriptor::float_t;
    using method_t = typename Descriptor::method_t;
    using task_t = typename Descriptor::task_t;
    using input_t = compute_input<task_t>;
    using result_t = compute_result<task_t>;
    using descriptor_base_t = descriptor_base<task_t>;

    void check_preconditions(const Descriptor& params, const input_t& input) const {
        using msg = dal::detail::error_messages;

        if (!input.get_x().has_data()) {
            throw domain_error(msg::input_x_is_empty());
        }
        if (!input.get_y().has_data()) {
            throw domain_error(msg::input_y_is_empty());
        }
        if (input.get_x().get_column_count() != input.get_y().get_column_count()) {
            throw invalid_argument(msg::input_x_cc_neq_y_cc());
        }
    }

    void check_postconditions(const Descriptor& params,
                              const input_t& input,
                              const result_t& result) const {
        ONEDAL_ASSERT(result.get_values().has_data());
        ONEDAL_ASSERT(input.get_x().get_row_count() == result.get_values().get_row_count());
        ONEDAL_ASSERT(input.get_y().get_row_count() == result.get_values().get_column_count());
    }

    template <typename Context>
    auto operator()(const Context& ctx, const Descriptor& desc, const input_t& input) const {
        check_preconditions(desc, input);
        const auto result =
            compute_ops_dispatcher<Context, float_t, method_t, task_t>()(ctx, desc, input);
        check_postconditions(desc, input, result);
        return result;
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename Context>
    void operator()(const Context& ctx,
                    const Descriptor& desc,
                    const table& x,
                    const table& y,
                    homogen_table& res) {
        compute_ops_dispatcher<Context, float_t, method_t, task_t>()(ctx, desc, x, y, res);
    }
#endif
};

} // namespace v1

using v1::compute_ops;

} // namespace oneapi::dal::correlation_distance::detail
