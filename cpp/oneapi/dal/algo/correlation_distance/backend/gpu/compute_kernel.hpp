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
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/table/homogen.hpp"

namespace oneapi::dal::correlation_distance::backend {

template <typename Float, typename Method, typename Task>
struct compute_kernel_gpu {
    compute_result<Task> operator()(const dal::backend::context_gpu& ctx,
                                    const detail::descriptor_base<Task>& params,
                                    const compute_input<Task>& input) const;

#ifdef ONEDAL_DATA_PARALLEL
    void operator()(const dal::backend::context_gpu& ctx,
                    const detail::descriptor_base<Task>& params,
                    const table& x,
                    const table& y,
                    homogen_table& res);
#endif
};

} // namespace oneapi::dal::correlation_distance::backend
