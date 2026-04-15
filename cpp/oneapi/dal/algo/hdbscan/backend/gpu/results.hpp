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

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/algo/hdbscan/common.hpp"
#include "oneapi/dal/algo/hdbscan/compute_types.hpp"

namespace oneapi::dal::hdbscan::backend {

namespace pr = oneapi::dal::backend::primitives;

using descriptor_t = detail::descriptor_base<task::clustering>;
using result_t = compute_result<task::clustering>;

template <typename Float>
inline result_t make_results(sycl::queue& queue,
                             const descriptor_t& desc,
                             const pr::ndarray<std::int32_t, 1>& responses,
                             std::int64_t cluster_count) {
    const std::int64_t row_count = responses.get_dimension(0);
    ONEDAL_ASSERT(row_count > 0);

    auto results =
        result_t().set_cluster_count(cluster_count).set_result_options(desc.get_result_options());

    if (desc.get_result_options().test(result_options::responses)) {
        results.set_responses(dal::homogen_table::wrap(responses.flatten(queue), row_count, 1));
    }

    return results;
}

} // namespace oneapi::dal::hdbscan::backend
