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

#include <sycl/sycl.hpp>

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/selection.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/algo/dbscan/backend/gpu/kernels_fp.hpp"

namespace bk = oneapi::dal::backend;
namespace pr = oneapi::dal::backend::primitives;
namespace spmd = oneapi::dal::preview::spmd;

namespace oneapi::dal::dbscan::backend {

using descriptor_t = detail::descriptor_base<task::clustering>;
using result_t = compute_result<task::clustering>;

template <typename Index>
inline auto output_core_indices(sycl::queue& queue,
                                std::int64_t block_size,
                                std::int64_t core_count,
                                const pr::ndview<Index, 1>& cores,
                                const bk::event_vector& deps = {}) {
    using oneapi::dal::backend::operator+;

    ONEDAL_ASSERT(block_size > 0);
    ONEDAL_ASSERT(core_count > 0);
    ONEDAL_ASSERT(cores.has_data());

    auto [res, res_event] =
        pr::ndarray<Index, 1>::zeros(queue, core_count, sycl::usm::alloc::device);
    auto [err, err_event] = pr::ndarray<bool, 1>::full(queue, 1, false, sycl::usm::alloc::device);

    auto* const err_ptr = err.get_mutable_data();
    auto* const res_ptr = res.get_mutable_data();
    const auto* const cores_ptr = cores.get_data();
    auto full_deps = deps + bk::event_vector{ err_event, res_event };
    auto event = queue.submit([&](sycl::handler& h) {
        h.depends_on(full_deps);
        h.single_task([=]() {
            std::int64_t pos = 0;
            for (std::int64_t i = 0; i < block_size; i++) {
                if (*(cores_ptr + i) > 0) {
                    if (pos < core_count) {
                        *(res_ptr + pos) = i;
                        pos++;
                    }
                    else {
                        *err_ptr = true;
                        break;
                    }
                }
            }
        });
    });

    ONEDAL_ASSERT(!err.to_host(queue, { event }).at(0));
    return std::make_tuple(res, event);
}

template <typename Float, typename Index>
inline auto make_results(sycl::queue& queue,
                         const descriptor_t& desc,
                         const pr::ndarray<Float, 2> data,
                         const pr::ndarray<Index, 1> responses,
                         const pr::ndarray<Index, 1> cores,
                         std::int64_t cluster_count,
                         std::int64_t core_count = -1) {
    const std::int64_t column_count = data.get_dimension(1);
    ONEDAL_ASSERT(column_count > 0);
    const std::int64_t block_size = cores.get_dimension(0);
    ONEDAL_ASSERT(block_size == responses.get_dimension(0));
    auto results =
        result_t().set_cluster_count(cluster_count).set_result_options(desc.get_result_options());

    if (desc.get_result_options().test(result_options::responses)) {
        results.set_responses(dal::homogen_table::wrap(responses.flatten(queue), block_size, 1));
    }
    if (desc.get_result_options().test(result_options::core_flags)) {
        results.set_core_flags(dal::homogen_table::wrap(cores.flatten(queue), block_size, 1));
    }

    bool return_core_indices =
        desc.get_result_options().test(result_options::core_observation_indices);
    bool return_core_observations =
        desc.get_result_options().test(result_options::core_observations);

    if (return_core_indices || return_core_observations) {
        if (core_count == -1) {
            core_count = count_cores(queue, cores);
        }
        if (core_count == 0) {
            if (return_core_indices) {
                results.set_core_observation_indices(dal::homogen_table{});
            }
            if (return_core_observations) {
                results.set_core_observations(dal::homogen_table{});
            }
        }
        else {
            ONEDAL_ASSERT(core_count > 0);
            ONEDAL_ASSERT(block_size >= core_count);

            auto [ids_array, ids_event] = output_core_indices(queue, block_size, core_count, cores);

            if (return_core_indices) {
                results.set_core_observation_indices(
                    dal::homogen_table::wrap(ids_array.flatten(queue, { ids_event }),
                                             core_count,
                                             1));
            }
            if (return_core_observations) {
                auto res = pr::ndarray<Float, 2>::empty(queue,
                                                        { core_count, column_count },
                                                        sycl::usm::alloc::device);

                auto event = pr::select_indexed_rows(queue, ids_array, data, res, { ids_event });

                results.set_core_observations(
                    dal::homogen_table::wrap(res.flatten(queue, { event }),
                                             core_count,
                                             column_count));
            }
        }
    }
    return results;
}

} // namespace oneapi::dal::dbscan::backend
