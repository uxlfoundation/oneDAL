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

#include "oneapi/dal/algo/connected_components/detail/vertex_partitioning_default_kernel.hpp"
#include "oneapi/dal/algo/connected_components/backend/cpu/vertex_partitioning_default_kernel.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::preview::connected_components::detail {

template <typename Float>
vertex_partitioning_result<task::vertex_partitioning>
afforest<Float, task::vertex_partitioning, dal::preview::detail::topology<std::int32_t>>::
operator()(const dal::detail::host_policy& policy,
           const detail::descriptor_base<task::vertex_partitioning>& desc,
           const dal::preview::detail::topology<std::int32_t>& t,
           byte_alloc_iface* alloc_ptr) const {
    return dal::backend::dispatch_by_cpu(dal::backend::context_cpu{ policy }, [&](auto cpu) {
        return backend::afforest<decltype(cpu)>{}(desc, t, alloc_ptr);
    });
}

template struct afforest<float,
                        task::vertex_partitioning,
                        dal::preview::detail::topology<std::int32_t>>;

vertex_partitioning_result<task::vertex_partitioning> run_afforest(
    const dal::detail::host_policy& ctx,
    const detail::descriptor_base<task::vertex_partitioning>& desc,
    const dal::preview::detail::topology<std::int32_t>& t,
    byte_alloc_iface* alloc) {
    return afforest<float,
                    task::vertex_partitioning,
                    dal::preview::detail::topology<std::int32_t>>{}(ctx, desc, t, alloc);
}

} // namespace oneapi::dal::preview::connected_components::detail
