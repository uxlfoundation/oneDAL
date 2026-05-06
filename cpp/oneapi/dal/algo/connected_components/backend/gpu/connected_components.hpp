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

#include "oneapi/dal/algo/connected_components/common.hpp"
#include "oneapi/dal/algo/connected_components/vertex_partitioning_types.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::preview::connected_components::backend {

template <typename Float, typename Task, typename Topology>
struct vertex_partitioning_kernel_gpu {
    vertex_partitioning_result<Task> operator()(const dal::backend::context_gpu& ctx,
                                                const detail::descriptor_base<Task>& desc,
                                                const Topology& topology) const;
};

} // namespace oneapi::dal::preview::connected_components::backend
