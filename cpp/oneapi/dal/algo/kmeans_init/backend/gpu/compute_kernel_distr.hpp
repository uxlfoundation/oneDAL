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

#include "oneapi/dal/algo/kmeans_init/compute_types.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/primitives/rng/host_engine.hpp"

namespace oneapi::dal::kmeans_init::backend {

template <typename Float, typename Method, typename Task>
struct compute_kernel_distr {
    compute_result<Task> operator()(const dal::backend::context_gpu& ctx,
                                    const detail::descriptor_base<Task>& params,
                                    const compute_input<Task>& input) const;
};

namespace misc {
using ctx_t = dal::backend::context_gpu;
using ids_arr_t = array<std::int64_t>;

ids_arr_t generate_random_indices(std::int64_t count, std::int64_t scount, std::int64_t seed = 777);

ids_arr_t generate_random_indices_distr(const ctx_t& ctx,
                                        std::int64_t count,
                                        std::int64_t scount,
                                        std::int64_t rseed = 777);

} // namespace misc

} // namespace oneapi::dal::kmeans_init::backend
