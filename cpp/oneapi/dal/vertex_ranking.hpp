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

/// @file
/// Contains the definition of the main processing function for vertex
/// ranking family of the algorithms

#pragma once

#include "oneapi/dal/detail/vertex_ranking_ops.hpp"

namespace oneapi::dal::preview {

/// The main processing function for vertex ranking family of the algorithms
template <typename... Args>
auto vertex_ranking(Args &&...args) {
    return detail::vertex_ranking_dispatch(std::forward<Args>(args)...);
}

#ifdef ONEDAL_DATA_PARALLEL
/// Overload that accepts a SYCL queue for GPU execution
///
/// @param q     SYCL queue specifying the target device
/// @param args  Algorithm descriptor and input graph
template <typename... Args>
auto vertex_ranking(sycl::queue& q, Args &&...args) {
    const auto policy = dal::detail::data_parallel_policy{ q };
    return detail::vertex_ranking_dispatch(policy, std::forward<Args>(args)...);
}
#endif

} // namespace oneapi::dal::preview
