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

#include "oneapi/dal/detail/policy.hpp"

namespace oneapi::dal::preview {
namespace detail {

template <typename Descriptor, typename Graph, typename Tag>
struct vertex_partitioning_ops;

template <typename Descriptor, typename Head, typename... Tail>
auto vertex_partitioning_dispatch_by_input(const Descriptor &desc, Head &&head, Tail &&...tail) {
    using tag_t = typename Descriptor::tag_t;
    using ops_t = vertex_partitioning_ops<Descriptor, std::decay_t<Head>, tag_t>;
    using input_t = typename ops_t::input_t;

    auto input = input_t{ std::forward<Head>(head), std::forward<Tail>(tail)... };
    return ops_t()(dal::detail::host_policy::get_default(), desc, input);
}

template <typename Head, typename... Tail>
auto vertex_partitioning_dispatch(Head &&head, Tail &&...tail) {
    return vertex_partitioning_dispatch_by_input(std::forward<Head>(head),
                                                 std::forward<Tail>(tail)...);
}

#ifdef ONEDAL_DATA_PARALLEL
template <typename Descriptor, typename Head, typename... Tail>
auto vertex_partitioning_dispatch_by_input(const dal::detail::data_parallel_policy &policy,
                                           const Descriptor &desc,
                                           Head &&head,
                                           Tail &&...tail) {
    using tag_t = typename Descriptor::tag_t;
    using ops_t = vertex_partitioning_ops<Descriptor, std::decay_t<Head>, tag_t>;
    using input_t = typename ops_t::input_t;

    auto input = input_t{ std::forward<Head>(head), std::forward<Tail>(tail)... };
    return ops_t()(policy, desc, input);
}

template <typename... Tail>
auto vertex_partitioning_dispatch(const dal::detail::data_parallel_policy &policy, Tail &&...tail) {
    return vertex_partitioning_dispatch_by_input(policy, std::forward<Tail>(tail)...);
}
#endif

} // namespace detail
} // namespace oneapi::dal::preview
