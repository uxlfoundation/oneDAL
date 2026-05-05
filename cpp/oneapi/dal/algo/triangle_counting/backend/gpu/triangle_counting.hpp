#pragma once

#include "oneapi/dal/algo/triangle_counting/common.hpp"
#include "oneapi/dal/algo/triangle_counting/vertex_ranking_types.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::preview::triangle_counting::backend {

template <typename Float, typename Task, typename Topology>
struct vertex_ranking_kernel_gpu {
    vertex_ranking_result<Task> operator()(const dal::backend::context_gpu& ctx,
                                           const detail::descriptor_base<Task>& desc,
                                           const Topology& topology) const;
};

} // namespace oneapi::dal::preview::triangle_counting::backend
