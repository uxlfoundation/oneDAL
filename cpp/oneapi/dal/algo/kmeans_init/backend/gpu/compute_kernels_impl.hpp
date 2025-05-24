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

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/selection/select_indexed_rows.hpp"

#include "oneapi/dal/algo/kmeans_init/common.hpp"

namespace oneapi::dal::kmeans_init::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

using task_t = task::init;
using ctx_t = dal::backend::context_gpu;

template <typename Float, typename Method>
struct kmeans_init_kernel {
    static sycl::event compute_initial_centroids(const ctx_t& ctx,
                                                 const pr::ndview<Float, 2>& data,
                                                 pr::ndview<Float, 2>& centroids) {
        using msg = dal::detail::error_messages;
        if constexpr (std::is_same_v<Method, kmeans_init::method::plus_plus_dense>) {
            throw unimplemented(
                msg::kmeans_init_plus_plus_dense_method_is_not_implemented_for_gpu());
        }

        if constexpr (std::is_same_v<Method, kmeans_init::method::parallel_plus_dense>) {
            throw unimplemented(
                msg::kmeans_init_parallel_plus_dense_method_is_not_implemented_for_gpu());
        }
        return sycl::event();
    }
};

template <typename Float>
struct kmeans_init_kernel<Float, kmeans_init::method::dense> {
    static sycl::event compute_initial_centroids(const ctx_t& ctx,
                                                 const pr::ndview<Float, 2>& data,
                                                 pr::ndview<Float, 2>& centroids) {
        auto& queue = ctx.get_queue();
        ONEDAL_ASSERT(data.get_dimension(1) == centroids.get_dimension(1));
        ONEDAL_ASSERT(data.get_dimension(0) >= centroids.get_dimension(0));
        const std::int64_t cluster_count = centroids.get_dimension(0);
        const std::int64_t column_count = centroids.get_dimension(1);
        dal::detail::check_mul_overflow(cluster_count, column_count);
        const auto data_ptr = data.get_data();
        auto centroids_ptr = centroids.get_mutable_data();
        return bk::copy(queue, centroids_ptr, data_ptr, cluster_count * column_count);
    }
};

#endif

} // namespace oneapi::dal::kmeans_init::backend
