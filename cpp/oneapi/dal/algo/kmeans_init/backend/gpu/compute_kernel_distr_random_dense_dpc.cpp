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

#include "oneapi/dal/algo/kmeans_init/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/algo/kmeans_init/backend/gpu/compute_kernel_distr.hpp"
#include "oneapi/dal/algo/kmeans_init/backend/gpu/compute_kernels_impl.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::kmeans_init::backend {

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

template <typename Float, typename Method, typename Task>
compute_result<Task> compute_kernel_distr<Float, Method, Task>::operator()(
    const dal::backend::context_gpu& ctx,
    const detail::descriptor_base<Task>& params,
    const compute_input<Task>& input) const {
    auto& queue = ctx.get_queue();

    const auto& data_table = input.get_data();

    const auto sample_count = data_table.get_row_count();
    const auto cluster_count = params.get_cluster_count();
    const auto seed = params.get_seed();
    const auto feature_count = data_table.get_column_count();

    ONEDAL_ASSERT(0 < cluster_count);
    ONEDAL_ASSERT(cluster_count < sample_count);

    auto data = pr::table2ndarray<Float>(queue, data_table, sycl::usm::alloc::device);

    const auto rsize = dal::detail::check_mul_overflow<std::int64_t>(cluster_count, feature_count);
    auto resa = array<Float>::empty(queue, rsize, sycl::usm::alloc::device);
    auto ress =
        pr::ndview<Float, 2>::wrap(resa.get_mutable_data(), { cluster_count, feature_count });

    const auto indices =
        misc::generate_random_indices_distr(ctx, cluster_count, sample_count, seed);
    const auto ndids =
        pr::ndarray<std::int64_t, 1>::wrap(indices.get_data(), { cluster_count }).to_device(queue);

    select_indexed_rows(queue, ndids, data, ress).wait_and_throw();

    auto result = compute_result<Task>{}.set_centroids(
        homogen_table::wrap(resa, cluster_count, feature_count));

    return result;
}

template struct compute_kernel_distr<float, method::random_dense, task::init>;
template struct compute_kernel_distr<double, method::random_dense, task::init>;

namespace misc {

ids_arr_t generate_random_indices(std::int64_t count, std::int64_t scount, std::int64_t seed) {
    ids_arr_t result = ids_arr_t::empty(count);
    auto ndres = pr::ndview<std::int64_t, 1>::wrap(result.get_mutable_data(), { count });
    ONEDAL_ASSERT(count < scount);
    pr::partial_fisher_yates_shuffle(ndres, scount, seed);
    return result;
}

ids_arr_t generate_random_indices_distr(const ctx_t& ctx,
                                        std::int64_t count,
                                        std::int64_t scount,
                                        std::int64_t rseed) {
    auto& comm = ctx.get_communicator();
    const auto rank_count = comm.get_rank_count();

    ids_arr_t root_rand = ids_arr_t::empty(rank_count);

    if (comm.is_root_rank()) {
        const auto maxval = rank_count + 1;
        root_rand = generate_random_indices(rank_count, maxval, rseed);
    }

    {
        ONEDAL_PROFILER_TASK(bcast_root_rand);
        comm.bcast(root_rand).wait();
    }

    ONEDAL_ASSERT(root_rand.get_count() == rank_count);

    const auto seed = root_rand[comm.get_rank()];
    return generate_random_indices(count, scount, seed);
}

} // namespace misc

} // namespace oneapi::dal::kmeans_init::backend
